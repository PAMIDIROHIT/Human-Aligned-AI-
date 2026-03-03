"""Stage 3 PPO -- Training script with PPOTrainer.

Implements PPO policy optimization against the frozen reward model.
The actor is the SFT checkpoint, the reward is from Stage 2 RM,
and KL divergence is penalized against the frozen SFT reference.

Hardware target: 4x Tesla K80 (11 GB each, CC 3.7, fp32 only)
Multi-GPU strategy: actor on GPU 0-1, reward on GPU 2, ref on GPU 3
Compatible with: transformers==4.38.2, trl==0.7.11, peft==0.7.1

Usage:
    python -m src.stage3_ppo.train --config params.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType
from prometheus_client import Gauge, start_http_server
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from src.stage3_ppo.dataset import load_ppo_dataset, collator
from src.stage3_ppo.kl_controller import AdaptiveKLController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---- Prometheus Metrics (exposed on GPU server for remote laptop scraping) ----
PROM_MEAN_REWARD = Gauge("ppo_mean_reward", "PPO mean reward per step")
PROM_KL_DIV = Gauge("ppo_kl_divergence", "PPO KL divergence per step")
PROM_POLICY_ENTROPY = Gauge("ppo_policy_entropy", "PPO policy entropy per step")
PROM_VALUE_LOSS = Gauge("ppo_value_loss", "PPO value function loss per step")
PROM_PG_LOSS = Gauge("ppo_pg_loss", "PPO policy gradient loss per step")
PROM_CLIP_FRACTION = Gauge("ppo_clip_fraction", "PPO clip fraction per step")
PROM_KL_COEF = Gauge("ppo_kl_coef", "PPO KL penalty coefficient (beta)")


@dataclass
class PPOTrainConfig:
    """Configuration for PPO training, parsed from params.yaml."""

    base_model: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    seed: int = 42
    max_seq_length: int = 512
    sft_adapter_dir: str = "models/sft_adapter"
    reward_model_dir: str = "models/reward_model"
    output_dir: str = "models/ppo_policy"
    experiment_name: str = "rlhf-ppo"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # PPO config (K80-sized batches)
    learning_rate: float = 1.41e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    optimize_cuda_cache: bool = True
    early_stopping: bool = True
    target_kl: float = 0.1
    kl_penalty: str = "kl"
    use_score_scaling: bool = True
    use_score_norm: bool = True
    ppo_epochs: int = 4
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    max_grad_norm: float = 0.5

    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = True

    # KL controller
    init_kl_coef: float = 0.2
    kl_lr: float = 0.1
    kl_horizon: int = 10000

    # GAE
    gamma: float = 1.0
    lam: float = 0.95

    # Training loop
    total_steps: int = 2000
    log_every_n_steps: int = 10
    save_every_n_steps: int = 500
    eval_every_n_steps: int = 100

    # Gate
    max_kl: float = 0.15
    min_reward_improvement: float = 0.1

    # Monitoring
    metrics_port: int = 9091

    # Full config dict
    _raw_config: dict = field(default_factory=dict)


def load_config(config_path: str) -> PPOTrainConfig:
    """Load PPO configuration from params.yaml.

    Args:
        config_path: Path to the params.yaml file.

    Returns:
        A PPOTrainConfig dataclass populated from the YAML file.
    """
    with open(config_path) as f:
        params = yaml.safe_load(f)

    g = params.get("global", {})
    ppo = params.get("ppo", {})
    c = ppo.get("config", {})
    gen = ppo.get("generation", {})
    kl = ppo.get("kl_controller", {})
    gae = ppo.get("gae", {})
    t = ppo.get("training", {})
    e = ppo.get("eval_gate", {})
    sft = params.get("sft", {})
    rm = params.get("reward_model", {})
    mon = params.get("monitoring", {})

    _d = PPOTrainConfig()

    return PPOTrainConfig(
        base_model=g.get("base_model", _d.base_model),
        seed=g.get("seed", _d.seed),
        max_seq_length=g.get("max_seq_length", _d.max_seq_length),
        sft_adapter_dir=sft.get("output_dir", _d.sft_adapter_dir),
        reward_model_dir=rm.get("output_dir", _d.reward_model_dir),
        output_dir=ppo.get("output_dir", _d.output_dir),
        experiment_name=ppo.get("experiment_name", _d.experiment_name),
        mlflow_tracking_uri=g.get("mlflow_tracking_uri", _d.mlflow_tracking_uri),
        learning_rate=c.get("learning_rate", _d.learning_rate),
        batch_size=c.get("batch_size", _d.batch_size),
        mini_batch_size=c.get("mini_batch_size", _d.mini_batch_size),
        gradient_accumulation_steps=c.get("gradient_accumulation_steps", _d.gradient_accumulation_steps),
        optimize_cuda_cache=c.get("optimize_cuda_cache", _d.optimize_cuda_cache),
        early_stopping=c.get("early_stopping", _d.early_stopping),
        target_kl=c.get("target_kl", _d.target_kl),
        kl_penalty=c.get("kl_penalty", _d.kl_penalty),
        use_score_scaling=c.get("use_score_scaling", _d.use_score_scaling),
        use_score_norm=c.get("use_score_norm", _d.use_score_norm),
        ppo_epochs=c.get("ppo_epochs", _d.ppo_epochs),
        cliprange=c.get("cliprange", _d.cliprange),
        cliprange_value=c.get("cliprange_value", _d.cliprange_value),
        vf_coef=c.get("vf_coef", _d.vf_coef),
        max_grad_norm=c.get("max_grad_norm", _d.max_grad_norm),
        max_new_tokens=gen.get("max_new_tokens", _d.max_new_tokens),
        temperature=gen.get("temperature", _d.temperature),
        top_k=gen.get("top_k", _d.top_k),
        top_p=gen.get("top_p", _d.top_p),
        do_sample=gen.get("do_sample", _d.do_sample),
        init_kl_coef=kl.get("init_kl_coef", _d.init_kl_coef),
        kl_lr=kl.get("kl_lr", _d.kl_lr),
        kl_horizon=kl.get("horizon", _d.kl_horizon),
        gamma=gae.get("gamma", _d.gamma),
        lam=gae.get("lam", _d.lam),
        total_steps=t.get("total_steps", _d.total_steps),
        log_every_n_steps=t.get("log_every_n_steps", _d.log_every_n_steps),
        save_every_n_steps=t.get("save_every_n_steps", _d.save_every_n_steps),
        eval_every_n_steps=t.get("eval_every_n_steps", _d.eval_every_n_steps),
        max_kl=e.get("max_kl", _d.max_kl),
        min_reward_improvement=e.get("min_reward_improvement", _d.min_reward_improvement),
        metrics_port=mon.get("metrics_port", _d.metrics_port),
        _raw_config=params,
    )


def load_reward_model_for_scoring(
    model_dir: str,
    device: str = "cuda:2",
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the frozen reward model for scoring PPO rollouts.

    Places on a specific GPU to spread load across 4 K80s.

    Args:
        model_dir: Path to the saved reward model directory.
        device: Target GPU device for the reward model.

    Returns:
        Tuple of (reward_model, tokenizer), both frozen.
    """
    logger.info("Loading frozen reward model from: %s -> %s", model_dir, device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        num_labels=1,
    )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Reward model loaded and frozen on %s", device)
    return model, tokenizer


def score_responses(
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    queries: list[str],
    responses: list[str],
    max_length: int = 512,
) -> list[torch.Tensor]:
    """Score query-response pairs using the reward model.

    Args:
        reward_model: The frozen reward model.
        reward_tokenizer: Tokenizer for the reward model.
        queries: List of prompt strings.
        responses: List of response strings.
        max_length: Maximum sequence length.

    Returns:
        List of scalar reward tensors.
    """
    device = next(reward_model.parameters()).device
    rewards = []

    for query, response in zip(queries, responses):
        full_text = query + response
        encodings = reward_tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = reward_model(input_ids=input_ids, attention_mask=attention_mask)
            reward = outputs.logits.squeeze(-1).squeeze(-1)

        rewards.append(reward.cpu().detach())

    return rewards


def get_gpu_device_map(num_gpus: int) -> dict:
    """Create device map for multi-GPU model placement.

    Strategy for 4x K80 (11 GB each):
    - PPO actor: GPU 0 (primary training)
    - Reference: GPU 1 or share
    - Reward: GPU 2
    - Buffer: GPU 3

    Args:
        num_gpus: Number of available GPUs.

    Returns:
        Device map dict or 'auto'.
    """
    if num_gpus >= 2:
        return "auto"
    return {"": 0}


def train_ppo(cfg: PPOTrainConfig) -> None:
    """Execute the full PPO training pipeline on 4x K80 GPUs.

    Args:
        cfg: PPO training configuration.
    """
    set_seed(cfg.seed)
    logger.info("Random seed set to %d", cfg.seed)

    num_gpus = torch.cuda.device_count()
    logger.info("Available GPUs: %d", num_gpus)

    # Start Prometheus metrics server (exposed for laptop to scrape)
    try:
        start_http_server(int(cfg.metrics_port))
        logger.info("Prometheus metrics server started on port %d (bind 0.0.0.0)", cfg.metrics_port)
    except Exception as e:
        logger.warning("Could not start Prometheus server: %s", e)

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name="ppo-training") as run:
        logger.info("MLflow run ID: %s", run.info.run_id)

        mlflow.log_params({
            "base_model": cfg.base_model,
            "seed": cfg.seed,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "mini_batch_size": cfg.mini_batch_size,
            "ppo_epochs": cfg.ppo_epochs,
            "cliprange": cfg.cliprange,
            "target_kl": cfg.target_kl,
            "init_kl_coef": cfg.init_kl_coef,
            "gamma": cfg.gamma,
            "lam": cfg.lam,
            "total_steps": cfg.total_steps,
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "precision": "fp32",
            "num_gpus": num_gpus,
        })

        # Load PPO prompts from local disk
        logger.info("Loading PPO prompt dataset from local disk...")
        prompt_dataset = load_ppo_dataset(
            config=cfg._raw_config,
            seed=cfg.seed,
        )
        mlflow.log_metric("num_prompts", len(prompt_dataset))

        # Load tokenizer from SFT adapter
        logger.info("Loading tokenizer from SFT adapter: %s", cfg.sft_adapter_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.sft_adapter_dir, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # LoRA config for PPO fine-tuning (smaller rank for memory)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # PPO Config (TRL 0.7.11 API)
        ppo_config = PPOConfig(
            model_name=cfg.base_model,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            mini_batch_size=cfg.mini_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            optimize_cuda_cache=cfg.optimize_cuda_cache,
            early_stopping=cfg.early_stopping,
            target_kl=cfg.target_kl,
            kl_penalty=cfg.kl_penalty,
            seed=cfg.seed,
            use_score_scaling=cfg.use_score_scaling,
            use_score_norm=cfg.use_score_norm,
            ppo_epochs=cfg.ppo_epochs,
            cliprange=cfg.cliprange,
            cliprange_value=cfg.cliprange_value,
            vf_coef=cfg.vf_coef,
            log_with=None,
        )

        # Load policy model (SFT checkpoint + value head, fp32)
        logger.info("Loading policy model with value head (fp32)...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.sft_adapter_dir,
            device_map=get_gpu_device_map(num_gpus),
            trust_remote_code=True,
            peft_config=lora_config,
            torch_dtype=torch.float32,
        )

        # Load reference model (frozen SFT for KL computation)
        logger.info("Loading reference model (frozen SFT, fp32)...")
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.sft_adapter_dir,
            device_map=get_gpu_device_map(num_gpus),
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        # Load frozen reward model on separate GPU if available
        reward_device = f"cuda:{min(2, num_gpus - 1)}" if num_gpus > 1 else "cuda:0"
        logger.info("Loading frozen reward model on %s...", reward_device)
        reward_model, reward_tokenizer = load_reward_model_for_scoring(
            cfg.reward_model_dir,
            device=reward_device,
        )

        # Create PPOTrainer (TRL 0.7.11 API)
        logger.info("Initializing PPOTrainer...")
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=prompt_dataset,
            data_collator=collator,
        )

        # Generation config
        generation_kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "do_sample": cfg.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }

        # Adaptive KL controller
        kl_controller = AdaptiveKLController(
            init_kl_coef=cfg.init_kl_coef,
            target_kl=cfg.target_kl,
            horizon=cfg.kl_horizon,
            kl_lr=cfg.kl_lr,
        )

        # Training loop
        logger.info("Starting PPO training for %d steps on %d GPUs (fp32)...", cfg.total_steps, num_gpus)
        training_records = []
        initial_reward = None

        for step, batch in enumerate(ppo_trainer.dataloader):
            if step >= cfg.total_steps:
                break

            # 1. Get query tensors
            query_tensors = batch["input_ids"]

            # 2. Generate responses
            response_tensors = ppo_trainer.generate(
                query_tensors, return_prompt=False, **generation_kwargs
            )

            # 3. Decode responses
            batch_responses = tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )
            batch_queries = batch["query"]

            # 4. Score with frozen RM
            rewards = score_responses(
                reward_model=reward_model,
                reward_tokenizer=reward_tokenizer,
                queries=batch_queries,
                responses=batch_responses,
                max_length=cfg.max_seq_length,
            )

            # 5. PPO update step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            # 6. Extract metrics
            mean_reward = torch.stack(rewards).mean().item()

            approx_kl = stats.get("ppo/mean_kl", stats.get("objective/kl", 0.0))
            if isinstance(approx_kl, torch.Tensor):
                approx_kl = approx_kl.item()

            entropy = stats.get("ppo/mean_entropy", stats.get("objective/entropy", 0.0))
            if isinstance(entropy, torch.Tensor):
                entropy = entropy.item()

            value_loss = stats.get("ppo/value_loss", stats.get("ppo/loss/value", 0.0))
            if isinstance(value_loss, torch.Tensor):
                value_loss = value_loss.item()

            pg_loss = stats.get("ppo/policy_gradient_loss", stats.get("ppo/loss/policy", 0.0))
            if isinstance(pg_loss, torch.Tensor):
                pg_loss = pg_loss.item()

            clip_fraction = stats.get("ppo/clip_fraction", stats.get("ppo/policy/clipfrac", 0.0))
            if isinstance(clip_fraction, torch.Tensor):
                clip_fraction = clip_fraction.item()

            if initial_reward is None:
                initial_reward = mean_reward

            # 7. Update adaptive KL controller
            kl_controller.update(approx_kl, step=step)

            # 8. Log metrics
            if step % cfg.log_every_n_steps == 0:
                mlflow.log_metric("mean_reward", mean_reward, step=step)
                mlflow.log_metric("kl_divergence", approx_kl, step=step)
                mlflow.log_metric("policy_entropy", entropy, step=step)
                mlflow.log_metric("value_loss", value_loss, step=step)
                mlflow.log_metric("pg_loss", pg_loss, step=step)
                mlflow.log_metric("ppo_clip_fraction", clip_fraction, step=step)
                kl_controller.log_metrics(step)

                # Update Prometheus gauges (exposed for laptop scraping)
                PROM_MEAN_REWARD.set(mean_reward)
                PROM_KL_DIV.set(approx_kl)
                PROM_POLICY_ENTROPY.set(entropy)
                PROM_VALUE_LOSS.set(value_loss)
                PROM_PG_LOSS.set(pg_loss)
                PROM_CLIP_FRACTION.set(clip_fraction)
                PROM_KL_COEF.set(kl_controller.kl_coef)

                logger.info(
                    "Step %d/%d: reward=%.4f, kl=%.6f, entropy=%.4f, "
                    "vf_loss=%.4f, pg_loss=%.4f, clip=%.4f, beta=%.4f",
                    step, cfg.total_steps, mean_reward, approx_kl, entropy,
                    value_loss, pg_loss, clip_fraction, kl_controller.kl_coef,
                )

            # Record for CSV
            training_records.append({
                "step": step,
                "mean_reward": mean_reward,
                "kl_divergence": approx_kl,
                "policy_entropy": entropy,
                "value_loss": value_loss,
                "pg_loss": pg_loss,
                "clip_fraction": clip_fraction,
                "kl_coef": kl_controller.kl_coef,
            })

            # 9. Save checkpoint periodically
            if step > 0 and step % cfg.save_every_n_steps == 0:
                ckpt_dir = Path(cfg.output_dir) / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ppo_trainer.save_pretrained(str(ckpt_dir))
                logger.info("Checkpoint saved to %s", ckpt_dir)

            # 10. Early stopping on KL explosion
            if kl_controller.is_kl_exploding():
                logger.warning(
                    "KL EXPLOSION detected at step %d (kl=%.4f vs target=%.4f). "
                    "Consider stopping training.",
                    step, approx_kl, cfg.target_kl,
                )

        # Save final model
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving final PPO policy to %s", output_dir)
        ppo_trainer.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        mlflow.log_artifacts(str(output_dir), artifact_path="ppo_policy")

        # Save training curves CSV
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = reports_dir / "ppo_training_curves.csv"
        if training_records:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=training_records[0].keys())
                writer.writeheader()
                writer.writerows(training_records)

        # Save metrics JSON
        final_reward = training_records[-1]["mean_reward"] if training_records else 0.0
        final_kl = training_records[-1]["kl_divergence"] if training_records else 0.0
        metrics = {
            "initial_reward": initial_reward or 0.0,
            "final_reward": final_reward,
            "reward_improvement": final_reward - (initial_reward or 0.0),
            "final_kl": final_kl,
            "final_kl_coef": kl_controller.kl_coef,
            "total_steps": len(training_records),
            "kl_controller_stats": kl_controller.get_stats(),
        }
        metrics_path = reports_dir / "ppo_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info("PPO training complete. Final reward: %.4f, Final KL: %.6f", final_reward, final_kl)
        logger.info("Policy saved to %s", output_dir)


def main() -> None:
    """CLI entry point for PPO training."""
    parser = argparse.ArgumentParser(description="Stage 3: PPO Policy Optimization (fp32, 4xK80)")
    parser.add_argument(
        "--config",
        type=str,
        default="params.yaml",
        help="Path to params.yaml config file",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    cfg = load_config(args.config)
    train_ppo(cfg)


if __name__ == "__main__":
    main()
