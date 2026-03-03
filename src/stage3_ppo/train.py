"""Stage 3 PPO — Training script with PPOTrainer.

Implements PPO policy optimization against the frozen reward model.
The actor is the SFT checkpoint, the reward is from Stage 2 RM,
and KL divergence is penalized against the frozen SFT reference.

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
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType
from prometheus_client import Gauge, start_http_server
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
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

# ---- Prometheus Metrics ----
PROM_MEAN_REWARD = Gauge("ppo_mean_reward", "PPO mean reward per step")
PROM_KL_DIV = Gauge("ppo_kl_divergence", "PPO KL divergence per step")
PROM_POLICY_ENTROPY = Gauge("ppo_policy_entropy", "PPO policy entropy per step")
PROM_VALUE_LOSS = Gauge("ppo_value_loss", "PPO value function loss per step")
PROM_PG_LOSS = Gauge("ppo_pg_loss", "PPO policy gradient loss per step")
PROM_CLIP_FRACTION = Gauge("ppo_clip_fraction", "PPO clip fraction per step")
PROM_KL_COEF = Gauge("ppo_kl_coef", "PPO KL penalty coefficient (beta)")


@dataclass
class PPOTrainConfig:
    """Configuration for PPO training, parsed from params.yaml.

    Attributes:
        base_model: HuggingFace model identifier.
        seed: Random seed for reproducibility.
        max_seq_length: Maximum sequence length.
        dataset_name: Dataset for PPO prompts.
        sft_adapter_dir: Path to SFT LoRA adapter.
        reward_model_dir: Path to trained reward model.
        output_dir: Directory to save the PPO policy.
        experiment_name: MLflow experiment name.
        mlflow_tracking_uri: MLflow tracking URI.
        learning_rate: PPO learning rate.
        batch_size: Batch size for PPO.
        mini_batch_size: Mini-batch size for PPO updates.
        gradient_accumulation_steps: Gradient accumulation steps.
        optimize_cuda_cache: Whether to optimize CUDA cache.
        early_stopping: Whether to enable early stopping.
        target_kl: Target KL divergence.
        kl_penalty: KL penalty type.
        use_score_scaling: Whether to scale scores.
        use_score_norm: Whether to normalize scores.
        ppo_epochs: Number of PPO epochs per batch.
        cliprange: PPO clip range for policy ratio.
        cliprange_value: Value function clip range.
        vf_coef: Value function loss coefficient.
        max_grad_norm: Max gradient norm.
        max_new_tokens: Max new tokens for generation.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        do_sample: Whether to sample during generation.
        init_kl_coef: Initial KL coefficient.
        kl_lr: KL controller learning rate.
        kl_horizon: KL controller horizon.
        gamma: GAE discount factor.
        lam: GAE lambda.
        total_steps: Total PPO training steps.
        log_every_n_steps: Logging frequency.
        save_every_n_steps: Checkpoint save frequency.
        eval_every_n_steps: Evaluation frequency.
        max_kl: Maximum allowed KL for gate check.
        min_reward_improvement: Minimum reward improvement for gate check.
        prometheus_port: Port for Prometheus metrics.
    """

    base_model: str = "meta-llama/Llama-3.2-1B"
    seed: int = 42
    max_seq_length: int = 512
    dataset_name: str = "Anthropic/hh-rlhf"
    sft_adapter_dir: str = "models/sft_adapter"
    reward_model_dir: str = "models/reward_model"
    output_dir: str = "models/ppo_policy"
    experiment_name: str = "rlhf-ppo"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # PPO config
    learning_rate: float = 1.41e-5
    batch_size: int = 128
    mini_batch_size: int = 32
    gradient_accumulation_steps: int = 1
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
    max_new_tokens: int = 200
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
    prometheus_port: int = 9091


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

    return PPOTrainConfig(
        base_model=g.get("base_model", PPOTrainConfig.base_model),
        seed=g.get("seed", PPOTrainConfig.seed),
        max_seq_length=g.get("max_seq_length", PPOTrainConfig.max_seq_length),
        dataset_name=ppo.get("dataset", PPOTrainConfig.dataset_name),
        sft_adapter_dir=sft.get("output_dir", PPOTrainConfig.sft_adapter_dir),
        reward_model_dir=rm.get("output_dir", PPOTrainConfig.reward_model_dir),
        output_dir=ppo.get("output_dir", PPOTrainConfig.output_dir),
        experiment_name=ppo.get("experiment_name", PPOTrainConfig.experiment_name),
        mlflow_tracking_uri=g.get("mlflow_tracking_uri", PPOTrainConfig.mlflow_tracking_uri),
        learning_rate=c.get("learning_rate", PPOTrainConfig.learning_rate),
        batch_size=c.get("batch_size", PPOTrainConfig.batch_size),
        mini_batch_size=c.get("mini_batch_size", PPOTrainConfig.mini_batch_size),
        gradient_accumulation_steps=c.get("gradient_accumulation_steps", PPOTrainConfig.gradient_accumulation_steps),
        optimize_cuda_cache=c.get("optimize_cuda_cache", PPOTrainConfig.optimize_cuda_cache),
        early_stopping=c.get("early_stopping", PPOTrainConfig.early_stopping),
        target_kl=c.get("target_kl", PPOTrainConfig.target_kl),
        kl_penalty=c.get("kl_penalty", PPOTrainConfig.kl_penalty),
        use_score_scaling=c.get("use_score_scaling", PPOTrainConfig.use_score_scaling),
        use_score_norm=c.get("use_score_norm", PPOTrainConfig.use_score_norm),
        ppo_epochs=c.get("ppo_epochs", PPOTrainConfig.ppo_epochs),
        cliprange=c.get("cliprange", PPOTrainConfig.cliprange),
        cliprange_value=c.get("cliprange_value", PPOTrainConfig.cliprange_value),
        vf_coef=c.get("vf_coef", PPOTrainConfig.vf_coef),
        max_grad_norm=c.get("max_grad_norm", PPOTrainConfig.max_grad_norm),
        max_new_tokens=gen.get("max_new_tokens", PPOTrainConfig.max_new_tokens),
        temperature=gen.get("temperature", PPOTrainConfig.temperature),
        top_k=gen.get("top_k", PPOTrainConfig.top_k),
        top_p=gen.get("top_p", PPOTrainConfig.top_p),
        do_sample=gen.get("do_sample", PPOTrainConfig.do_sample),
        init_kl_coef=kl.get("init_kl_coef", PPOTrainConfig.init_kl_coef),
        kl_lr=kl.get("kl_lr", PPOTrainConfig.kl_lr),
        kl_horizon=kl.get("horizon", PPOTrainConfig.kl_horizon),
        gamma=gae.get("gamma", PPOTrainConfig.gamma),
        lam=gae.get("lam", PPOTrainConfig.lam),
        total_steps=t.get("total_steps", PPOTrainConfig.total_steps),
        log_every_n_steps=t.get("log_every_n_steps", PPOTrainConfig.log_every_n_steps),
        save_every_n_steps=t.get("save_every_n_steps", PPOTrainConfig.save_every_n_steps),
        eval_every_n_steps=t.get("eval_every_n_steps", PPOTrainConfig.eval_every_n_steps),
        max_kl=e.get("max_kl", PPOTrainConfig.max_kl),
        min_reward_improvement=e.get("min_reward_improvement", PPOTrainConfig.min_reward_improvement),
        prometheus_port=mon.get("metrics_push_gateway", "").split(":")[-1] or 9091,
    )


def load_reward_model_for_scoring(
    model_dir: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the frozen reward model for scoring PPO rollouts.

    Args:
        model_dir: Path to the saved reward model directory.

    Returns:
        A tuple of (reward_model, tokenizer), both frozen for inference.
    """
    logger.info("Loading frozen reward model from: %s", model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        num_labels=1,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info("Reward model loaded and frozen")
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
        max_length: Maximum sequence length for scoring.

    Returns:
        A list of scalar reward tensors, one per query-response pair.
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

        rewards.append(reward.cpu())

    return rewards


def train_ppo(cfg: PPOTrainConfig) -> None:
    """Execute the full PPO training pipeline.

    Loads the SFT model as policy, the RM for scoring, and runs PPO
    optimization with adaptive KL penalty control.

    Args:
        cfg: PPO training configuration.
    """
    set_seed(cfg.seed)
    logger.info("Random seed set to %d", cfg.seed)

    # Start Prometheus metrics server
    try:
        start_http_server(int(cfg.prometheus_port))
        logger.info("Prometheus metrics server started on port %s", cfg.prometheus_port)
    except Exception as e:
        logger.warning("Could not start Prometheus server: %s", e)

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name="ppo-training") as run:
        logger.info("MLflow run ID: %s", run.info.run_id)

        # Log all config
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
        })

        # Load PPO prompts
        logger.info("Loading PPO prompt dataset...")
        prompt_dataset = load_ppo_dataset(
            dataset_name=cfg.dataset_name,
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

        # QLoRA config for the policy model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # LoRA config for PPO fine-tuning
        lora_config = LoraConfig(
            r=16,  # Smaller rank for PPO (memory efficiency)
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # PPO Config
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
            log_with=None,  # Using MLflow manually
        )

        # Load policy model (SFT checkpoint with value head)
        logger.info("Loading policy model with value head...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.sft_adapter_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            peft_config=lora_config,
        )

        # Load reference model (frozen SFT model for KL computation)
        logger.info("Loading reference model (frozen SFT)...")
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.sft_adapter_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load frozen reward model
        logger.info("Loading frozen reward model...")
        reward_model, reward_tokenizer = load_reward_model_for_scoring(
            cfg.reward_model_dir
        )

        # Create PPOTrainer
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
        logger.info("Starting PPO training for %d steps...", cfg.total_steps)
        training_records = []
        initial_reward = None

        for step, batch in enumerate(ppo_trainer.dataloader):
            if step >= cfg.total_steps:
                break

            # 1. Get query tensors
            query_tensors = batch["input_ids"]

            # 2. Generate responses from the policy
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

            # 6. Extract and log metrics
            mean_reward = torch.stack(rewards).mean().item()
            kl_div = stats.get("ppo/mean_non_score_reward", 0.0)
            if isinstance(kl_div, torch.Tensor):
                kl_div = kl_div.item()

            # Approximate KL from stats
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

            # Track initial reward for gate check
            if initial_reward is None:
                initial_reward = mean_reward

            # 7. Update adaptive KL controller
            kl_controller.update(approx_kl, step=step)

            # 8. Log to MLflow
            if step % cfg.log_every_n_steps == 0:
                mlflow.log_metric("mean_reward", mean_reward, step=step)
                mlflow.log_metric("kl_divergence", approx_kl, step=step)
                mlflow.log_metric("policy_entropy", entropy, step=step)
                mlflow.log_metric("value_loss", value_loss, step=step)
                mlflow.log_metric("pg_loss", pg_loss, step=step)
                mlflow.log_metric("ppo_clip_fraction", clip_fraction, step=step)
                kl_controller.log_metrics(step)

                # Update Prometheus gauges
                PROM_MEAN_REWARD.set(mean_reward)
                PROM_KL_DIV.set(approx_kl)
                PROM_POLICY_ENTROPY.set(entropy)
                PROM_VALUE_LOSS.set(value_loss)
                PROM_PG_LOSS.set(pg_loss)
                PROM_CLIP_FRACTION.set(clip_fraction)
                PROM_KL_COEF.set(kl_controller.kl_coef)

                logger.info(
                    "Step %d/%d: reward=%.4f, kl=%.6f, entropy=%.4f, "
                    "vf_loss=%.4f, pg_loss=%.4f, clip=%.4f, β=%.4f",
                    step,
                    cfg.total_steps,
                    mean_reward,
                    approx_kl,
                    entropy,
                    value_loss,
                    pg_loss,
                    clip_fraction,
                    kl_controller.kl_coef,
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
                    step,
                    approx_kl,
                    cfg.target_kl,
                )

        # Save final model
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving final PPO policy to %s", output_dir)
        ppo_trainer.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Log final model as artifact
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
    parser = argparse.ArgumentParser(description="Stage 3: PPO Policy Optimization")
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
