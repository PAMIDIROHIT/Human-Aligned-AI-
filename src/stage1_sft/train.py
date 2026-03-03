"""Stage 1 SFT -- Training script with LoRA (fp32) and SFTTrainer.

Loads TinyLlama-1.1B in fp32 (no quantization), applies LoRA adapters,
and fine-tunes on UltraChat-200k + Guanaco using TRL's SFTTrainer.
All hyperparameters are read from params.yaml.

Hardware target: 4x Tesla K80 (11 GB each, CC 3.7, fp32 only)
Compatible with: transformers==4.38.2, trl==0.7.11, peft==0.7.1

Usage:
    accelerate launch -m src.stage1_sft.train --config params.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

from src.stage1_sft.dataset import load_sft_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training, parsed from params.yaml."""

    base_model: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    seed: int = 42
    max_seq_length: int = 512
    output_dir: str = "models/sft_adapter"
    experiment_name: str = "rlhf-sft"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # LoRA (no quantization)
    lora_r: int = 64
    lora_alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    lora_bias: str = "none"

    # Training (fp32, K80-sized batches)
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    fp16: bool = False
    packing: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    optim: str = "adamw_torch"
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True

    # Eval gate
    val_split_ratio: float = 0.1
    max_perplexity: float = 15.0

    # Full config dict (for dataset loader)
    _raw_config: dict = field(default_factory=dict)


def load_config(config_path: str) -> SFTConfig:
    """Load SFT configuration from params.yaml.

    Args:
        config_path: Path to the params.yaml file.

    Returns:
        An SFTConfig dataclass populated with values from the YAML file.
    """
    with open(config_path) as f:
        params = yaml.safe_load(f)

    g = params.get("global", {})
    s = params.get("sft", {})
    lora = s.get("lora", s.get("qlora", {}))
    t = s.get("training", {})
    e = s.get("eval_gate", {})

    return SFTConfig(
        base_model=g.get("base_model", SFTConfig.base_model),
        seed=g.get("seed", SFTConfig.seed),
        max_seq_length=g.get("max_seq_length", SFTConfig.max_seq_length),
        output_dir=s.get("output_dir", SFTConfig.output_dir),
        experiment_name=s.get("experiment_name", SFTConfig.experiment_name),
        mlflow_tracking_uri=g.get("mlflow_tracking_uri", SFTConfig.mlflow_tracking_uri),
        lora_r=lora.get("r", SFTConfig.lora_r),
        lora_alpha=lora.get("lora_alpha", SFTConfig.lora_alpha),
        target_modules=lora.get("target_modules", SFTConfig.target_modules),
        lora_dropout=lora.get("lora_dropout", SFTConfig.lora_dropout),
        lora_bias=lora.get("bias", SFTConfig.lora_bias),
        per_device_train_batch_size=t.get("per_device_train_batch_size", SFTConfig.per_device_train_batch_size),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", SFTConfig.gradient_accumulation_steps),
        warmup_ratio=t.get("warmup_ratio", SFTConfig.warmup_ratio),
        num_train_epochs=t.get("num_train_epochs", SFTConfig.num_train_epochs),
        learning_rate=t.get("learning_rate", SFTConfig.learning_rate),
        fp16=t.get("fp16", SFTConfig.fp16),
        packing=t.get("packing", SFTConfig.packing),
        logging_steps=t.get("logging_steps", SFTConfig.logging_steps),
        save_strategy=t.get("save_strategy", SFTConfig.save_strategy),
        evaluation_strategy=t.get("evaluation_strategy", SFTConfig.evaluation_strategy),
        optim=t.get("optim", SFTConfig.optim),
        weight_decay=t.get("weight_decay", SFTConfig.weight_decay),
        max_grad_norm=t.get("max_grad_norm", SFTConfig.max_grad_norm),
        lr_scheduler_type=t.get("lr_scheduler_type", SFTConfig.lr_scheduler_type),
        gradient_checkpointing=t.get("gradient_checkpointing", SFTConfig.gradient_checkpointing),
        val_split_ratio=e.get("val_split_ratio", SFTConfig.val_split_ratio),
        max_perplexity=e.get("max_perplexity", SFTConfig.max_perplexity),
        _raw_config=params,
    )


def create_lora_config(cfg: SFTConfig) -> LoraConfig:
    """Create PEFT LoRA configuration.

    Args:
        cfg: SFT configuration dataclass.

    Returns:
        A LoraConfig for the PEFT library.
    """
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )


def create_training_args(cfg: SFTConfig) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config.

    Args:
        cfg: SFT configuration dataclass.

    Returns:
        A TrainingArguments object.
    """
    return TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        evaluation_strategy=cfg.evaluation_strategy,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        ddp_find_unused_parameters=False,
    )


class MLflowLoggingCallback(TrainerCallback):
    """Custom callback to log training metrics to MLflow."""

    def __init__(self) -> None:
        self._loss_records: list[dict[str, float]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            mlflow.log_metric("train_loss", logs["loss"], step=step)
            self._loss_records.append({"step": step, "loss": logs["loss"]})
        if "eval_loss" in logs:
            mlflow.log_metric("eval_loss", logs["eval_loss"], step=step)
            perplexity = torch.exp(torch.tensor(logs["eval_loss"])).item()
            mlflow.log_metric("eval_perplexity", perplexity, step=step)

    @property
    def loss_records(self) -> list[dict[str, float]]:
        return self._loss_records


def train_sft(cfg: SFTConfig) -> None:
    """Execute the full SFT training pipeline.

    Loads TinyLlama in fp32, applies LoRA adapters, trains on
    UltraChat + Guanaco, and logs everything to MLflow.

    Args:
        cfg: SFT configuration dataclass.
    """
    set_seed(cfg.seed)
    logger.info("Random seed set to %d", cfg.seed)

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be slow on CPU.")

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name="sft-lora-training") as run:
        logger.info("MLflow run ID: %s", run.info.run_id)

        mlflow.log_params({
            "base_model": cfg.base_model,
            "seed": cfg.seed,
            "max_seq_length": cfg.max_seq_length,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "target_modules": str(cfg.target_modules),
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.num_train_epochs,
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "packing": cfg.packing,
            "precision": "fp32",
            "hardware": "4xK80",
        })

        # Load dataset from local disk
        logger.info("Loading SFT dataset from local disk...")
        dataset = load_sft_dataset(
            config=cfg._raw_config,
            val_split_ratio=cfg.val_split_ratio,
            seed=cfg.seed,
        )
        mlflow.log_metric("train_samples", len(dataset["train"]))
        mlflow.log_metric("val_samples", len(dataset["validation"]))

        # Load tokenizer
        logger.info("Loading tokenizer: %s", cfg.base_model)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load model in fp32 (no quantization for K80)
        logger.info("Loading model in fp32 (no quantization): %s", cfg.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False

        # Enable gradient checkpointing to save memory
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Apply LoRA
        logger.info("Applying LoRA adapters (r=%d, alpha=%d)", cfg.lora_r, cfg.lora_alpha)
        lora_config = create_lora_config(cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Log trainable parameters
        trainable, total = model.get_nb_trainable_parameters()
        mlflow.log_metric("trainable_params", trainable)
        mlflow.log_metric("total_params", total)
        mlflow.log_metric("trainable_pct", 100 * trainable / total)

        # Training arguments
        training_args = create_training_args(cfg)

        # MLflow callback
        mlflow_callback = MLflowLoggingCallback()

        # Create SFTTrainer (TRL 0.7.11 API)
        logger.info("Initializing SFTTrainer with packing=%s", cfg.packing)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            packing=cfg.packing,
            max_seq_length=cfg.max_seq_length,
            dataset_text_field="text",
            callbacks=[mlflow_callback],
        )

        # Train
        logger.info("Starting SFT training (fp32, LoRA, K80)...")
        train_result = trainer.train()

        # Log final metrics
        mlflow.log_metric("final_train_loss", train_result.training_loss)
        mlflow.log_metric("train_runtime_seconds", train_result.metrics["train_runtime"])
        mlflow.log_metric(
            "train_samples_per_second",
            train_result.metrics["train_samples_per_second"],
        )

        # Evaluate
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        eval_loss = eval_results["eval_loss"]
        eval_perplexity = torch.exp(torch.tensor(eval_loss)).item()
        mlflow.log_metric("final_eval_loss", eval_loss)
        mlflow.log_metric("final_eval_perplexity", eval_perplexity)
        logger.info("Final eval perplexity: %.4f", eval_perplexity)

        # Save LoRA adapter
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving LoRA adapter to %s", output_dir)
        trainer.model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        # Log adapter as MLflow artifact
        mlflow.log_artifacts(str(output_dir), artifact_path="sft_adapter")

        # Save loss curve CSV for DVC plots
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        loss_csv_path = reports_dir / "sft_loss_curve.csv"
        with open(loss_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "loss"])
            writer.writeheader()
            writer.writerows(mlflow_callback.loss_records)

        # Save metrics JSON for DVC
        metrics = {
            "train_loss": train_result.training_loss,
            "eval_loss": eval_loss,
            "eval_perplexity": eval_perplexity,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total,
            "train_runtime_seconds": train_result.metrics["train_runtime"],
        }
        metrics_path = reports_dir / "sft_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("SFT training complete. Adapter saved to %s", output_dir)
        logger.info("Metrics saved to %s", metrics_path)


def main() -> None:
    """CLI entry point for SFT training."""
    parser = argparse.ArgumentParser(description="Stage 1: SFT with LoRA (fp32, K80)")
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
    train_sft(cfg)


if __name__ == "__main__":
    main()
