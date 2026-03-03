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
from prometheus_client import Gauge, start_http_server
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

    _defaults = SFTConfig()

    return SFTConfig(
        base_model=g.get("base_model", _defaults.base_model),
        seed=g.get("seed", _defaults.seed),
        max_seq_length=g.get("max_seq_length", _defaults.max_seq_length),
        output_dir=s.get("output_dir", _defaults.output_dir),
        experiment_name=s.get("experiment_name", _defaults.experiment_name),
        mlflow_tracking_uri=g.get("mlflow_tracking_uri", _defaults.mlflow_tracking_uri),
        lora_r=lora.get("r", _defaults.lora_r),
        lora_alpha=lora.get("lora_alpha", _defaults.lora_alpha),
        target_modules=lora.get("target_modules", _defaults.target_modules),
        lora_dropout=lora.get("lora_dropout", _defaults.lora_dropout),
        lora_bias=lora.get("bias", _defaults.lora_bias),
        per_device_train_batch_size=t.get("per_device_train_batch_size", _defaults.per_device_train_batch_size),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", _defaults.gradient_accumulation_steps),
        warmup_ratio=t.get("warmup_ratio", _defaults.warmup_ratio),
        num_train_epochs=t.get("num_train_epochs", _defaults.num_train_epochs),
        learning_rate=t.get("learning_rate", _defaults.learning_rate),
        fp16=t.get("fp16", _defaults.fp16),
        packing=t.get("packing", _defaults.packing),
        logging_steps=t.get("logging_steps", _defaults.logging_steps),
        save_strategy=t.get("save_strategy", _defaults.save_strategy),
        evaluation_strategy=t.get("evaluation_strategy", _defaults.evaluation_strategy),
        optim=t.get("optim", _defaults.optim),
        weight_decay=t.get("weight_decay", _defaults.weight_decay),
        max_grad_norm=t.get("max_grad_norm", _defaults.max_grad_norm),
        lr_scheduler_type=t.get("lr_scheduler_type", _defaults.lr_scheduler_type),
        gradient_checkpointing=t.get("gradient_checkpointing", _defaults.gradient_checkpointing),
        val_split_ratio=e.get("val_split_ratio", _defaults.val_split_ratio),
        max_perplexity=e.get("max_perplexity", _defaults.max_perplexity),
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
        dataloader_num_workers=2,
        dataloader_prefetch_factor=2,
    )


# --------------- Prometheus gauges (shared across callbacks) ---------------
_PROM_LOSS = Gauge("rlhf_sft_train_loss", "SFT training loss")
_PROM_LR = Gauge("rlhf_sft_learning_rate", "SFT learning rate")
_PROM_EPOCH = Gauge("rlhf_sft_epoch", "SFT current epoch")
_PROM_STEP = Gauge("rlhf_sft_global_step", "SFT global training step")
_PROM_GRAD_NORM = Gauge("rlhf_sft_grad_norm", "SFT gradient norm")
_PROM_EVAL_LOSS = Gauge("rlhf_sft_eval_loss", "SFT evaluation loss")
_PROM_EVAL_PPL = Gauge("rlhf_sft_eval_perplexity", "SFT evaluation perplexity")
_PROM_GPU_MEM = Gauge("rlhf_sft_gpu_memory_gb", "GPU memory used (GB)", ["gpu"])

_prom_server_started = False


def _start_prometheus_server(port: int = 9091) -> None:
    """Start Prometheus HTTP metrics server (only on rank 0)."""
    global _prom_server_started
    if _prom_server_started:
        return
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if rank != 0:
        return
    try:
        start_http_server(port, addr="0.0.0.0")
        _prom_server_started = True
        logger.info("Prometheus metrics server started on port %d", port)
    except OSError as e:
        logger.warning("Could not start Prometheus server on port %d: %s", port, e)


class PrometheusCallback(TrainerCallback):
    """Exports live training metrics to Prometheus on port 9091."""

    def on_train_begin(self, args, state, control, **kwargs):
        _start_prometheus_server(9091)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            _PROM_LOSS.set(logs["loss"])
        if "learning_rate" in logs:
            _PROM_LR.set(logs["learning_rate"])
        if "epoch" in logs:
            _PROM_EPOCH.set(logs["epoch"])
        if "grad_norm" in logs:
            _PROM_GRAD_NORM.set(logs["grad_norm"])
        if "eval_loss" in logs:
            _PROM_EVAL_LOSS.set(logs["eval_loss"])
            ppl = torch.exp(torch.tensor(logs["eval_loss"])).item()
            _PROM_EVAL_PPL.set(ppl)
        _PROM_STEP.set(state.global_step)
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_gb = torch.cuda.memory_allocated(i) / 1e9
                _PROM_GPU_MEM.labels(gpu=str(i)).set(round(mem_gb, 2))


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

        # Callbacks
        mlflow_callback = MLflowLoggingCallback()
        prom_callback = PrometheusCallback()

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
            callbacks=[mlflow_callback, prom_callback],
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
