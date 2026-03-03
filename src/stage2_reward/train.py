"""Stage 2 Reward Model -- Training script with RewardTrainer and Bradley-Terry loss.

Trains a reward model on HH-RLHF + UltraFeedback chosen/rejected pairs
using TRL's RewardTrainer. The loss is: L = -log(sigmoid(r_chosen - r_rejected))

Hardware target: 4x Tesla K80 (fp32, no quantization).
Compatible with: transformers==4.38.2, trl==0.7.11, peft==0.7.1

Usage:
    accelerate launch -m src.stage2_reward.train --config params.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import RewardTrainer

from src.stage2_reward.dataset import load_reward_dataset
from src.stage2_reward.model import load_reward_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RMConfig:
    """Configuration for Reward Model training, parsed from params.yaml."""

    base_model: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    seed: int = 42
    max_seq_length: int = 512
    sft_adapter_dir: str = "models/sft_adapter"
    output_dir: str = "models/reward_model"
    experiment_name: str = "rlhf-reward-model"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Model
    freeze_layers_except_last_n: int = 2

    # Training (fp32, K80-sized batches)
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    fp16: bool = False
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "steps"
    eval_steps: int = 200
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    max_length: int = 512
    gradient_checkpointing: bool = True

    # Eval gate
    min_accuracy: float = 0.60
    min_reward_margin_ratio: float = 0.60

    # Full config dict
    _raw_config: dict = field(default_factory=dict)


def load_config(config_path: str) -> RMConfig:
    """Load RM configuration from params.yaml.

    Args:
        config_path: Path to the params.yaml file.

    Returns:
        An RMConfig dataclass populated from the YAML file.
    """
    with open(config_path) as f:
        params = yaml.safe_load(f)

    g = params.get("global", {})
    rm = params.get("reward_model", {})
    m = rm.get("model", {})
    t = rm.get("training", {})
    e = rm.get("eval_gate", {})
    sft = params.get("sft", {})

    _d = RMConfig()

    return RMConfig(
        base_model=g.get("base_model", _d.base_model),
        seed=g.get("seed", _d.seed),
        max_seq_length=g.get("max_seq_length", _d.max_seq_length),
        sft_adapter_dir=sft.get("output_dir", _d.sft_adapter_dir),
        output_dir=rm.get("output_dir", _d.output_dir),
        experiment_name=rm.get("experiment_name", _d.experiment_name),
        mlflow_tracking_uri=g.get("mlflow_tracking_uri", _d.mlflow_tracking_uri),
        freeze_layers_except_last_n=m.get("freeze_layers_except_last_n", _d.freeze_layers_except_last_n),
        per_device_train_batch_size=t.get("per_device_train_batch_size", _d.per_device_train_batch_size),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", _d.gradient_accumulation_steps),
        num_train_epochs=t.get("num_train_epochs", _d.num_train_epochs),
        learning_rate=t.get("learning_rate", _d.learning_rate),
        warmup_steps=t.get("warmup_steps", _d.warmup_steps),
        fp16=t.get("fp16", _d.fp16),
        logging_steps=t.get("logging_steps", _d.logging_steps),
        save_strategy=t.get("save_strategy", _d.save_strategy),
        evaluation_strategy=t.get("evaluation_strategy", _d.evaluation_strategy),
        eval_steps=t.get("eval_steps", _d.eval_steps),
        optim=t.get("optim", _d.optim),
        weight_decay=t.get("weight_decay", _d.weight_decay),
        max_grad_norm=t.get("max_grad_norm", _d.max_grad_norm),
        lr_scheduler_type=t.get("lr_scheduler_type", _d.lr_scheduler_type),
        max_length=t.get("max_length", _d.max_length),
        gradient_checkpointing=t.get("gradient_checkpointing", _d.gradient_checkpointing),
        min_accuracy=e.get("min_accuracy", _d.min_accuracy),
        min_reward_margin_ratio=e.get("min_reward_margin_ratio", _d.min_reward_margin_ratio),
        _raw_config=params,
    )


def compute_rm_metrics(eval_pred: tuple) -> dict[str, float]:
    """Compute reward model accuracy from predictions.

    Args:
        eval_pred: Tuple of (predictions, labels) from the Trainer.

    Returns:
        Dictionary with 'accuracy' and 'mean_reward_margin' metrics.
    """
    predictions = eval_pred.predictions
    if len(predictions.shape) == 1:
        half = len(predictions) // 2
        chosen_rewards = predictions[:half]
        rejected_rewards = predictions[half:]
    else:
        chosen_rewards = predictions[:, 0]
        rejected_rewards = predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]

    correct = (chosen_rewards > rejected_rewards).sum()
    accuracy = correct / max(len(chosen_rewards), 1)

    margins = chosen_rewards - rejected_rewards
    mean_margin = float(np.mean(margins))

    return {
        "accuracy": float(accuracy),
        "mean_reward_margin": mean_margin,
    }


def train_reward_model(cfg: RMConfig) -> None:
    """Execute the full reward model training pipeline (fp32, K80).

    Args:
        cfg: RM configuration dataclass.
    """
    set_seed(cfg.seed)
    logger.info("Random seed set to %d", cfg.seed)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name="reward-model-training") as run:
        logger.info("MLflow run ID: %s", run.info.run_id)

        mlflow.log_params({
            "base_model": cfg.base_model,
            "seed": cfg.seed,
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.num_train_epochs,
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "freeze_layers_except_last_n": cfg.freeze_layers_except_last_n,
            "max_length": cfg.max_length,
            "precision": "fp32",
        })

        # Load dataset from local disk
        logger.info("Loading reward dataset from local disk...")
        dataset = load_reward_dataset(
            config=cfg._raw_config,
            seed=cfg.seed,
        )
        mlflow.log_metric("train_pairs", len(dataset["train"]))
        mlflow.log_metric("test_pairs", len(dataset["test"]))

        # Load reward model (fp32, no quantization)
        logger.info("Loading reward model from SFT checkpoint (fp32)...")
        model, tokenizer = load_reward_model(
            base_model_name=cfg.base_model,
            adapter_path=cfg.sft_adapter_dir,
            freeze_layers_except_last_n=cfg.freeze_layers_except_last_n,
        )

        # Training arguments (use TrainingArguments, NOT RewardConfig)
        # TRL 0.7.11 does not have RewardConfig
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.num_train_epochs,
            learning_rate=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            fp16=cfg.fp16,
            logging_steps=cfg.logging_steps,
            save_strategy=cfg.save_strategy,
            evaluation_strategy=cfg.evaluation_strategy,
            eval_steps=cfg.eval_steps,
            optim=cfg.optim,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            lr_scheduler_type=cfg.lr_scheduler_type,
            seed=cfg.seed,
            report_to="none",
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            gradient_checkpointing=cfg.gradient_checkpointing,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=2,
            dataloader_prefetch_factor=2,
        )

        # Create RewardTrainer (TRL 0.7.11 API)
        logger.info("Initializing RewardTrainer with Bradley-Terry loss...")
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_rm_metrics,
            max_length=cfg.max_length,
        )

        # Train
        logger.info("Starting reward model training (fp32, K80)...")
        train_result = trainer.train()

        mlflow.log_metric("rm_train_loss", train_result.training_loss)
        mlflow.log_metric("rm_train_runtime_seconds", train_result.metrics["train_runtime"])

        # Evaluate
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        accuracy = eval_results.get("eval_accuracy", 0.0)
        mean_margin = eval_results.get("eval_mean_reward_margin", 0.0)

        mlflow.log_metric("rm_eval_accuracy", accuracy)
        mlflow.log_metric("rm_eval_mean_reward_margin", mean_margin)
        logger.info("RM accuracy: %.4f, Mean reward margin: %.4f", accuracy, mean_margin)

        # Save model
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving reward model to %s", output_dir)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        mlflow.log_artifacts(str(output_dir), artifact_path="reward_model")

        # Save metrics JSON
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "train_loss": train_result.training_loss,
            "eval_accuracy": accuracy,
            "eval_mean_reward_margin": mean_margin,
            "train_pairs": len(dataset["train"]),
            "test_pairs": len(dataset["test"]),
        }
        metrics_path = reports_dir / "rm_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Reward model training complete. Model saved to %s", output_dir)


def main() -> None:
    """CLI entry point for reward model training."""
    parser = argparse.ArgumentParser(description="Stage 2: Reward Model Training (fp32, K80)")
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
    train_reward_model(cfg)


if __name__ == "__main__":
    main()
