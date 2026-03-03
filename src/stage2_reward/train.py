"""Stage 2 Reward Model — Training script with RewardTrainer and Bradley-Terry loss.

Trains a reward model on HH-RLHF chosen/rejected pairs using TRL's RewardTrainer.
The Bradley-Terry loss is: L = -log(sigmoid(r_chosen - r_rejected))

Usage:
    python -m src.stage2_reward.train --config params.yaml
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
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import RewardConfig, RewardTrainer

from src.stage2_reward.dataset import load_reward_dataset
from src.stage2_reward.model import load_reward_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RMConfig:
    """Configuration for Reward Model training, parsed from params.yaml.

    Attributes:
        base_model: HuggingFace model identifier.
        seed: Random seed for reproducibility.
        max_seq_length: Maximum sequence length.
        dataset_name: HuggingFace dataset identifier (HH-RLHF).
        sft_adapter_dir: Path to the SFT LoRA adapter.
        output_dir: Directory to save the reward model.
        experiment_name: MLflow experiment name.
        mlflow_tracking_uri: MLflow tracking server URI.
        freeze_layers_except_last_n: Number of transformer blocks to keep trainable.
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        num_train_epochs: Number of training epochs.
        learning_rate: Peak learning rate.
        warmup_steps: Number of warmup steps.
        fp16: Whether to use fp16 precision.
        logging_steps: Log metrics every N steps.
        save_strategy: Checkpoint saving strategy.
        evaluation_strategy: Evaluation strategy.
        eval_steps: Evaluate every N steps.
        optim: Optimizer name.
        weight_decay: Weight decay coefficient.
        max_grad_norm: Max gradient norm for clipping.
        lr_scheduler_type: LR scheduler type.
        max_length: Max sequence length for reward training.
        min_accuracy: Minimum RM accuracy for gate check.
        min_reward_margin_ratio: Min ratio of pairs where chosen > rejected.
    """

    base_model: str = "meta-llama/Llama-3.2-1B"
    seed: int = 42
    max_seq_length: int = 512
    dataset_name: str = "Anthropic/hh-rlhf"
    sft_adapter_dir: str = "models/sft_adapter"
    output_dir: str = "models/reward_model"
    experiment_name: str = "rlhf-reward-model"
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Model
    freeze_layers_except_last_n: int = 2

    # Training
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    fp16: bool = True
    logging_steps: int = 10
    save_strategy: str = "epoch"
    evaluation_strategy: str = "steps"
    eval_steps: int = 200
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    max_length: int = 512

    # Eval gate
    min_accuracy: float = 0.60
    min_reward_margin_ratio: float = 0.60


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

    return RMConfig(
        base_model=g.get("base_model", RMConfig.base_model),
        seed=g.get("seed", RMConfig.seed),
        max_seq_length=g.get("max_seq_length", RMConfig.max_seq_length),
        dataset_name=rm.get("dataset", RMConfig.dataset_name),
        sft_adapter_dir=sft.get("output_dir", RMConfig.sft_adapter_dir),
        output_dir=rm.get("output_dir", RMConfig.output_dir),
        experiment_name=rm.get("experiment_name", RMConfig.experiment_name),
        mlflow_tracking_uri=g.get("mlflow_tracking_uri", RMConfig.mlflow_tracking_uri),
        freeze_layers_except_last_n=m.get(
            "freeze_layers_except_last_n", RMConfig.freeze_layers_except_last_n
        ),
        per_device_train_batch_size=t.get(
            "per_device_train_batch_size", RMConfig.per_device_train_batch_size
        ),
        gradient_accumulation_steps=t.get(
            "gradient_accumulation_steps", RMConfig.gradient_accumulation_steps
        ),
        num_train_epochs=t.get("num_train_epochs", RMConfig.num_train_epochs),
        learning_rate=t.get("learning_rate", RMConfig.learning_rate),
        warmup_steps=t.get("warmup_steps", RMConfig.warmup_steps),
        fp16=t.get("fp16", RMConfig.fp16),
        logging_steps=t.get("logging_steps", RMConfig.logging_steps),
        save_strategy=t.get("save_strategy", RMConfig.save_strategy),
        evaluation_strategy=t.get("evaluation_strategy", RMConfig.evaluation_strategy),
        eval_steps=t.get("eval_steps", RMConfig.eval_steps),
        optim=t.get("optim", RMConfig.optim),
        weight_decay=t.get("weight_decay", RMConfig.weight_decay),
        max_grad_norm=t.get("max_grad_norm", RMConfig.max_grad_norm),
        lr_scheduler_type=t.get("lr_scheduler_type", RMConfig.lr_scheduler_type),
        max_length=t.get("max_length", RMConfig.max_length),
        min_accuracy=e.get("min_accuracy", RMConfig.min_accuracy),
        min_reward_margin_ratio=e.get(
            "min_reward_margin_ratio", RMConfig.min_reward_margin_ratio
        ),
    )


def compute_rm_metrics(eval_pred: tuple) -> dict[str, float]:
    """Compute reward model accuracy from predictions.

    The RewardTrainer outputs logits as (rewards_chosen, rewards_rejected)
    concatenated. Accuracy is the fraction where chosen reward > rejected reward.

    Args:
        eval_pred: Tuple of (predictions, labels) from the Trainer.

    Returns:
        Dictionary with 'accuracy' and 'mean_reward_margin' metrics.
    """
    predictions = eval_pred.predictions
    # RewardTrainer concatenates [chosen_rewards, rejected_rewards]
    # Each has shape (batch_size, 1)
    if len(predictions.shape) == 1:
        half = len(predictions) // 2
        chosen_rewards = predictions[:half]
        rejected_rewards = predictions[half:]
    else:
        chosen_rewards = predictions[:, 0]
        rejected_rewards = predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]

    # Accuracy: fraction where chosen > rejected
    correct = (chosen_rewards > rejected_rewards).sum()
    accuracy = correct / len(chosen_rewards)

    # Reward margin
    margins = chosen_rewards - rejected_rewards
    mean_margin = float(np.mean(margins))

    return {
        "accuracy": float(accuracy),
        "mean_reward_margin": mean_margin,
    }


def train_reward_model(cfg: RMConfig) -> None:
    """Execute the full reward model training pipeline.

    Loads the SFT checkpoint, adds a reward head, trains with Bradley-Terry
    loss on HH-RLHF pairs, and logs everything to MLflow.

    Args:
        cfg: RM configuration dataclass.
    """
    set_seed(cfg.seed)
    logger.info("Random seed set to %d", cfg.seed)

    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name="reward-model-training") as run:
        logger.info("MLflow run ID: %s", run.info.run_id)

        # Log config
        mlflow.log_params({
            "base_model": cfg.base_model,
            "seed": cfg.seed,
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.num_train_epochs,
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "freeze_layers_except_last_n": cfg.freeze_layers_except_last_n,
            "max_length": cfg.max_length,
            "sft_adapter_dir": cfg.sft_adapter_dir,
        })

        # Load dataset
        logger.info("Loading HH-RLHF dataset...")
        dataset = load_reward_dataset(
            dataset_name=cfg.dataset_name,
            seed=cfg.seed,
        )
        mlflow.log_metric("train_pairs", len(dataset["train"]))
        mlflow.log_metric("test_pairs", len(dataset["test"]))

        # Load reward model
        logger.info("Loading reward model from SFT checkpoint...")
        model, tokenizer = load_reward_model(
            base_model_name=cfg.base_model,
            adapter_path=cfg.sft_adapter_dir,
            freeze_layers_except_last_n=cfg.freeze_layers_except_last_n,
        )

        # RewardConfig encapsulates training args for the reward trainer
        reward_config = RewardConfig(
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
            max_length=cfg.max_length,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )

        # Create RewardTrainer
        logger.info("Initializing RewardTrainer with Bradley-Terry loss...")
        trainer = RewardTrainer(
            model=model,
            args=reward_config,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_rm_metrics,
        )

        # Train
        logger.info("Starting reward model training...")
        train_result = trainer.train()

        # Log final training metrics
        mlflow.log_metric("rm_train_loss", train_result.training_loss)
        mlflow.log_metric(
            "rm_train_runtime_seconds", train_result.metrics["train_runtime"]
        )

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

        # Log model artifact
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
    parser = argparse.ArgumentParser(description="Stage 2: Reward Model Training")
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
