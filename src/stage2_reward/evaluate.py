"""Stage 2 Reward Model -- Evaluation script with accuracy gate check.

Computes RM accuracy on test split, logs to MLflow, enforces gate.

Hardware target: 4x Tesla K80 (fp32).
Compatible with: transformers==4.38.2

Usage:
    python -m src.stage2_reward.evaluate --config params.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from src.stage2_reward.dataset import load_reward_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def compute_reward_scores(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int = 512,
    batch_size: int = 8,
) -> np.ndarray:
    """Compute scalar reward scores for a list of texts.

    Args:
        model: The reward model.
        tokenizer: The tokenizer.
        texts: List of text strings.
        max_length: Maximum sequence length.
        batch_size: Inference batch size.

    Returns:
        Array of reward scores.
    """
    model.eval()
    device = next(model.parameters()).device
    all_scores = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
            all_scores.extend(scores.tolist() if scores.ndim > 0 else [scores.item()])

    return np.array(all_scores)


def evaluate_reward_model(config_path: str) -> None:
    """Run RM evaluation: accuracy, reward margin, and gate check.

    Args:
        config_path: Path to params.yaml.
    """
    with open(config_path) as f:
        params = yaml.safe_load(f)

    global_cfg = params.get("global", {})
    rm_cfg = params.get("reward_model", {})
    eval_gate = rm_cfg.get("eval_gate", {})

    seed = global_cfg.get("seed", 42)
    max_length = rm_cfg.get("training", {}).get("max_length", 512)
    model_dir = rm_cfg.get("output_dir", "models/reward_model")
    min_accuracy = eval_gate.get("min_accuracy", 0.60)
    min_margin_ratio = eval_gate.get("min_reward_margin_ratio", 0.60)
    experiment_name = rm_cfg.get("experiment_name", "rlhf-reward-model")
    mlflow_uri = global_cfg.get("mlflow_tracking_uri", "http://localhost:5000")

    set_seed(seed)

    # Load test dataset from local disk
    logger.info("Loading test dataset...")
    dataset = load_reward_dataset(config=params, seed=seed)
    test_data = dataset["test"]

    eval_samples = min(1000, len(test_data))
    test_data = test_data.select(range(eval_samples))

    # Load reward model (fp32)
    logger.info("Loading reward model from %s (fp32)", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    chosen_texts = test_data["chosen"]
    rejected_texts = test_data["rejected"]

    logger.info("Computing reward scores for %d pairs...", eval_samples)
    chosen_scores = compute_reward_scores(model, tokenizer, chosen_texts, max_length=max_length)
    rejected_scores = compute_reward_scores(model, tokenizer, rejected_texts, max_length=max_length)

    margins = chosen_scores - rejected_scores
    accuracy = float(np.mean(chosen_scores > rejected_scores))
    mean_margin = float(np.mean(margins))
    positive_margin_ratio = float(np.mean(margins > 0))

    logger.info("RM accuracy: %.4f (threshold: %.4f)", accuracy, min_accuracy)
    logger.info("Mean reward margin: %.4f", mean_margin)
    logger.info("Positive margin ratio: %.4f (threshold: %.4f)", positive_margin_ratio, min_margin_ratio)

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="rm-evaluation"):
        mlflow.log_metric("rm_test_accuracy", accuracy)
        mlflow.log_metric("rm_test_mean_margin", mean_margin)
        mlflow.log_metric("rm_test_positive_margin_ratio", positive_margin_ratio)
        mlflow.log_metric("gate_passed", int(accuracy >= min_accuracy))

    # Save metrics
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    eval_metrics = {
        "test_accuracy": accuracy,
        "mean_reward_margin": mean_margin,
        "positive_margin_ratio": positive_margin_ratio,
        "min_accuracy_threshold": min_accuracy,
        "gate_passed": accuracy >= min_accuracy and positive_margin_ratio >= min_margin_ratio,
        "num_eval_pairs": eval_samples,
    }
    metrics_path = reports_dir / "rm_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info("Evaluation metrics saved to %s", metrics_path)

    if accuracy < min_accuracy:
        logger.error("GATE CHECK FAILED: Accuracy %.4f < threshold %.4f", accuracy, min_accuracy)
        sys.exit(1)
    if positive_margin_ratio < min_margin_ratio:
        logger.error("GATE CHECK FAILED: Margin ratio %.4f < threshold %.4f", positive_margin_ratio, min_margin_ratio)
        sys.exit(1)
    logger.info("GATE CHECK PASSED: All RM evaluation metrics meet thresholds")


def main() -> None:
    """CLI entry point for reward model evaluation."""
    parser = argparse.ArgumentParser(description="Stage 2: RM Evaluation Gate")
    parser.add_argument("--config", type=str, default="params.yaml")
    args = parser.parse_args()
    if not Path(args.config).exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)
    evaluate_reward_model(args.config)


if __name__ == "__main__":
    main()
