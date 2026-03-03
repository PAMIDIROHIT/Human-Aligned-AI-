"""Stage 1 SFT — Evaluation script with perplexity gate check.

Computes validation perplexity on the held-out split, logs to MLflow,
and enforces the perplexity gate defined in params.yaml.

Usage:
    python -m src.stage1_sft.evaluate --config params.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import DatasetDict
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, set_seed

from src.stage1_sft.dataset import load_sft_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def compute_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int = 512,
    batch_size: int = 4,
) -> float:
    """Compute perplexity on a list of text sequences.

    Args:
        model: The language model (PEFT or base).
        tokenizer: The tokenizer for the model.
        texts: List of text strings to evaluate.
        max_length: Maximum sequence length for tokenization.
        batch_size: Evaluation batch size.

    Returns:
        The perplexity (exp of mean cross-entropy loss).
    """
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

        # Only count non-padding tokens
        num_tokens = attention_mask.sum().item()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def evaluate_sft(config_path: str) -> None:
    """Run SFT evaluation: compute perplexity and enforce gate check.

    Args:
        config_path: Path to params.yaml.

    Raises:
        SystemExit: With code 1 if perplexity exceeds the gate threshold.
    """
    with open(config_path) as f:
        params = yaml.safe_load(f)

    global_cfg = params.get("global", {})
    sft_cfg = params.get("sft", {})
    eval_gate = sft_cfg.get("eval_gate", {})

    seed = global_cfg.get("seed", 42)
    base_model = global_cfg.get("base_model", "meta-llama/Llama-3.2-1B")
    max_seq_length = global_cfg.get("max_seq_length", 512)
    adapter_dir = sft_cfg.get("output_dir", "models/sft_adapter")
    max_perplexity = eval_gate.get("max_perplexity", 15.0)
    val_split_ratio = eval_gate.get("val_split_ratio", 0.1)
    experiment_name = sft_cfg.get("experiment_name", "rlhf-sft")
    mlflow_uri = global_cfg.get("mlflow_tracking_uri", "http://localhost:5000")

    set_seed(seed)

    # Load validation dataset
    logger.info("Loading validation dataset...")
    dataset = load_sft_dataset(
        dataset_name=sft_cfg.get("dataset", "tatsu-lab/alpaca"),
        val_split_ratio=val_split_ratio,
        seed=seed,
    )
    val_texts = dataset["validation"]["text"]

    # Load the fine-tuned model (LoRA adapter)
    logger.info("Loading SFT adapter from %s", adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Compute perplexity
    logger.info("Computing validation perplexity on %d samples...", len(val_texts))
    perplexity = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=val_texts[:500],  # Cap for speed in CI
        max_length=max_seq_length,
    )
    logger.info("Validation perplexity: %.4f (threshold: %.4f)", perplexity, max_perplexity)

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="sft-evaluation"):
        mlflow.log_metric("val_perplexity", perplexity)
        mlflow.log_metric("perplexity_threshold", max_perplexity)
        mlflow.log_metric("gate_passed", int(perplexity <= max_perplexity))

    # Save evaluation metrics
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    eval_metrics = {
        "val_perplexity": perplexity,
        "max_perplexity_threshold": max_perplexity,
        "gate_passed": perplexity <= max_perplexity,
        "num_eval_samples": min(len(val_texts), 500),
    }
    metrics_path = reports_dir / "sft_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info("Evaluation metrics saved to %s", metrics_path)

    # Gate check
    if perplexity > max_perplexity:
        logger.error(
            "GATE CHECK FAILED: Perplexity %.4f > threshold %.4f",
            perplexity,
            max_perplexity,
        )
        sys.exit(1)
    else:
        logger.info(
            "GATE CHECK PASSED: Perplexity %.4f <= threshold %.4f",
            perplexity,
            max_perplexity,
        )


def main() -> None:
    """CLI entry point for SFT evaluation."""
    parser = argparse.ArgumentParser(description="Stage 1: SFT Evaluation Gate")
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

    evaluate_sft(args.config)


if __name__ == "__main__":
    main()
