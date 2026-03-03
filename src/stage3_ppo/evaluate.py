"""Stage 3 PPO -- Evaluation script with KL and reward gate checks.

Hardware target: 4x Tesla K80 (fp32).
Compatible with: transformers==4.38.2

Usage:
    python -m src.stage3_ppo.evaluate --config params.yaml
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
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_responses(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> list[str]:
    """Generate responses from a causal LM.

    Args:
        model: The causal language model.
        tokenizer: The tokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Maximum new tokens.
        temperature: Sampling temperature.

    Returns:
        List of generated response strings.
    """
    model.eval()
    device = next(model.parameters()).device
    responses = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(response)

    return responses


def evaluate_ppo(config_path: str) -> None:
    """Run PPO evaluation: check KL and reward improvement gates.

    Args:
        config_path: Path to params.yaml.
    """
    with open(config_path) as f:
        params = yaml.safe_load(f)

    global_cfg = params.get("global", {})
    ppo_cfg = params.get("ppo", {})
    eval_gate = ppo_cfg.get("eval_gate", {})

    seed = global_cfg.get("seed", 42)
    ppo_dir = ppo_cfg.get("output_dir", "models/ppo_policy")
    max_kl = eval_gate.get("max_kl", 0.15)
    min_reward_improvement = eval_gate.get("min_reward_improvement", 0.1)
    experiment_name = ppo_cfg.get("experiment_name", "rlhf-ppo")
    mlflow_uri = global_cfg.get("mlflow_tracking_uri", "http://localhost:5000")

    set_seed(seed)

    # Load PPO training metrics
    ppo_metrics_path = Path("reports") / "ppo_metrics.json"
    if ppo_metrics_path.exists():
        with open(ppo_metrics_path) as f:
            ppo_metrics = json.load(f)
    else:
        ppo_metrics = {}

    final_kl = ppo_metrics.get("final_kl", 0.0)
    reward_improvement = ppo_metrics.get("reward_improvement", 0.0)

    logger.info("PPO final KL: %.6f (max: %.4f)", final_kl, max_kl)
    logger.info("Reward improvement: %.4f (min: %.4f)", reward_improvement, min_reward_improvement)

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="ppo-evaluation"):
        mlflow.log_metric("final_kl", final_kl)
        mlflow.log_metric("reward_improvement", reward_improvement)
        mlflow.log_metric("kl_gate_passed", int(final_kl <= max_kl))
        mlflow.log_metric("reward_gate_passed", int(reward_improvement >= min_reward_improvement))

    # Save metrics
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    eval_metrics = {
        "final_kl": final_kl,
        "max_kl_threshold": max_kl,
        "kl_gate_passed": final_kl <= max_kl,
        "reward_improvement": reward_improvement,
        "min_reward_improvement_threshold": min_reward_improvement,
        "reward_gate_passed": reward_improvement >= min_reward_improvement,
        "overall_gate_passed": (final_kl <= max_kl) and (reward_improvement >= min_reward_improvement),
    }
    metrics_path = reports_dir / "ppo_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info("PPO evaluation metrics saved to %s", metrics_path)

    gate_passed = True
    if final_kl > max_kl:
        logger.error("GATE CHECK FAILED: KL %.6f > threshold %.4f", final_kl, max_kl)
        gate_passed = False
    if reward_improvement < min_reward_improvement:
        logger.error("GATE CHECK FAILED: Reward improvement %.4f < threshold %.4f", reward_improvement, min_reward_improvement)
        gate_passed = False
    if not gate_passed:
        sys.exit(1)
    logger.info("GATE CHECK PASSED: All PPO evaluation metrics meet thresholds")


def main() -> None:
    """CLI entry point for PPO evaluation."""
    parser = argparse.ArgumentParser(description="Stage 3: PPO Evaluation Gate")
    parser.add_argument("--config", type=str, default="params.yaml")
    args = parser.parse_args()
    if not Path(args.config).exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)
    evaluate_ppo(args.config)


if __name__ == "__main__":
    main()
