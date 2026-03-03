"""KL vs Reward Tradeoff Analysis — Pareto frontier analysis.

Loads PPO training logs from MLflow/CSV and produces:
- KL divergence (x-axis) vs Mean Reward (y-axis) scatter plot
- Pareto frontier identification
- Reward hacking region annotation
- JSON data export for programmatic consumption

Usage:
    python -m src.evaluation.kl_reward_tradeoff --config params.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def compute_pareto_frontier(
    kl_values: np.ndarray,
    reward_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Pareto frontier: max reward at each KL budget.

    The Pareto frontier identifies training checkpoints that achieve
    the best reward for a given level of KL divergence. Points below
    the frontier are dominated.

    Args:
        kl_values: Array of KL divergence values.
        reward_values: Array of corresponding mean reward values.

    Returns:
        Tuple of (pareto_kl, pareto_reward) arrays defining the frontier.
    """
    # Sort by KL (ascending)
    sorted_indices = np.argsort(kl_values)
    sorted_kl = kl_values[sorted_indices]
    sorted_rewards = reward_values[sorted_indices]

    # Build Pareto frontier (non-dominated points)
    pareto_kl = [sorted_kl[0]]
    pareto_reward = [sorted_rewards[0]]
    max_reward_seen = sorted_rewards[0]

    for i in range(1, len(sorted_kl)):
        if sorted_rewards[i] > max_reward_seen:
            pareto_kl.append(sorted_kl[i])
            pareto_reward.append(sorted_rewards[i])
            max_reward_seen = sorted_rewards[i]

    return np.array(pareto_kl), np.array(pareto_reward)


def detect_reward_hacking(
    kl_values: np.ndarray,
    reward_values: np.ndarray,
    kl_threshold: float = 0.5,
    window_size: int = 50,
) -> dict[str, Any]:
    """Detect reward hacking: reward increases but KL explodes.

    Reward hacking is identified when:
    - Reward continues increasing
    - KL divergence grows exponentially
    - Policy entropy collapses

    Args:
        kl_values: Array of KL divergence values over training.
        reward_values: Array of mean reward values over training.
        kl_threshold: KL value above which we consider it "exploding".
        window_size: Rolling window for trend detection.

    Returns:
        Dictionary with hacking detection results:
        - 'detected': bool
        - 'hacking_start_step': int or None
        - 'kl_at_detection': float or None
        - 'reward_at_detection': float or None
    """
    if len(kl_values) < window_size * 2:
        return {"detected": False, "hacking_start_step": None}

    # Look for the point where KL starts exploding while reward increases
    for i in range(window_size, len(kl_values) - window_size):
        recent_kl_trend = np.mean(kl_values[i:i + window_size]) - np.mean(
            kl_values[i - window_size:i]
        )
        recent_reward_trend = np.mean(reward_values[i:i + window_size]) - np.mean(
            reward_values[i - window_size:i]
        )

        # KL increasing rapidly while reward still going up
        if kl_values[i] > kl_threshold and recent_kl_trend > 0.01 and recent_reward_trend > 0:
            return {
                "detected": True,
                "hacking_start_step": i,
                "kl_at_detection": float(kl_values[i]),
                "reward_at_detection": float(reward_values[i]),
            }

    return {"detected": False, "hacking_start_step": None}


def plot_kl_reward_tradeoff(
    training_data: pd.DataFrame,
    output_path: str = "reports/kl_reward_tradeoff.png",
    target_kl: float = 0.1,
    sample_every: int = 100,
) -> None:
    """Generate the KL vs Reward tradeoff plot with Pareto frontier.

    Args:
        training_data: DataFrame with 'step', 'kl_divergence', 'mean_reward' columns.
        output_path: Path to save the output plot.
        target_kl: The target KL value (drawn as vertical line).
        sample_every: Sample data every N steps for scatter plot.
    """
    kl = training_data["kl_divergence"].values
    reward = training_data["mean_reward"].values
    steps = training_data["step"].values

    # Sample for scatter plot
    sample_mask = np.arange(len(kl)) % sample_every == 0
    kl_sampled = kl[sample_mask]
    reward_sampled = reward[sample_mask]
    steps_sampled = steps[sample_mask]

    # Compute Pareto frontier
    pareto_kl, pareto_reward = compute_pareto_frontier(kl_sampled, reward_sampled)

    # Detect reward hacking
    hacking = detect_reward_hacking(kl, reward)

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Scatter plot: color by training step (earlier = blue, later = red)
    scatter = ax.scatter(
        kl_sampled,
        reward_sampled,
        c=steps_sampled,
        cmap="coolwarm",
        alpha=0.6,
        s=30,
        edgecolors="none",
        label="Training checkpoints",
    )
    plt.colorbar(scatter, ax=ax, label="Training Step")

    # Pareto frontier
    ax.plot(
        pareto_kl,
        pareto_reward,
        "g-",
        linewidth=2.5,
        label="Pareto Frontier",
        zorder=5,
    )
    ax.scatter(
        pareto_kl,
        pareto_reward,
        c="green",
        s=60,
        zorder=6,
        edgecolors="darkgreen",
    )

    # Target KL line
    ax.axvline(
        x=target_kl,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Target KL = {target_kl}",
    )

    # Mark reward hacking region
    if hacking["detected"]:
        hacking_step = hacking["hacking_start_step"]
        ax.axvspan(
            kl[hacking_step],
            kl.max(),
            alpha=0.15,
            color="red",
            label="Reward Hacking Region",
        )
        ax.annotate(
            "Reward Hacking\n(KL Exploding)",
            xy=(hacking["kl_at_detection"], hacking["reward_at_detection"]),
            xytext=(hacking["kl_at_detection"] + 0.05, hacking["reward_at_detection"] - 0.3),
            fontsize=10,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )

    # Labels and formatting
    ax.set_xlabel("KL Divergence (nats)", fontsize=14)
    ax.set_ylabel("Mean Reward", fontsize=14)
    ax.set_title("KL vs Reward Tradeoff — PPO Training Dynamics", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add annotation for optimal point
    if len(pareto_kl) > 0:
        # Optimal = highest reward on Pareto frontier within target KL
        within_target = pareto_kl <= target_kl
        if within_target.any():
            best_idx = np.argmax(pareto_reward[within_target])
            ax.annotate(
                "Optimal\n(max reward\nwithin target KL)",
                xy=(pareto_kl[within_target][best_idx], pareto_reward[within_target][best_idx]),
                xytext=(pareto_kl[within_target][best_idx] + 0.02, pareto_reward[within_target][best_idx] + 0.2),
                fontsize=9,
                color="green",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
            )

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("KL-Reward tradeoff plot saved to %s", output_path)


def run_analysis(config_path: str) -> None:
    """Run the full KL vs Reward tradeoff analysis.

    Args:
        config_path: Path to params.yaml.
    """
    with open(config_path) as f:
        params = yaml.safe_load(f)

    eval_cfg = params.get("evaluation", {})
    ppo_cfg = params.get("ppo", {})
    plot_path = eval_cfg.get("kl_reward_plot_path", "reports/kl_reward_tradeoff.png")
    target_kl = ppo_cfg.get("config", {}).get("target_kl", 0.1)

    # Load PPO training curves CSV
    csv_path = Path("reports") / "ppo_training_curves.csv"
    if not csv_path.exists():
        logger.error("PPO training curves not found at %s", csv_path)
        logger.info("Run PPO training first: python -m src.stage3_ppo.train --config params.yaml")
        sys.exit(1)

    logger.info("Loading PPO training data from %s", csv_path)
    training_data = pd.read_csv(csv_path)
    logger.info("Loaded %d training steps", len(training_data))

    # Generate plot
    plot_kl_reward_tradeoff(
        training_data=training_data,
        output_path=plot_path,
        target_kl=target_kl,
    )

    # Compute and save analysis data
    kl = training_data["kl_divergence"].values
    reward = training_data["mean_reward"].values
    pareto_kl, pareto_reward = compute_pareto_frontier(kl, reward)
    hacking = detect_reward_hacking(kl, reward)

    analysis_data = {
        "total_steps": len(training_data),
        "target_kl": target_kl,
        "pareto_frontier": {
            "kl_values": pareto_kl.tolist(),
            "reward_values": pareto_reward.tolist(),
            "num_points": len(pareto_kl),
        },
        "reward_hacking": hacking,
        "summary_statistics": {
            "mean_kl": float(np.mean(kl)),
            "std_kl": float(np.std(kl)),
            "max_kl": float(np.max(kl)),
            "mean_reward": float(np.mean(reward)),
            "std_reward": float(np.std(reward)),
            "max_reward": float(np.max(reward)),
            "final_kl": float(kl[-1]) if len(kl) > 0 else None,
            "final_reward": float(reward[-1]) if len(reward) > 0 else None,
        },
    }

    json_path = Path(plot_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(analysis_data, f, indent=2)
    logger.info("Analysis data saved to %s", json_path)


def main() -> None:
    """CLI entry point for KL-Reward tradeoff analysis."""
    parser = argparse.ArgumentParser(description="KL vs Reward Tradeoff Analysis")
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

    run_analysis(args.config)


if __name__ == "__main__":
    main()
