"""PPO fine-tuning (Stage 3 of the RLHF pipeline).

Example
-------
    python scripts/train_ppo.py \\
        --policy_path outputs/sft \\
        --reward_path outputs/reward_model \\
        --prompts_path data/prompts.json \\
        --output_dir outputs/ppo \\
        --num_steps 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rlhf.config import PPOConfig
from rlhf.models.policy_model import PolicyModel
from rlhf.models.reward_model import RewardModel
from rlhf.training.ppo_trainer import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO Fine-Tuning")
    parser.add_argument(
        "--policy_path",
        required=True,
        help="Path to SFT-initialised policy model",
    )
    parser.add_argument(
        "--reward_path",
        required=True,
        help="Path to trained reward model",
    )
    parser.add_argument(
        "--prompts_path",
        required=True,
        help="Path to JSON file with a list of prompt strings",
    )
    parser.add_argument("--output_dir", default="outputs/ppo")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.prompts_path) as f:
        prompts = json.load(f)

    config = PPOConfig(
        learning_rate=args.lr,
        num_train_steps=args.num_steps,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    policy = PolicyModel(args.policy_path)
    reward_model = RewardModel.from_pretrained(args.reward_path)
    trainer = PPOTrainer(policy, reward_model, config, kl_coef=args.kl_coef)
    metrics = trainer.train(prompts)
    final = metrics[-1] if metrics else {}
    print(f"PPO complete. Final metrics: {final}")


if __name__ == "__main__":
    main()
