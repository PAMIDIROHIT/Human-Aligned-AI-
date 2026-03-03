"""Train the reward model (Stage 2 of the RLHF pipeline).

Example
-------
    python scripts/train_reward.py \\
        --model_name gpt2 \\
        --data_path data/preference_data.json \\
        --output_dir outputs/reward_model \\
        --num_epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rlhf.config import RewardModelConfig
from rlhf.data.dataset import load_preference_dataset
from rlhf.models.reward_model import RewardModel
from rlhf.training.reward_trainer import RewardTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward Model Training")
    parser.add_argument("--model_name", default="gpt2", help="HuggingFace model name")
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to JSON file with [{prompt, chosen, rejected}, ...] records",
    )
    parser.add_argument("--output_dir", default="outputs/reward_model")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.data_path) as f:
        preference_data = json.load(f)

    config = RewardModelConfig(
        model_name=args.model_name,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
    )

    reward_model = RewardModel(args.model_name)
    dataset = load_preference_dataset(
        preference_data, reward_model.tokenizer, config.max_seq_length
    )
    trainer = RewardTrainer(reward_model, config)
    losses = trainer.train(dataset)
    print(f"Training complete. Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
