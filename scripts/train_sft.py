"""Run Supervised Fine-Tuning (Stage 1 of the RLHF pipeline).

Example
-------
    python scripts/train_sft.py \\
        --model_name gpt2 \\
        --data_path data/sft_data.json \\
        --output_dir outputs/sft \\
        --num_epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rlhf.config import SFTConfig
from rlhf.data.dataset import load_sft_dataset
from rlhf.models.policy_model import PolicyModel
from rlhf.training.sft_trainer import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")
    parser.add_argument("--model_name", default="gpt2", help="HuggingFace model name")
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to JSON file with [{prompt, response}, ...] records",
    )
    parser.add_argument("--output_dir", default="outputs/sft")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.data_path) as f:
        sft_data = json.load(f)

    config = SFTConfig(
        model_name=args.model_name,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
    )

    policy = PolicyModel(args.model_name)
    dataset = load_sft_dataset(sft_data, policy.tokenizer, config.max_seq_length)
    trainer = SFTTrainer(policy, config)
    losses = trainer.train(dataset)
    print(f"Training complete. Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
