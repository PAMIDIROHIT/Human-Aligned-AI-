"""Supervised Fine-Tuning (SFT) trainer."""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import SFTConfig
from ..data.dataset import PromptDataset
from ..models.policy_model import PolicyModel


class SFTTrainer:
    """Train a :class:`~rlhf.models.PolicyModel` via supervised fine-tuning.

    The trainer performs standard causal-LM next-token prediction on
    (prompt, response) pairs.

    Args:
        model: The policy model to fine-tune.
        config: Hyper-parameter configuration.
    """

    def __init__(self, model: PolicyModel, config: SFTConfig) -> None:
        self.model = model
        self.config = config

    def train(self, dataset: PromptDataset) -> list[float]:
        """Run the SFT training loop.

        Args:
            dataset: A :class:`~rlhf.data.PromptDataset` of
                     (prompt, response) pairs.

        Returns:
            List of per-step loss values.
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
        )

        optimizer = AdamW(
            self.model.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps = len(dataloader) * self.config.num_train_epochs
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=total_steps,
        )

        self.model.model.train()
        losses: list[float] = []

        for epoch in range(self.config.num_train_epochs):
            epoch_bar = tqdm(
                dataloader,
                desc=f"SFT epoch {epoch + 1}/{self.config.num_train_epochs}",
            )
            for batch in epoch_bar:
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                labels = batch["labels"].to(self.model.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss: torch.Tensor = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()

                loss_val = loss.item()
                losses.append(loss_val)
                epoch_bar.set_postfix(loss=f"{loss_val:.4f}")

        os.makedirs(self.config.output_dir, exist_ok=True)
        self.model.save_pretrained(self.config.output_dir)
        return losses
