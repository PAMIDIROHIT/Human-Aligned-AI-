"""Reward model trainer using a Bradley-Terry ranking loss."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import RewardModelConfig
from ..data.dataset import PreferenceDataset
from ..models.reward_model import RewardModel


class RewardTrainer:
    """Train a :class:`~rlhf.models.RewardModel` on human preference data.

    The loss is the negative log-likelihood of the Bradley-Terry model:

    .. math::

        \\mathcal{L} = -\\mathbb{E}[\\log \\sigma(r_{\\text{chosen}} - r_{\\text{rejected}})]

    Args:
        model: The reward model to train.
        config: Hyper-parameter configuration.
    """

    def __init__(self, model: RewardModel, config: RewardModelConfig) -> None:
        self.model = model
        self.config = config

    def train(self, dataset: PreferenceDataset) -> list[float]:
        """Run the reward-model training loop.

        Args:
            dataset: A :class:`~rlhf.data.PreferenceDataset` of
                     (prompt, chosen, rejected) triples.

        Returns:
            List of per-step loss values.
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
        )

        params = list(self.model.backbone.parameters()) + list(
            self.model.value_head.parameters()
        )
        optimizer = AdamW(
            params,
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

        self.model.train()
        losses: list[float] = []

        for epoch in range(self.config.num_train_epochs):
            epoch_bar = tqdm(
                dataloader,
                desc=f"Reward epoch {epoch + 1}/{self.config.num_train_epochs}",
            )
            for batch in epoch_bar:
                chosen_ids = batch["chosen_input_ids"].to(self.model.device)
                chosen_mask = batch["chosen_attention_mask"].to(self.model.device)
                rejected_ids = batch["rejected_input_ids"].to(self.model.device)
                rejected_mask = batch["rejected_attention_mask"].to(self.model.device)

                chosen_scores = self.model(chosen_ids, chosen_mask)
                rejected_scores = self.model(rejected_ids, rejected_mask)

                # Bradley-Terry loss: maximise P(chosen > rejected)
                loss: torch.Tensor = -F.logsigmoid(
                    chosen_scores - rejected_scores
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                loss_val = loss.item()
                losses.append(loss_val)
                epoch_bar.set_postfix(loss=f"{loss_val:.4f}")

        os.makedirs(self.config.output_dir, exist_ok=True)
        self.model.save_pretrained(self.config.output_dir)
        return losses
