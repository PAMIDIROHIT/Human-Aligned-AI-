"""Reward model that scores a prompt-response pair with a scalar value."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel


class RewardModel(nn.Module):
    """Scalar reward model built on top of a pre-trained encoder.

    A linear "value head" maps the pooled hidden representation to a single
    scalar score.  During reward-model training the model is optimised with
    a Bradley-Terry ranking loss on (chosen, rejected) pairs.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        device: Target device.
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        device: str | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone: PreTrainedModel = AutoModel.from_pretrained(
            model_name_or_path
        ).to(device)
        hidden_size: int = self.backbone.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(device)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute scalar reward scores for a batch.

        Args:
            input_ids: ``(batch, seq_len)`` token IDs.
            attention_mask: ``(batch, seq_len)`` mask.

        Returns:
            ``(batch,)`` float tensor of reward scores.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use the last token's hidden state as the sequence representation
        # (appropriate for decoder-only models like GPT-2).
        last_hidden = outputs.last_hidden_state  # (B, T, H)
        # Find the index of the last non-padding token for each sample.
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        else:
            seq_lengths = torch.full(
                (last_hidden.size(0),),
                last_hidden.size(1) - 1,
                device=last_hidden.device,
            )
        # Gather the hidden state at the last real token position.
        seq_lengths = seq_lengths.clamp(min=0)
        pooled = last_hidden[
            torch.arange(last_hidden.size(0), device=last_hidden.device),
            seq_lengths,
        ]  # (B, H)
        scores = self.value_head(pooled).squeeze(-1)  # (B,)
        return scores

    # ------------------------------------------------------------------
    # Convenience scoring from raw text
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        prompts: list[str],
        responses: list[str],
        max_length: int = 512,
    ) -> list[float]:
        """Score a list of (prompt, response) pairs.

        Args:
            prompts: List of plain-text prompts.
            responses: List of plain-text responses.
            max_length: Maximum tokenised length.

        Returns:
            List of scalar reward scores (one per pair).
        """
        texts = [p + r for p, r in zip(prompts, responses)]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        scores = self(enc["input_ids"], enc["attention_mask"])
        return scores.tolist()

    def save_pretrained(self, output_dir: str) -> None:
        """Save backbone and tokenizer to *output_dir*.

        The value-head weights are saved separately as ``value_head.pt``.
        """
        import os

        os.makedirs(output_dir, exist_ok=True)
        self.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(
            self.value_head.state_dict(),
            os.path.join(output_dir, "value_head.pt"),
        )

    @classmethod
    def from_pretrained(cls, output_dir: str, device: str | None = None) -> "RewardModel":
        """Load a previously saved :class:`RewardModel`.

        Args:
            output_dir: Directory produced by :meth:`save_pretrained`.
            device: Target device.

        Returns:
            Loaded :class:`RewardModel` instance.
        """
        import os

        model = cls(output_dir, device=device)
        value_head_path = os.path.join(output_dir, "value_head.pt")
        if os.path.exists(value_head_path):
            model.value_head.load_state_dict(
                torch.load(value_head_path, map_location=model.device)
            )
        return model
