"""Dataset classes and loading utilities for RLHF training."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """Dataset of (prompt, response) pairs for supervised fine-tuning.

    Each sample is a dict with keys:
        - ``input_ids``: tokenised prompt + response token IDs
        - ``attention_mask``: corresponding attention mask
        - ``labels``: same as ``input_ids`` (causal LM objective)
    """

    def __init__(
        self,
        prompts: List[str],
        responses: List[str],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        if len(prompts) != len(responses):
            raise ValueError(
                f"prompts and responses must have the same length, "
                f"got {len(prompts)} vs {len(responses)}"
            )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict[str, torch.Tensor]] = []

        for prompt, response in zip(prompts, responses):
            text = prompt + response
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            # For causal LM the labels are shifted inside the model; we use
            # -100 to mask padding tokens so they are ignored in the loss.
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class PreferenceDataset(Dataset):
    """Dataset of (prompt, chosen_response, rejected_response) triples for
    reward-model training.

    Each sample exposes tokenised tensors for both the *chosen* and *rejected*
    continuations so that a Bradley-Terry ranking loss can be applied.
    """

    def __init__(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        if not (len(prompts) == len(chosen) == len(rejected)):
            raise ValueError(
                "prompts, chosen, and rejected must all have the same length."
            )
        self.samples: List[Dict[str, torch.Tensor]] = []

        for prompt, chosen_resp, rejected_resp in zip(prompts, chosen, rejected):
            chosen_enc = tokenizer(
                prompt + chosen_resp,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_enc = tokenizer(
                prompt + rejected_resp,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.samples.append(
                {
                    "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
                    "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
                    "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
                    "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def load_sft_dataset(
    data: List[Dict[str, str]],
    tokenizer,
    max_length: int = 512,
) -> PromptDataset:
    """Create a :class:`PromptDataset` from a list of ``{"prompt": …, "response": …}``
    dicts.

    Args:
        data: List of dicts, each with ``"prompt"`` and ``"response"`` keys.
        tokenizer: A HuggingFace tokenizer.
        max_length: Maximum token length per sample.

    Returns:
        A :class:`PromptDataset` ready for SFT training.
    """
    prompts = [d["prompt"] for d in data]
    responses = [d["response"] for d in data]
    return PromptDataset(prompts, responses, tokenizer, max_length)


def load_preference_dataset(
    data: List[Dict[str, str]],
    tokenizer,
    max_length: int = 512,
) -> PreferenceDataset:
    """Create a :class:`PreferenceDataset` from a list of preference dicts.

    Args:
        data: List of dicts, each with ``"prompt"``, ``"chosen"``, and
              ``"rejected"`` keys.
        tokenizer: A HuggingFace tokenizer.
        max_length: Maximum token length per sample.

    Returns:
        A :class:`PreferenceDataset` ready for reward-model training.
    """
    prompts = [d["prompt"] for d in data]
    chosen = [d["chosen"] for d in data]
    rejected = [d["rejected"] for d in data]
    return PreferenceDataset(prompts, chosen, rejected, tokenizer, max_length)
