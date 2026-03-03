"""Unit tests for data utilities."""

from __future__ import annotations

import pytest

from rlhf.data.dataset import (
    PreferenceDataset,
    PromptDataset,
    load_preference_dataset,
    load_sft_dataset,
)


class _TokenizerOutput(dict):
    """Dict subclass that supports ``.to(device)`` like BatchEncoding."""

    def to(self, device):
        return _TokenizerOutput({k: v.to(device) for k, v in self.items()})


class _DummyTokenizer:
    """Minimal tokenizer stub that avoids loading a real model."""

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        text,
        truncation=True,
        max_length=16,
        padding="max_length",
        return_tensors="pt",
    ):
        import torch

        if isinstance(text, list):
            batch = [self._encode(t, max_length) for t in text]
            input_ids = torch.cat([b["input_ids"] for b in batch], dim=0)
            attention_mask = torch.cat([b["attention_mask"] for b in batch], dim=0)
            return _TokenizerOutput(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
        return _TokenizerOutput(self._encode(text, max_length))

    def _encode(self, text: str, max_length: int):
        import torch

        ids = [ord(c) % 50 + 2 for c in text[:max_length]]
        pad_len = max_length - len(ids)
        ids = ids + [self.pad_token_id] * pad_len
        mask = [1] * (max_length - pad_len) + [0] * pad_len
        return {
            "input_ids": torch.tensor(ids).unsqueeze(0),
            "attention_mask": torch.tensor(mask).unsqueeze(0),
        }

    def save_pretrained(self, output_dir: str) -> None:
        pass


@pytest.fixture
def tokenizer():
    return _DummyTokenizer()


# ---------------------------------------------------------------------------
# PromptDataset
# ---------------------------------------------------------------------------


def test_prompt_dataset_length(tokenizer):
    prompts = ["Hello ", "World "]
    responses = ["there", "here"]
    ds = PromptDataset(prompts, responses, tokenizer, max_length=16)
    assert len(ds) == 2


def test_prompt_dataset_keys(tokenizer):
    ds = PromptDataset(["Hi "], ["there"], tokenizer, max_length=16)
    sample = ds[0]
    assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}


def test_prompt_dataset_tensor_shape(tokenizer):
    import torch

    ds = PromptDataset(["Hi "], ["there"], tokenizer, max_length=16)
    sample = ds[0]
    assert sample["input_ids"].shape == torch.Size([16])
    assert sample["labels"].shape == torch.Size([16])


def test_prompt_dataset_length_mismatch(tokenizer):
    with pytest.raises(ValueError, match="same length"):
        PromptDataset(["a", "b"], ["x"], tokenizer)


def test_load_sft_dataset(tokenizer):
    data = [
        {"prompt": "Q: What is AI? ", "response": "A: Artificial Intelligence."},
        {"prompt": "Q: Hello? ", "response": "A: Hi!"},
    ]
    ds = load_sft_dataset(data, tokenizer, max_length=32)
    assert len(ds) == 2


# ---------------------------------------------------------------------------
# PreferenceDataset
# ---------------------------------------------------------------------------


def test_preference_dataset_length(tokenizer):
    prompts = ["P1 ", "P2 "]
    chosen = ["good1", "good2"]
    rejected = ["bad1", "bad2"]
    ds = PreferenceDataset(prompts, chosen, rejected, tokenizer, max_length=16)
    assert len(ds) == 2


def test_preference_dataset_keys(tokenizer):
    ds = PreferenceDataset(["P "], ["good"], ["bad"], tokenizer, max_length=16)
    sample = ds[0]
    assert "chosen_input_ids" in sample
    assert "rejected_input_ids" in sample
    assert "chosen_attention_mask" in sample
    assert "rejected_attention_mask" in sample


def test_preference_dataset_mismatch(tokenizer):
    with pytest.raises(ValueError, match="same length"):
        PreferenceDataset(["a"], ["x", "y"], ["z"], tokenizer)


def test_load_preference_dataset(tokenizer):
    data = [
        {"prompt": "P ", "chosen": "great answer", "rejected": "bad answer"},
    ]
    ds = load_preference_dataset(data, tokenizer, max_length=16)
    assert len(ds) == 1
