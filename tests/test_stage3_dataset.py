"""Unit tests for Stage 3 PPO dataset loader.

Tests extract_prompt, PPOPromptDataset wrapper, and collator.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import torch
from datasets import Dataset

from src.stage3_ppo.dataset import (
    extract_prompt,
    PPOPromptDataset,
    collator,
    load_ppo_dataset,
)


# ---------------------------------------------------------------------------
# extract_prompt
# ---------------------------------------------------------------------------

class TestExtractPrompt:
    """Tests for the PPO prompt extractor."""

    def test_standard_conversation(self) -> None:
        """Should extract all text before the last Assistant turn."""
        example = {
            "chosen": "\n\nHuman: How are you?\n\nAssistant: I'm fine."
        }
        result = extract_prompt(example)
        assert "How are you?" in result["query"]
        assert "I'm fine." not in result["query"]

    def test_appends_assistant_marker(self) -> None:
        """Output query should end with '\\n\\nAssistant:' for generation."""
        example = {
            "chosen": "\n\nHuman: Hello\n\nAssistant: World"
        }
        result = extract_prompt(example)
        assert result["query"].endswith("\n\nAssistant:")

    def test_multi_turn(self) -> None:
        """Multi-turn conversation should preserve earlier turns."""
        example = {
            "chosen": (
                "\n\nHuman: First"
                "\n\nAssistant: Reply1"
                "\n\nHuman: Second"
                "\n\nAssistant: Reply2"
            )
        }
        result = extract_prompt(example)
        assert "First" in result["query"]
        assert "Reply1" in result["query"]
        assert "Second" in result["query"]
        assert "Reply2" not in result["query"]

    def test_no_assistant_marker(self) -> None:
        """If no Assistant marker, entire text should be the prompt."""
        example = {"chosen": "Some raw text without markers"}
        result = extract_prompt(example)
        assert "Some raw text" in result["query"]
        assert result["query"].endswith("\n\nAssistant:")

    def test_returns_query_key(self) -> None:
        """Result should have a 'query' key."""
        result = extract_prompt({"chosen": "text"})
        assert "query" in result


# ---------------------------------------------------------------------------
# PPOPromptDataset
# ---------------------------------------------------------------------------

class TestPPOPromptDataset:
    """Tests for the PyTorch Dataset wrapper."""

    @staticmethod
    def _make_dataset() -> tuple[Dataset, MagicMock]:
        """Create a small HF dataset and a mock tokenizer."""
        ds = Dataset.from_dict({"query": ["Hello", "World", "Test"]})
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
        }
        return ds, tokenizer

    def test_len(self) -> None:
        """__len__ should match underlying dataset length."""
        ds, tok = self._make_dataset()
        ppo_ds = PPOPromptDataset(ds, tok, max_length=64)
        assert len(ppo_ds) == 3

    def test_getitem_has_input_ids(self) -> None:
        """__getitem__ should return a dict with 'input_ids' tensor."""
        ds, tok = self._make_dataset()
        ppo_ds = PPOPromptDataset(ds, tok, max_length=64)
        item = ppo_ds[0]
        assert "input_ids" in item
        assert isinstance(item["input_ids"], torch.Tensor)

    def test_getitem_has_query(self) -> None:
        """__getitem__ should include the raw query string."""
        ds, tok = self._make_dataset()
        ppo_ds = PPOPromptDataset(ds, tok, max_length=64)
        item = ppo_ds[0]
        assert "query" in item
        assert isinstance(item["query"], str)

    def test_tokenizer_called_with_truncation(self) -> None:
        """Tokenizer should be called with truncation=True."""
        ds, tok = self._make_dataset()
        ppo_ds = PPOPromptDataset(ds, tok, max_length=128)
        _ = ppo_ds[0]
        tok.assert_called_once_with(
            "Hello",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )


# ---------------------------------------------------------------------------
# collator
# ---------------------------------------------------------------------------

class TestCollator:
    """Tests for the PPO batch collator."""

    def test_batches_input_ids(self) -> None:
        """Should batch input_ids into a list of tensors."""
        data = [
            {"input_ids": torch.tensor([1, 2]), "query": "a"},
            {"input_ids": torch.tensor([3, 4, 5]), "query": "b"},
        ]
        batch = collator(data)
        assert len(batch["input_ids"]) == 2
        assert torch.equal(batch["input_ids"][0], torch.tensor([1, 2]))

    def test_batches_queries(self) -> None:
        """Should batch queries into a list of strings."""
        data = [
            {"input_ids": torch.tensor([1]), "query": "hello"},
            {"input_ids": torch.tensor([2]), "query": "world"},
        ]
        batch = collator(data)
        assert batch["query"] == ["hello", "world"]

    def test_empty_batch(self) -> None:
        """Empty batch should return empty lists."""
        batch = collator([])
        assert batch["input_ids"] == []
        assert batch["query"] == []


# ---------------------------------------------------------------------------
# load_ppo_dataset (mocked)
# ---------------------------------------------------------------------------

class TestLoadPPODataset:
    """Tests for the PPO dataset loading function (mocked)."""

    @patch("src.stage3_ppo.dataset.load_dataset")
    def test_returns_hf_dataset(self, mock_load: MagicMock) -> None:
        """load_ppo_dataset should return a HuggingFace Dataset."""
        data = {
            "chosen": [
                f"\n\nHuman: Q{i}\n\nAssistant: A{i}" for i in range(20)
            ],
            "rejected": [f"r{i}" for i in range(20)],
        }
        mock_load.return_value = Dataset.from_dict(data)
        result = load_ppo_dataset(max_samples=10)

        assert isinstance(result, Dataset)
        assert "query" in result.column_names
        assert len(result) <= 10

    @patch("src.stage3_ppo.dataset.load_dataset")
    def test_deduplication(self, mock_load: MagicMock) -> None:
        """Duplicate prompts should be removed."""
        data = {
            "chosen": [
                "\n\nHuman: Same\n\nAssistant: A",
                "\n\nHuman: Same\n\nAssistant: B",
                "\n\nHuman: Different\n\nAssistant: C",
            ],
            "rejected": ["r1", "r2", "r3"],
        }
        mock_load.return_value = Dataset.from_dict(data)
        result = load_ppo_dataset()

        # "Same" prompt appears twice but should deduplicate to one
        queries = result["query"]
        assert len(queries) == 2
