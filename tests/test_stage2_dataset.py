"""Unit tests for Stage 2 Reward Model dataset loader.

Tests HH-RLHF formatting helpers: extract_prompt_from_conversation,
format_hh_rlhf_pair, format_prompt_only, and load_reward_dataset.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from datasets import Dataset, DatasetDict

from src.stage2_reward.dataset import (
    extract_prompt_from_conversation,
    format_hh_rlhf_pair,
    format_prompt_only,
    load_reward_dataset,
)


# ---------------------------------------------------------------------------
# extract_prompt_from_conversation
# ---------------------------------------------------------------------------

class TestExtractPrompt:
    """Tests for extracting the prompt from HH-RLHF conversation strings."""

    def test_standard_conversation(self) -> None:
        """Should return everything before the final Assistant turn."""
        text = (
            "\n\nHuman: What is 2+2?"
            "\n\nAssistant: It is 4."
        )
        prompt = extract_prompt_from_conversation(text)
        assert "What is 2+2?" in prompt
        assert "It is 4." not in prompt

    def test_multi_turn(self) -> None:
        """Multi-turn conversation: should keep all turns except last Assistant."""
        text = (
            "\n\nHuman: Hi"
            "\n\nAssistant: Hello!"
            "\n\nHuman: How are you?"
            "\n\nAssistant: I am fine."
        )
        prompt = extract_prompt_from_conversation(text)
        assert "How are you?" in prompt
        assert "Hello!" in prompt  # earlier assistant turn is kept
        assert "I am fine." not in prompt

    def test_no_assistant(self) -> None:
        """If no Assistant marker exists, return the whole text."""
        text = "\n\nHuman: Just a question"
        prompt = extract_prompt_from_conversation(text)
        assert prompt == text

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert extract_prompt_from_conversation("") == ""


# ---------------------------------------------------------------------------
# format_hh_rlhf_pair
# ---------------------------------------------------------------------------

class TestFormatHHRLHFPair:
    """Tests for formatting chosen/rejected pairs."""

    def test_strips_whitespace(self) -> None:
        """Should strip leading/trailing whitespace."""
        example = {
            "chosen": "  \n\nHuman: Hi\n\nAssistant: Hello  ",
            "rejected": "  \n\nHuman: Hi\n\nAssistant: Go away  ",
        }
        result = format_hh_rlhf_pair(example)
        assert not result["chosen"].startswith(" ")
        assert not result["rejected"].endswith(" ")

    def test_preserves_content(self) -> None:
        """Core content should be preserved after formatting."""
        example = {
            "chosen": "\n\nHuman: Q?\n\nAssistant: Good answer",
            "rejected": "\n\nHuman: Q?\n\nAssistant: Bad answer",
        }
        result = format_hh_rlhf_pair(example)
        assert "Good answer" in result["chosen"]
        assert "Bad answer" in result["rejected"]

    def test_returns_two_keys(self) -> None:
        """Result dict should have exactly 'chosen' and 'rejected' keys."""
        result = format_hh_rlhf_pair({"chosen": "a", "rejected": "b"})
        assert set(result.keys()) == {"chosen", "rejected"}


# ---------------------------------------------------------------------------
# format_prompt_only
# ---------------------------------------------------------------------------

class TestFormatPromptOnly:
    """Tests for extracting prompt-only text for PPO."""

    def test_extracts_query(self) -> None:
        """Should produce a 'query' key with the prompt portion."""
        example = {
            "chosen": "\n\nHuman: Hello\n\nAssistant: Hi there!"
        }
        result = format_prompt_only(example)
        assert "query" in result
        assert "Hello" in result["query"]
        assert "Hi there!" not in result["query"]

    def test_single_key_output(self) -> None:
        """Output should have exactly one key: 'query'."""
        result = format_prompt_only({"chosen": "text"})
        assert list(result.keys()) == ["query"]


# ---------------------------------------------------------------------------
# load_reward_dataset (mocked)
# ---------------------------------------------------------------------------

class TestLoadRewardDataset:
    """Tests for the reward dataset loading function (mocked)."""

    @staticmethod
    def _make_mock_hh_dataset(n: int = 20) -> DatasetDict:
        """Create a small mock HH-RLHF DatasetDict."""
        data = {
            "chosen": [
                f"\n\nHuman: Q{i}\n\nAssistant: Good {i}" for i in range(n)
            ],
            "rejected": [
                f"\n\nHuman: Q{i}\n\nAssistant: Bad {i}" for i in range(n)
            ],
        }
        ds = Dataset.from_dict(data)
        split = ds.train_test_split(test_size=0.2, seed=42)
        return DatasetDict({"train": split["train"], "test": split["test"]})

    @patch("src.stage2_reward.dataset.load_dataset")
    def test_returns_dataset_dict(self, mock_load: MagicMock) -> None:
        """load_reward_dataset should return a DatasetDict with train/test."""
        mock_load.return_value = self._make_mock_hh_dataset(20)
        result = load_reward_dataset()

        assert isinstance(result, DatasetDict)
        assert "train" in result
        assert "test" in result

    @patch("src.stage2_reward.dataset.load_dataset")
    def test_chosen_rejected_columns(self, mock_load: MagicMock) -> None:
        """Each split should have 'chosen' and 'rejected' columns."""
        mock_load.return_value = self._make_mock_hh_dataset(10)
        result = load_reward_dataset()

        for split in ["train", "test"]:
            assert "chosen" in result[split].column_names
            assert "rejected" in result[split].column_names

    @patch("src.stage2_reward.dataset.load_dataset")
    def test_max_samples(self, mock_load: MagicMock) -> None:
        """max_samples should cap each split's size."""
        mock_load.return_value = self._make_mock_hh_dataset(50)
        result = load_reward_dataset(max_samples=3)

        for split_name in result:
            assert len(result[split_name]) <= 3

    @patch("src.stage2_reward.dataset.load_dataset")
    def test_creates_test_split_if_missing(self, mock_load: MagicMock) -> None:
        """If no 'test' split, it should create one from training data."""
        # Return a DatasetDict with only 'train'
        data = {
            "chosen": [f"c{i}" for i in range(20)],
            "rejected": [f"r{i}" for i in range(20)],
        }
        ds = Dataset.from_dict(data)
        mock_load.return_value = DatasetDict({"train": ds})

        result = load_reward_dataset()
        assert "test" in result
        assert len(result["test"]) > 0
