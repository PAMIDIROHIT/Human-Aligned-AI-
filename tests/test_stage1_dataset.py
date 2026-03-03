"""Unit tests for Stage 1 SFT dataset loader.

Tests format_alpaca_example and verifies prompt templating
for examples with and without input fields.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from datasets import Dataset, DatasetDict

from src.stage1_sft.dataset import (
    format_alpaca_example,
    load_sft_dataset,
    ALPACA_PROMPT_TEMPLATE,
    ALPACA_PROMPT_TEMPLATE_NO_INPUT,
)


# ---------------------------------------------------------------------------
# format_alpaca_example
# ---------------------------------------------------------------------------

class TestFormatAlpacaExample:
    """Tests for the Alpaca prompt formatting function."""

    def test_with_input(self) -> None:
        """Example with an input field should use the full template."""
        example = {
            "instruction": "Summarize the text.",
            "input": "The quick brown fox jumps over the lazy dog.",
            "output": "A fox jumps over a dog.",
        }
        result = format_alpaca_example(example)
        assert "text" in result
        assert "### Instruction:\nSummarize the text." in result["text"]
        assert "### Input:\nThe quick brown fox" in result["text"]
        assert "### Response:\nA fox jumps over a dog." in result["text"]

    def test_without_input(self) -> None:
        """Example with empty input should use the no-input template."""
        example = {
            "instruction": "Tell me a joke.",
            "input": "",
            "output": "Why did the chicken cross the road?",
        }
        result = format_alpaca_example(example)
        assert "text" in result
        assert "### Input:" not in result["text"]
        assert "### Instruction:\nTell me a joke." in result["text"]
        assert "### Response:\nWhy did the chicken" in result["text"]

    def test_whitespace_only_input(self) -> None:
        """Whitespace-only input should be treated as no-input."""
        example = {
            "instruction": "Say hello.",
            "input": "   ",
            "output": "Hello!",
        }
        result = format_alpaca_example(example)
        assert "### Input:" not in result["text"]

    def test_strips_whitespace(self) -> None:
        """Leading/trailing whitespace should be stripped from all fields."""
        example = {
            "instruction": "  Do something.  ",
            "input": "  some context  ",
            "output": "  result  ",
        }
        result = format_alpaca_example(example)
        assert "  Do something.  " not in result["text"]
        assert "Do something." in result["text"]
        assert "some context" in result["text"]
        assert result["text"].endswith("result")

    def test_missing_input_key(self) -> None:
        """If 'input' key is missing entirely, should use no-input template."""
        example = {
            "instruction": "Say hi.",
            "input": None,
            "output": "Hi!",
        }
        # `None` should be falsy so the no-input path executes
        result = format_alpaca_example(example)
        assert "### Input:" not in result["text"]

    def test_output_is_single_key(self) -> None:
        """Result should contain exactly one key: 'text'."""
        example = {
            "instruction": "Test.",
            "input": "",
            "output": "Done.",
        }
        result = format_alpaca_example(example)
        assert list(result.keys()) == ["text"]


# ---------------------------------------------------------------------------
# load_sft_dataset (with mocking — avoids real network calls)
# ---------------------------------------------------------------------------

class TestLoadSftDataset:
    """Tests for the SFT dataset loading function (mocked)."""

    @staticmethod
    def _make_mock_dataset(n: int = 20) -> Dataset:
        """Create a small mock Alpaca-style dataset."""
        data = {
            "instruction": [f"instruction_{i}" for i in range(n)],
            "input": ["" if i % 2 == 0 else f"input_{i}" for i in range(n)],
            "output": [f"output_{i}" for i in range(n)],
        }
        return Dataset.from_dict(data)

    @patch("src.stage1_sft.dataset.load_dataset")
    def test_returns_dataset_dict(self, mock_load: MagicMock) -> None:
        """load_sft_dataset should return a DatasetDict with train/validation."""
        mock_load.return_value = self._make_mock_dataset(20)
        result = load_sft_dataset(dataset_name="tatsu-lab/alpaca", val_split_ratio=0.2)

        assert isinstance(result, DatasetDict)
        assert "train" in result
        assert "validation" in result

    @patch("src.stage1_sft.dataset.load_dataset")
    def test_text_column_exists(self, mock_load: MagicMock) -> None:
        """After formatting, each split should have a 'text' column."""
        mock_load.return_value = self._make_mock_dataset(10)
        result = load_sft_dataset(dataset_name="tatsu-lab/alpaca")

        for split in ["train", "validation"]:
            assert "text" in result[split].column_names

    @patch("src.stage1_sft.dataset.load_dataset")
    def test_max_samples_caps_data(self, mock_load: MagicMock) -> None:
        """max_samples should cap the total number of examples."""
        mock_load.return_value = self._make_mock_dataset(100)
        result = load_sft_dataset(
            dataset_name="tatsu-lab/alpaca",
            max_samples=5,
        )
        total = len(result["train"]) + len(result["validation"])
        assert total == 5

    @patch("src.stage1_sft.dataset.load_dataset")
    def test_unsupported_dataset_raises(self, mock_load: MagicMock) -> None:
        """An unsupported dataset name should raise ValueError."""
        mock_load.return_value = self._make_mock_dataset(5)
        with pytest.raises(ValueError, match="Unsupported dataset"):
            load_sft_dataset(dataset_name="some/unknown-dataset")

    @patch("src.stage1_sft.dataset.load_dataset")
    def test_val_split_ratio(self, mock_load: MagicMock) -> None:
        """Validation size should roughly match the requested ratio."""
        n = 100
        mock_load.return_value = self._make_mock_dataset(n)
        result = load_sft_dataset(
            dataset_name="tatsu-lab/alpaca",
            val_split_ratio=0.2,
        )
        val_ratio = len(result["validation"]) / (
            len(result["train"]) + len(result["validation"])
        )
        assert 0.15 <= val_ratio <= 0.25
