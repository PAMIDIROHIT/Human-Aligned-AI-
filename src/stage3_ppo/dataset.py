"""Stage 3 PPO — Dataset loader for prompts from HH-RLHF.

Loads prompt-only data from the Anthropic HH-RLHF dataset
for PPO rollout generation.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)


def extract_prompt(example: dict[str, Any]) -> dict[str, str]:
    """Extract the human prompt from an HH-RLHF conversation.

    Args:
        example: A dictionary with a 'chosen' key containing the
            full conversation string.

    Returns:
        A dictionary with a 'query' key containing the prompt text.
    """
    text = example["chosen"]
    # Find the last Assistant response and take everything before it
    last_assistant = text.rfind("\n\nAssistant:")
    if last_assistant != -1:
        prompt = text[:last_assistant].strip()
    else:
        prompt = text.strip()

    # Ensure the prompt ends with the Assistant marker for generation
    if not prompt.endswith("\n\nAssistant:"):
        prompt = prompt + "\n\nAssistant:"

    return {"query": prompt}


class PPOPromptDataset(TorchDataset):
    """PyTorch Dataset wrapper for PPO prompts.

    Wraps a HuggingFace Dataset of prompts into a PyTorch-compatible
    Dataset that returns tokenized queries for PPO rollouts.

    Attributes:
        dataset: The underlying HuggingFace Dataset.
        tokenizer: The tokenizer used for encoding queries.
        max_length: Maximum length for tokenized queries.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Any,
        max_length: int = 512,
    ) -> None:
        """Initialize the PPO prompt dataset.

        Args:
            dataset: HuggingFace Dataset with a 'query' column.
            tokenizer: Tokenizer for encoding queries.
            max_length: Maximum tokenization length.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of prompts.

        Returns:
            Length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a tokenized prompt by index.

        Args:
            idx: Index into the dataset.

        Returns:
            A dictionary with 'input_ids' and 'query' keys.
        """
        query = self.dataset[idx]["query"]
        encodings = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "query": query,
        }


def load_ppo_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    split: str = "train",
    seed: int = 42,
    max_samples: int | None = None,
) -> Dataset:
    """Load and preprocess prompt data for PPO rollouts.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Which split to use ('train' or 'test').
        seed: Random seed for reproducibility.
        max_samples: Optional cap on total number of prompts.

    Returns:
        A HuggingFace Dataset with 'query' column containing prompts.
    """
    logger.info("Loading PPO prompt dataset: %s (split=%s)", dataset_name, split)

    dataset = load_dataset(dataset_name, split=split)

    # Extract prompts
    dataset = dataset.map(
        extract_prompt,
        remove_columns=dataset.column_names,
        desc="Extracting prompts for PPO",
    )

    # Deduplicate
    df = dataset.to_pandas()
    df = df.drop_duplicates(subset=["query"])
    dataset = Dataset.from_pandas(df, preserve_index=False)

    # Shuffle
    dataset = dataset.shuffle(seed=seed)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info("Capped PPO dataset to %d prompts", len(dataset))

    logger.info("Loaded %d unique prompts for PPO", len(dataset))
    return dataset


def collator(data: list[dict[str, Any]]) -> dict[str, list]:
    """Custom collator for PPO that handles variable-length input_ids.

    Args:
        data: List of examples from the dataset, each containing
            'input_ids' and 'query'.

    Returns:
        A dictionary with batched 'input_ids' (list of tensors) and
        'query' (list of strings).
    """
    return {
        "input_ids": [d["input_ids"] for d in data],
        "query": [d["query"] for d in data],
    }
