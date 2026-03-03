"""Stage 2 Reward Model — Dataset loader for Anthropic HH-RLHF.

Loads chosen/rejected pairs and formats them for RewardTrainer.
Each example contains a 'chosen' and 'rejected' response to the same prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


def extract_prompt_from_conversation(text: str) -> str:
    """Extract the human prompt from an HH-RLHF conversation string.

    The HH-RLHF dataset uses '\\n\\nHuman:' and '\\n\\nAssistant:' delimiters.
    This extracts all content up to and including the last Human turn.

    Args:
        text: Raw conversation string from HH-RLHF.

    Returns:
        The prompt portion of the conversation (everything before the
        final Assistant response).
    """
    # Find the last occurrence of "\n\nAssistant:"
    last_assistant_idx = text.rfind("\n\nAssistant:")
    if last_assistant_idx == -1:
        return text
    return text[:last_assistant_idx].strip()


def format_hh_rlhf_pair(example: dict[str, Any]) -> dict[str, str]:
    """Format an HH-RLHF example into chosen/rejected text pairs.

    Args:
        example: A dictionary with 'chosen' and 'rejected' keys,
            each containing a full conversation string.

    Returns:
        A dictionary with 'chosen' and 'rejected' keys containing
        the formatted conversation strings.
    """
    return {
        "chosen": example["chosen"].strip(),
        "rejected": example["rejected"].strip(),
    }


def format_prompt_only(example: dict[str, Any]) -> dict[str, str]:
    """Extract prompt-only from an HH-RLHF example (for PPO rollouts).

    Args:
        example: A dictionary with a 'chosen' key containing the
            full conversation string.

    Returns:
        A dictionary with a 'query' key containing only the prompt.
    """
    prompt = extract_prompt_from_conversation(example["chosen"])
    return {"query": prompt}


def load_reward_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    max_length: int = 512,
    seed: int = 42,
    max_samples: int | None = None,
    test_split_ratio: float = 0.1,
) -> DatasetDict:
    """Load and preprocess the HH-RLHF dataset for reward model training.

    Args:
        dataset_name: HuggingFace dataset identifier.
        max_length: Maximum sequence length (used for downstream filtering).
        seed: Random seed for reproducible splitting.
        max_samples: Optional cap on total samples (for debugging).
        test_split_ratio: Fraction of training data for test set if no
            test split exists in the dataset.

    Returns:
        A DatasetDict with 'train' and 'test' splits, each containing
        'chosen' and 'rejected' columns.
    """
    logger.info("Loading HH-RLHF dataset: %s", dataset_name)

    dataset = load_dataset(dataset_name)

    # The HH-RLHF dataset has 'train' and 'test' splits
    if "test" not in dataset:
        logger.info("No test split found, creating one from train split")
        split = dataset["train"].train_test_split(
            test_size=test_split_ratio, seed=seed
        )
        dataset = DatasetDict({"train": split["train"], "test": split["test"]})

    # Format pairs
    for split_name in dataset:
        dataset[split_name] = dataset[split_name].map(
            format_hh_rlhf_pair,
            desc=f"Formatting HH-RLHF pairs ({split_name})",
        )

    # Cap samples if requested
    if max_samples is not None:
        for split_name in dataset:
            n = min(max_samples, len(dataset[split_name]))
            dataset[split_name] = dataset[split_name].select(range(n))
            logger.info("Capped %s to %d samples", split_name, n)

    logger.info(
        "HH-RLHF loaded — train: %d, test: %d",
        len(dataset["train"]),
        len(dataset["test"]),
    )
    return dataset


def load_prompt_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    split: str = "train",
    seed: int = 42,
    max_samples: int | None = None,
) -> Dataset:
    """Load prompt-only data from HH-RLHF for PPO rollouts.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Which split to load ('train' or 'test').
        seed: Random seed for reproducibility.
        max_samples: Optional cap on total samples.

    Returns:
        A Dataset with a 'query' column containing prompt strings.
    """
    logger.info("Loading HH-RLHF prompts for PPO: %s (split=%s)", dataset_name, split)

    dataset = load_dataset(dataset_name, split=split)

    # Extract prompts
    dataset = dataset.map(
        format_prompt_only,
        remove_columns=dataset.column_names,
        desc="Extracting prompts",
    )

    # Deduplicate prompts
    df = dataset.to_pandas()
    df = df.drop_duplicates(subset=["query"])
    dataset = Dataset.from_pandas(df, preserve_index=False)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    logger.info("Loaded %d unique prompts for PPO", len(dataset))
    return dataset
