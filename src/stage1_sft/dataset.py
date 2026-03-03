"""Stage 1 SFT — Dataset loader for Alpaca/Dolly with prompt templating.

Loads the tatsu-lab/alpaca dataset, applies the instruction prompt template,
and returns a HuggingFace Dataset ready for SFTTrainer with packing support.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

# Alpaca-style prompt template
ALPACA_PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_PROMPT_TEMPLATE_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_alpaca_example(example: dict[str, Any]) -> dict[str, str]:
    """Format a single Alpaca example into the instruction prompt template.

    Args:
        example: A dictionary with keys 'instruction', 'input', and 'output'.

    Returns:
        A dictionary with a single 'text' key containing the formatted prompt.
    """
    if example.get("input") and example["input"].strip():
        text = ALPACA_PROMPT_TEMPLATE.format(
            instruction=example["instruction"].strip(),
            input=example["input"].strip(),
            output=example["output"].strip(),
        )
    else:
        text = ALPACA_PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=example["instruction"].strip(),
            output=example["output"].strip(),
        )
    return {"text": text}


def load_sft_dataset(
    dataset_name: str = "tatsu-lab/alpaca",
    val_split_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int | None = None,
) -> DatasetDict:
    """Load and preprocess the SFT dataset with train/validation split.

    Args:
        dataset_name: HuggingFace dataset identifier. Supports 'tatsu-lab/alpaca'
            and 'databricks/databricks-dolly-15k'.
        val_split_ratio: Fraction of data to reserve for validation.
        seed: Random seed for reproducible splitting.
        max_samples: Optional cap on total samples (for debugging).

    Returns:
        A DatasetDict with 'train' and 'validation' splits, each containing
        a 'text' column with the formatted prompt.

    Raises:
        ValueError: If the dataset_name is not supported.
    """
    logger.info("Loading dataset: %s", dataset_name)

    if "alpaca" in dataset_name.lower():
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.map(
            format_alpaca_example,
            remove_columns=dataset.column_names,
            desc="Formatting Alpaca prompts",
        )
    elif "dolly" in dataset_name.lower():
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.map(
            _format_dolly_example,
            remove_columns=dataset.column_names,
            desc="Formatting Dolly prompts",
        )
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            "Use 'tatsu-lab/alpaca' or 'databricks/databricks-dolly-15k'."
        )

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info("Capped dataset to %d samples", len(dataset))

    # Train/validation split
    split = dataset.train_test_split(test_size=val_split_ratio, seed=seed)
    dataset_dict = DatasetDict(
        {"train": split["train"], "validation": split["test"]}
    )

    logger.info(
        "Dataset loaded — train: %d, validation: %d",
        len(dataset_dict["train"]),
        len(dataset_dict["validation"]),
    )
    return dataset_dict


def _format_dolly_example(example: dict[str, Any]) -> dict[str, str]:
    """Format a Dolly example into the instruction prompt template.

    Args:
        example: A dictionary with keys 'instruction', 'context', and 'response'.

    Returns:
        A dictionary with a single 'text' key containing the formatted prompt.
    """
    if example.get("context") and example["context"].strip():
        text = ALPACA_PROMPT_TEMPLATE.format(
            instruction=example["instruction"].strip(),
            input=example["context"].strip(),
            output=example["response"].strip(),
        )
    else:
        text = ALPACA_PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=example["instruction"].strip(),
            output=example["response"].strip(),
        )
    return {"text": text}


def get_formatting_func() -> None:
    """Return None to signal SFTTrainer to use the 'text' column directly.

    When packing is enabled and the dataset has a 'text' column,
    SFTTrainer uses it directly without a formatting function.

    Returns:
        None — SFTTrainer will use the 'text' column from the dataset.
    """
    return None
