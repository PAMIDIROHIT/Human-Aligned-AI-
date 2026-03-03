"""Stage 1 SFT — Dataset loader for UltraChat-200k + OpenAssistant Guanaco.

Loads locally-downloaded datasets from data/sft/, applies prompt templating,
and returns a HuggingFace DatasetDict ready for SFTTrainer with packing.

Hardware target: 4x Tesla K80 (fp32, no quantization).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

logger = logging.getLogger(__name__)

# Chat-style prompt template
CHAT_TEMPLATE = "### Human:\n{user}\n\n### Assistant:\n{assistant}"


def format_ultrachat_example(example: dict[str, Any]) -> dict[str, str]:
    """Format a single UltraChat-200k multi-turn conversation into text.

    Args:
        example: A single example with 'messages' field.

    Returns:
        Dictionary with a 'text' field containing the formatted conversation.
    """
    messages = example.get("messages", [])
    if not messages:
        prompt = example.get("prompt", "")
        response = example.get("response", "")
        if prompt and response:
            return {"text": CHAT_TEMPLATE.format(user=prompt.strip(), assistant=response.strip())}
        return {"text": ""}

    parts = []
    for i in range(0, len(messages) - 1, 2):
        user_msg = messages[i].get("content", "") if isinstance(messages[i], dict) else str(messages[i])
        if i + 1 < len(messages):
            asst_msg = messages[i + 1].get("content", "") if isinstance(messages[i + 1], dict) else str(messages[i + 1])
            parts.append(CHAT_TEMPLATE.format(user=user_msg.strip(), assistant=asst_msg.strip()))

    if len(messages) % 2 == 1:
        last = messages[-1]
        user_msg = last.get("content", "") if isinstance(last, dict) else str(last)
        parts.append(f"### Human:\n{user_msg.strip()}")

    return {"text": "\n\n".join(parts)}


def format_guanaco_example(example: dict[str, Any]) -> dict[str, str]:
    """Format a single OpenAssistant Guanaco example.

    Args:
        example: A single example with 'text' field.

    Returns:
        Dictionary with a 'text' field.
    """
    text = example.get("text", "")
    return {"text": text.strip()}


def load_sft_dataset(
    config: dict | None = None,
    val_split_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int | None = None,
    dataset_name: str | None = None,
) -> DatasetDict:
    """Load SFT datasets from local disk (UltraChat-200k + Guanaco).

    Args:
        config: Full params.yaml config dict (optional).
        val_split_ratio: Fraction to reserve for validation.
        seed: Random seed for shuffling/splitting.
        max_samples: Maximum samples to use (None = use all).
        dataset_name: Legacy parameter for backward compatibility.

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
    datasets_to_combine = []

    data_cfg = config.get("data", {}) if config else {}
    primary_path = data_cfg.get("sft_primary", "data/sft/ultrachat_200k")
    supplement_path = data_cfg.get("sft_supplement", "data/sft/openassistant_guanaco")

    # Load UltraChat-200k (primary)
    primary_dir = Path(primary_path)
    if primary_dir.exists():
        logger.info("Loading UltraChat-200k from local: %s", primary_dir)
        ds = load_from_disk(str(primary_dir))
        ds = ds.map(
            format_ultrachat_example,
            remove_columns=ds.column_names,
            desc="Formatting UltraChat-200k",
            num_proc=4,
        )
        ds = ds.filter(lambda x: len(x["text"]) > 50)
        datasets_to_combine.append(ds)
        logger.info("UltraChat-200k: %d samples loaded", len(ds))
    else:
        logger.warning("UltraChat-200k not found at %s", primary_dir)

    # Load OpenAssistant Guanaco (supplement)
    supplement_dir = Path(supplement_path)
    if supplement_dir.exists():
        logger.info("Loading Guanaco from local: %s", supplement_dir)
        ds = load_from_disk(str(supplement_dir))
        ds = ds.map(
            format_guanaco_example,
            remove_columns=ds.column_names,
            desc="Formatting Guanaco",
        )
        ds = ds.filter(lambda x: len(x["text"]) > 50)
        datasets_to_combine.append(ds)
        logger.info("Guanaco: %d samples loaded", len(ds))
    else:
        logger.warning("Guanaco not found at %s", supplement_dir)

    if not datasets_to_combine:
        raise FileNotFoundError(
            f"No SFT datasets found at {primary_path} or {supplement_path}. "
            "Run the dataset download script first."
        )

    combined = concatenate_datasets(datasets_to_combine)
    combined = combined.shuffle(seed=seed)
    logger.info("Combined SFT dataset: %d samples", len(combined))

    if max_samples is not None:
        combined = combined.select(range(min(max_samples, len(combined))))
        logger.info("Capped to %d samples", len(combined))

    split = combined.train_test_split(test_size=val_split_ratio, seed=seed)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    logger.info(
        "SFT dataset ready -- train: %d, validation: %d",
        len(dataset_dict["train"]),
        len(dataset_dict["validation"]),
    )
    return dataset_dict


def get_formatting_func() -> None:
    """Placeholder for compatibility."""
    return None
