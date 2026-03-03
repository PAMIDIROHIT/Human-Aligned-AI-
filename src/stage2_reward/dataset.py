"""Stage 2 Reward Model — Dataset loader for Anthropic HH-RLHF + UltraFeedback.

Loads locally-downloaded preference datasets from data/reward/,
formats chosen/rejected pairs for RewardTrainer.

Hardware target: 4x Tesla K80 (fp32, no quantization).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

logger = logging.getLogger(__name__)


def format_hh_rlhf_pair(example: dict[str, Any]) -> dict[str, str]:
    """Format an Anthropic HH-RLHF chosen/rejected pair.

    Args:
        example: Example with 'chosen' and 'rejected' fields.

    Returns:
        Dictionary with 'chosen' and 'rejected' text.
    """
    return {
        "chosen": example["chosen"].strip(),
        "rejected": example["rejected"].strip(),
    }


def format_ultrafeedback_pair(example: dict[str, Any]) -> dict[str, str]:
    """Format an UltraFeedback binarized preference pair.

    UltraFeedback has 'chosen' and 'rejected' as lists of message dicts.

    Args:
        example: Example from ultrafeedback_binarized.

    Returns:
        Dictionary with 'chosen' and 'rejected' text.
    """
    chosen = example.get("chosen", [])
    rejected = example.get("rejected", [])

    # Handle list-of-dicts format (messages)
    if isinstance(chosen, list):
        chosen_text = "\n".join(
            m.get("content", str(m)) if isinstance(m, dict) else str(m)
            for m in chosen
        )
    else:
        chosen_text = str(chosen)

    if isinstance(rejected, list):
        rejected_text = "\n".join(
            m.get("content", str(m)) if isinstance(m, dict) else str(m)
            for m in rejected
        )
    else:
        rejected_text = str(rejected)

    return {
        "chosen": chosen_text.strip(),
        "rejected": rejected_text.strip(),
    }


def extract_prompt_from_conversation(text: str) -> str:
    """Extract the prompt (human turn) from an HH-RLHF conversation.

    Args:
        text: Full conversation text.

    Returns:
        The prompt portion up to the last assistant turn.
    """
    last_assistant_idx = text.rfind("\n\nAssistant:")
    if last_assistant_idx == -1:
        return text
    return text[:last_assistant_idx].strip()


def format_prompt_only(example: dict[str, Any]) -> dict[str, str]:
    """Extract just the prompt from a preference pair for PPO use.

    Args:
        example: Example with 'chosen' field.

    Returns:
        Dictionary with 'query' field.
    """
    prompt = extract_prompt_from_conversation(example["chosen"])
    return {"query": prompt}


def load_reward_dataset(
    config: dict | None = None,
    seed: int = 42,
    max_samples: int | None = None,
    test_split_ratio: float = 0.1,
    dataset_name: str | None = None,
    max_length: int = 512,
) -> DatasetDict:
    """Load reward model datasets from local disk.

    Combines Anthropic HH-RLHF + UltraFeedback binarized.

    Args:
        config: Full params.yaml config dict (optional).
        seed: Random seed.
        max_samples: Maximum samples per split.
        test_split_ratio: Fraction for test split.
        dataset_name: Legacy parameter.
        max_length: Unused, kept for compatibility.

    Returns:
        DatasetDict with 'train' and 'test' splits, each with 'chosen'/'rejected'.
    """
    datasets_to_combine = []

    data_cfg = config.get("data", {}) if config else {}
    primary_path = data_cfg.get("reward_primary", "data/reward/anthropic_hh_rlhf")
    supplement_path = data_cfg.get("reward_supplement", "data/reward/ultrafeedback_binarized")

    # Load Anthropic HH-RLHF (primary)
    primary_dir = Path(primary_path)
    if primary_dir.exists():
        logger.info("Loading Anthropic HH-RLHF from local: %s", primary_dir)
        ds = load_from_disk(str(primary_dir))
        ds = ds.map(
            format_hh_rlhf_pair,
            desc="Formatting HH-RLHF pairs",
        )
        # Keep only chosen/rejected columns
        keep_cols = {"chosen", "rejected"}
        remove_cols = [c for c in ds.column_names if c not in keep_cols]
        if remove_cols:
            ds = ds.remove_columns(remove_cols)
        datasets_to_combine.append(ds)
        logger.info("HH-RLHF: %d pairs loaded", len(ds))
    else:
        logger.warning("HH-RLHF not found at %s", primary_dir)

    # Load UltraFeedback binarized (supplement)
    supplement_dir = Path(supplement_path)
    if supplement_dir.exists():
        logger.info("Loading UltraFeedback from local: %s", supplement_dir)
        ds = load_from_disk(str(supplement_dir))
        ds = ds.map(
            format_ultrafeedback_pair,
            remove_columns=ds.column_names,
            desc="Formatting UltraFeedback pairs",
        )
        datasets_to_combine.append(ds)
        logger.info("UltraFeedback: %d pairs loaded", len(ds))
    else:
        logger.warning("UltraFeedback not found at %s", supplement_dir)

    if not datasets_to_combine:
        raise FileNotFoundError(
            f"No reward datasets found at {primary_path} or {supplement_path}. "
            "Run the dataset download script first."
        )

    combined = concatenate_datasets(datasets_to_combine)
    combined = combined.shuffle(seed=seed)
    logger.info("Combined reward dataset: %d pairs", len(combined))

    if max_samples is not None:
        combined = combined.select(range(min(max_samples, len(combined))))

    # Train/test split
    split = combined.train_test_split(test_size=test_split_ratio, seed=seed)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "test": split["test"],
    })

    logger.info(
        "Reward dataset -- train: %d, test: %d",
        len(dataset_dict["train"]),
        len(dataset_dict["test"]),
    )
    return dataset_dict


def load_prompt_dataset(
    config: dict | None = None,
    split: str = "train",
    seed: int = 42,
    max_samples: int | None = None,
    dataset_name: str | None = None,
) -> Dataset:
    """Load prompts from HH-RLHF for PPO (backward compatibility).

    Args:
        config: Full params.yaml config dict (optional).
        split: Unused, loads all data.
        seed: Random seed.
        max_samples: Maximum samples.
        dataset_name: Legacy parameter.

    Returns:
        Dataset with 'query' field.
    """
    data_cfg = config.get("data", {}) if config else {}
    primary_path = data_cfg.get("reward_primary", "data/reward/anthropic_hh_rlhf")

    primary_dir = Path(primary_path)
    if not primary_dir.exists():
        raise FileNotFoundError(f"HH-RLHF not found at {primary_dir}")

    ds = load_from_disk(str(primary_dir))
    ds = ds.map(
        format_prompt_only,
        remove_columns=ds.column_names,
        desc="Extracting prompts",
    )

    df = ds.to_pandas()
    df = df.drop_duplicates(subset=["query"])
    ds = Dataset.from_pandas(df, preserve_index=False)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    logger.info("Loaded %d unique prompts", len(ds))
    return ds
