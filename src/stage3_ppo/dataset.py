"""Stage 3 PPO — Dataset loader for prompts from UltraFeedback + PKU-SafeRLHF.

Loads locally-downloaded prompt datasets from data/ppo/ for PPO rollout generation.

Hardware target: 4x Tesla K80 (fp32, no quantization).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_from_disk
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)


def extract_ultrafeedback_prompt(example: dict[str, Any]) -> dict[str, str]:
    """Extract prompt from UltraFeedback binarized generation split.

    Args:
        example: Example from ultrafeedback_binarized.

    Returns:
        Dictionary with 'query' field.
    """
    # UltraFeedback may have 'prompt', 'instruction', or 'chosen' field
    prompt = example.get("prompt", "")
    if not prompt:
        prompt = example.get("instruction", "")
    if not prompt:
        # Try extracting from chosen (list of messages)
        chosen = example.get("chosen", [])
        if isinstance(chosen, list) and len(chosen) > 0:
            first = chosen[0]
            prompt = first.get("content", str(first)) if isinstance(first, dict) else str(first)
        elif isinstance(chosen, str):
            prompt = chosen.split("\n")[0]

    if not prompt.strip():
        return {"query": ""}

    return {"query": f"### Human:\n{prompt.strip()}\n\n### Assistant:"}


def extract_saferlhf_prompt(example: dict[str, Any]) -> dict[str, str]:
    """Extract prompt from PKU-SafeRLHF dataset.

    Args:
        example: Example from PKU-SafeRLHF.

    Returns:
        Dictionary with 'query' field.
    """
    prompt = example.get("prompt", "")
    if not prompt:
        prompt = example.get("question", "")
    if not prompt.strip():
        return {"query": ""}

    return {"query": f"### Human:\n{prompt.strip()}\n\n### Assistant:"}


class PPOPromptDataset(TorchDataset):
    """Torch-compatible wrapper around a HuggingFace Dataset for PPO prompts.

    Args:
        dataset: HuggingFace Dataset with 'query' field.
        tokenizer: Tokenizer for encoding queries.
        max_length: Maximum sequence length for tokenization.
    """

    def __init__(self, dataset: Dataset, tokenizer: Any, max_length: int = 512) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
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
    config: dict | None = None,
    seed: int = 42,
    max_samples: int | None = None,
    dataset_name: str | None = None,
    split: str = "train",
) -> Dataset:
    """Load PPO prompt datasets from local disk.

    Combines UltraFeedback (generation prompts) + PKU-SafeRLHF (safety prompts).

    Args:
        config: Full params.yaml config dict (optional).
        seed: Random seed.
        max_samples: Maximum number of prompts.
        dataset_name: Legacy parameter for backward compatibility.
        split: Legacy parameter.

    Returns:
        Dataset with 'query' field containing prompts.
    """
    datasets_to_combine = []

    data_cfg = config.get("data", {}) if config else {}
    primary_path = data_cfg.get("ppo_primary", "data/ppo/ultrafeedback_binarized_gen")
    safety_path = data_cfg.get("ppo_safety", "data/ppo/pku_saferlhf")

    # Load UltraFeedback generation prompts (primary)
    primary_dir = Path(primary_path)
    if primary_dir.exists():
        logger.info("Loading UltraFeedback prompts from local: %s", primary_dir)
        ds = load_from_disk(str(primary_dir))
        ds = ds.map(
            extract_ultrafeedback_prompt,
            remove_columns=ds.column_names,
            desc="Extracting UltraFeedback prompts",
        )
        ds = ds.filter(lambda x: len(x["query"]) > 20)
        datasets_to_combine.append(ds)
        logger.info("UltraFeedback prompts: %d loaded", len(ds))
    else:
        logger.warning("UltraFeedback not found at %s", primary_dir)

    # Load PKU-SafeRLHF safety prompts
    safety_dir = Path(safety_path)
    if safety_dir.exists():
        logger.info("Loading PKU-SafeRLHF from local: %s", safety_dir)
        ds = load_from_disk(str(safety_dir))
        ds = ds.map(
            extract_saferlhf_prompt,
            remove_columns=ds.column_names,
            desc="Extracting SafeRLHF prompts",
        )
        ds = ds.filter(lambda x: len(x["query"]) > 20)
        datasets_to_combine.append(ds)
        logger.info("SafeRLHF prompts: %d loaded", len(ds))
    else:
        logger.warning("PKU-SafeRLHF not found at %s", safety_dir)

    if not datasets_to_combine:
        raise FileNotFoundError(
            f"No PPO datasets found at {primary_path} or {safety_path}. "
            "Run the dataset download script first."
        )

    combined = concatenate_datasets(datasets_to_combine)

    # Deduplicate
    df = combined.to_pandas()
    df = df.drop_duplicates(subset=["query"])
    combined = Dataset.from_pandas(df, preserve_index=False)
    combined = combined.shuffle(seed=seed)

    if max_samples is not None:
        combined = combined.select(range(min(max_samples, len(combined))))

    logger.info("PPO prompt dataset: %d unique prompts", len(combined))
    return combined


def collator(data: list[dict[str, Any]]) -> dict[str, list]:
    """Collate function for PPO prompt batches.

    Args:
        data: List of example dicts from PPOPromptDataset.

    Returns:
        Batched dict with 'input_ids' and 'query' lists.
    """
    return {
        "input_ids": [d["input_ids"] for d in data],
        "query": [d["query"] for d in data],
    }
