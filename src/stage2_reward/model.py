"""Stage 2 Reward Model -- Model definition with scalar reward head.

Loads the SFT checkpoint (merges LoRA adapters), adds a scalar reward head
on top of the last hidden state, and freezes all layers except the reward
head and the last N transformer blocks.

Hardware target: 4x Tesla K80 (fp32, no quantization).
Compatible with: transformers==4.38.2, peft==0.7.1
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


def load_and_merge_sft_model(
    base_model_name: str,
    adapter_path: str,
    device_map: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model and merge the SFT LoRA adapter weights.

    Args:
        base_model_name: HuggingFace model identifier.
        adapter_path: Path to saved LoRA adapter directory.
        device_map: Device mapping strategy.

    Returns:
        Tuple of (merged_model, tokenizer).
    """
    logger.info("Loading base model in fp32: %s", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging LoRA adapters into base model...")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("SFT model merged successfully")
    return model, tokenizer


def load_reward_model(
    base_model_name: str,
    adapter_path: str,
    freeze_layers_except_last_n: int = 2,
    device_map: str | None = None,
    load_in_4bit: bool = False,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load a reward model based on the merged SFT checkpoint (fp32).

    Uses AutoModelForSequenceClassification with num_labels=1 for scalar
    reward prediction. No quantization for K80 compatibility.

    Args:
        base_model_name: HuggingFace model identifier.
        adapter_path: Path to the SFT LoRA adapter directory.
        freeze_layers_except_last_n: Number of final transformer blocks
            to keep trainable.
        device_map: Device mapping strategy.
        load_in_4bit: Ignored (always fp32 for K80).

    Returns:
        Tuple of (reward_model, tokenizer).
    """
    logger.info("Building reward model from SFT checkpoint (fp32)")

    # Load as sequence classification model with 1 output (scalar reward)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )

    # Load tokenizer from adapter path
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.config.pad_token_id = tokenizer.pad_token_id

    # Freeze layers
    _freeze_model_layers(model, freeze_layers_except_last_n)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Reward model: %d total params, %d trainable (%.2f%%)",
        total_params,
        trainable_params,
        100 * trainable_params / total_params,
    )

    return model, tokenizer


def _freeze_model_layers(
    model: nn.Module,
    keep_last_n: int = 2,
) -> None:
    """Freeze all transformer layers except the last N and the classification head.

    Args:
        model: The transformer model.
        keep_last_n: Number of final transformer blocks to keep trainable.
    """
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classification/score head
    for name, param in model.named_parameters():
        if "score" in name or "classifier" in name or "reward" in name:
            param.requires_grad = True
            logger.debug("Unfreezing: %s", name)

    # Unfreeze last N transformer blocks
    layer_patterns = [
        "model.layers",
        "transformer.h",
        "model.decoder.layers",
    ]

    for pattern in layer_patterns:
        layers = []
        for name, param in model.named_parameters():
            if pattern in name:
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == pattern.split(".")[-1]:
                        if i + 1 < len(parts) and parts[i + 1].isdigit():
                            layer_idx = int(parts[i + 1])
                            layers.append((layer_idx, name, param))
                            break

        if layers:
            max_layer = max(idx for idx, _, _ in layers)
            threshold = max_layer - keep_last_n + 1
            for layer_idx, name, param in layers:
                if layer_idx >= threshold:
                    param.requires_grad = True
                    logger.debug("Unfreezing layer %d: %s", layer_idx, name)
            break

    # Unfreeze final layer norm
    for name, param in model.named_parameters():
        if "norm" in name.lower() and ("final" in name.lower() or "ln_f" in name.lower()):
            param.requires_grad = True
            logger.debug("Unfreezing final norm: %s", name)
