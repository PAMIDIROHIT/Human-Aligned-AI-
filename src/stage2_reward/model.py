"""Stage 2 Reward Model — Model definition with scalar reward head.

Loads the SFT checkpoint (merges LoRA adapters), adds a scalar reward head
on top of the last hidden state, and freezes all layers except the reward
head and the last N transformer blocks.
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
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def load_and_merge_sft_model(
    base_model_name: str,
    adapter_path: str,
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model and merge the SFT LoRA adapter weights.

    Args:
        base_model_name: HuggingFace model identifier for the base model.
        adapter_path: Path to the saved LoRA adapter directory.
        device_map: Device mapping strategy (default: 'auto').

    Returns:
        A tuple of (merged_model, tokenizer) where the model has LoRA
        weights merged into the base weights.
    """
    logger.info("Loading base model: %s", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
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
    device_map: str = "auto",
    load_in_4bit: bool = True,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load a reward model based on the merged SFT checkpoint.

    Uses AutoModelForSequenceClassification with num_labels=1 for scalar
    reward prediction. Freezes all layers except the classification head
    and the last N transformer blocks.

    Args:
        base_model_name: HuggingFace model identifier.
        adapter_path: Path to the SFT LoRA adapter directory.
        freeze_layers_except_last_n: Number of final transformer blocks
            to keep trainable (in addition to the reward head).
        device_map: Device mapping strategy.
        load_in_4bit: Whether to load in 4-bit quantization.

    Returns:
        A tuple of (reward_model, tokenizer).
    """
    logger.info("Building reward model from SFT checkpoint")

    # Configure quantization
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    # Load as sequence classification model with 1 output (scalar reward)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Load tokenizer from adapter path (has any special tokens added during SFT)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set pad token ID in model config
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
        model: The transformer model with a classification head.
        keep_last_n: Number of final transformer blocks to keep trainable.
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classification/score head
    for name, param in model.named_parameters():
        if "score" in name or "classifier" in name or "reward" in name:
            param.requires_grad = True
            logger.debug("Unfreezing: %s", name)

    # Unfreeze last N transformer blocks
    # Different model architectures use different naming conventions
    layer_patterns = [
        "model.layers",        # Llama, Mistral
        "transformer.h",       # GPT-2
        "model.decoder.layers",  # OPT
    ]

    for pattern in layer_patterns:
        layers = []
        for name, param in model.named_parameters():
            if pattern in name:
                # Extract layer index
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

    # Also unfreeze the final layer norm
    for name, param in model.named_parameters():
        if "norm" in name.lower() and ("final" in name.lower() or "ln_f" in name.lower()):
            param.requires_grad = True
            logger.debug("Unfreezing final norm: %s", name)
