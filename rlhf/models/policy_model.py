"""Policy model wrapper used for both SFT and PPO training."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


class PolicyModel(nn.Module):
    """Causal language model used as the policy in RLHF.

    Wraps a HuggingFace ``AutoModelForCausalLM`` and exposes helpers used
    during SFT and PPO training.

    Args:
        model_name_or_path: A HuggingFace model identifier or local path.
        device: Target device (``"cpu"``, ``"cuda"``, …).  Defaults to
                ``"cuda"`` when available, otherwise ``"cpu"``.
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        device: str | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path
        ).to(device)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        """Standard causal-LM forward pass.

        Returns the HuggingFace ``CausalLMOutputWithCrossAttentions`` object.
        When ``labels`` is provided the returned object contains a ``loss``
        field that can be directly back-propagated.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[str]:
        """Generate responses for a list of prompt strings.

        Args:
            prompts: List of plain-text prompts.
            max_new_tokens: Maximum number of new tokens to generate per
                            prompt.
            do_sample: Whether to use sampling (True) or greedy decoding.
            temperature: Sampling temperature.
            top_p: Nucleus-sampling *p* value.

        Returns:
            List of decoded response strings (without the original prompt).
        """
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        prompt_len = enc["input_ids"].shape[1]

        output_ids = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the newly generated tokens
        responses = self.tokenizer.batch_decode(
            output_ids[:, prompt_len:],
            skip_special_tokens=True,
        )
        return responses

    def log_probs_of(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities for a batch of sequences.

        Args:
            input_ids: ``(batch, seq_len)`` token-ID tensor.
            attention_mask: ``(batch, seq_len)`` mask tensor.

        Returns:
            ``(batch, seq_len - 1)`` tensor of log-probabilities.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        # Gather the log-prob of the actual next token at each position
        next_token_ids = input_ids[:, 1:].unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, next_token_ids).squeeze(-1)
        return token_log_probs

    def save_pretrained(self, output_dir: str) -> None:
        """Save model and tokenizer weights to *output_dir*."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
