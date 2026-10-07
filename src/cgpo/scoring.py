"""Completion-only sequence scoring shared by target and source models."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

import torch


def completion_scores_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return length-normalized log probability and normalized entropy.

    ``labels`` follows the causal-LM convention and uses ``-100`` for prompt
    and padding tokens. Both quantities average over completion tokens only.
    Entropy is divided by ``log(vocab_size)`` and therefore lies in ``[0, 1]``.
    """

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits must be [batch, length, vocab] and labels [batch, length].")
    if logits.shape[-1] <= 1:
        raise ValueError("vocabulary size must exceed one.")

    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    if torch.any(mask.sum(dim=-1) == 0):
        raise ValueError("every example must contain at least one completion token.")

    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    sequence_logps = (token_logps * mask).sum(dim=-1) / mask.sum(dim=-1)

    probs = log_probs.exp()
    token_entropy = -(probs * log_probs).sum(dim=-1) / math.log(logits.shape[-1])
    normalized_entropy = (token_entropy * mask).sum(dim=-1) / mask.sum(dim=-1)
    return sequence_logps, normalized_entropy


def render_prompt(tokenizer: Any, prompt: str) -> str:
    """Render a user prompt with the model's own chat template when available."""

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def make_mismatched_anchor_completions(
    prompt_ids: Sequence[Any],
    completions: Sequence[str],
    *,
    seed: int,
) -> list[str]:
    """Select deterministic anchor completions from different prompts.

    The anchor estimates the shared policy-reference drift. Reusing a
    completion with its original prompt would turn that row into a matched
    likelihood rather than the intended mismatched statistic.
    """

    if len(prompt_ids) != len(completions) or not prompt_ids:
        raise ValueError("prompt_ids and completions must be non-empty and aligned.")
    if len(set(prompt_ids)) < 2:
        raise ValueError("anchor construction requires at least two distinct prompts.")

    order = list(range(len(prompt_ids)))
    random.Random(seed).shuffle(order)
    position = {row_index: offset for offset, row_index in enumerate(order)}
    anchors: list[str] = []
    for row_index, prompt_id in enumerate(prompt_ids):
        start = position[row_index]
        for offset in range(1, len(order) + 1):
            candidate = order[(start + offset) % len(order)]
            if prompt_ids[candidate] != prompt_id:
                anchors.append(str(completions[candidate]))
                break
        else:  # Guarded by the distinct-prompt check above.
            raise RuntimeError("failed to construct a mismatched anchor.")
    return anchors


def tokenize_prompt_completion_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    max_length: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Tokenize raw texts and mask every non-completion token.

    Prompt and completion strings remain shared across models, while each model
    applies its own tokenizer. Long prompts are truncated from the left before
    completion tokens are removed.
    """

    if len(prompts) != len(completions) or not prompts:
        raise ValueError("prompts and completions must be non-empty and aligned.")
    if max_length <= 1:
        raise ValueError("max_length must exceed one token.")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer must define a pad or EOS token.")

    encoded_rows: list[tuple[list[int], list[int]]] = []
    for prompt, completion in zip(prompts, completions):
        rendered = render_prompt(tokenizer, prompt)
        suffix = completion
        if tokenizer.eos_token and not suffix.endswith(tokenizer.eos_token):
            suffix += tokenizer.eos_token

        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        full_ids = tokenizer.encode(rendered + suffix, add_special_tokens=False)
        split = len(prompt_ids)
        if full_ids[:split] != prompt_ids:
            # A token may cross the string boundary. Assign that boundary token
            # to the completion, following preference-training tokenizers.
            split = max(0, split - 1)
        prompt_ids = full_ids[:split]
        completion_ids = full_ids[split:]
        if not completion_ids:
            raise ValueError("a completion produced no tokens.")

        completion_ids = completion_ids[: max_length - 1]
        prompt_budget = max_length - len(completion_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        if not prompt_ids:
            prefix_id = tokenizer.bos_token_id
            if prefix_id is None:
                prefix_id = tokenizer.eos_token_id
            if prefix_id is None:
                raise ValueError("tokenizer must provide context before completion tokens.")
            prompt_ids = [prefix_id]
            completion_ids = completion_ids[: max_length - 1]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        encoded_rows.append((input_ids, labels))

    width = max(len(row[0]) for row in encoded_rows)
    input_ids = torch.full((len(encoded_rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded_rows), width), dtype=torch.long)
    labels = torch.full((len(encoded_rows), width), -100, dtype=torch.long)
    for row_index, (row_ids, row_labels) in enumerate(encoded_rows):
        length = len(row_ids)
        input_ids[row_index, :length] = torch.tensor(row_ids)
        attention_mask[row_index, :length] = 1
        labels[row_index, :length] = torch.tensor(row_labels)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }
