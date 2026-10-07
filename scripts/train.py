#!/usr/bin/env python
"""Compact training entry point for the CGPO objective.

This script intentionally keeps orchestration small. It exposes the paper's
training logic while leaving cluster-specific launch and distributed settings
to the user.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgpo.io import read_jsonl, read_yaml
from cgpo.objective import CGPOConfig, ReferenceAssignment, cgpo_loss
from cgpo.scoring import completion_scores_from_logits, tokenize_prompt_completion_batch


class Rows(Dataset):
    def __init__(self, values: list[dict]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict:
        return self.values[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "main.yaml"))
    parser.add_argument(
        "--override-config",
        action="append",
        default=[],
        help="Partial YAML overlay; may be supplied more than once.",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def validate_rows(
    rows: list[dict],
    *,
    requires_consensus: bool,
    allocation_weight_field: str | None,
) -> None:
    required = {
        "prompt",
        "completion",
        "label",
        "initial_logp",
        "anchor_completion",
        "anchor_initial_logp",
    }
    if requires_consensus:
        required.add("aligned_consensus")
    if allocation_weight_field:
        required.add(allocation_weight_field)
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index} is missing {sorted(missing)}")
        if row["label"] not in {"D", "U"}:
            raise ValueError(f"row {index} label must be D or U")
        if allocation_weight_field:
            weight = float(row[allocation_weight_field])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"row {index} {allocation_weight_field} must be finite and non-negative"
                )


def class_distribution(
    rows: list[dict],
    label: str,
    allocation_weight_field: str | None,
) -> tuple[list[dict], torch.Tensor]:
    """Return one label partition and its normalized empirical distribution."""

    selected = [row for row in rows if row["label"] == label]
    if not selected:
        raise ValueError(f"training data must contain at least one {label} row")
    if allocation_weight_field is None:
        weights = torch.ones(len(selected), dtype=torch.double)
    else:
        weights = torch.tensor(
            [float(row[allocation_weight_field]) for row in selected],
            dtype=torch.double,
        )
    total = weights.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError(f"{label} allocation weights must have a positive finite sum")
    return selected, weights / total


def class_loader(
    rows: list[dict],
    label: str,
    allocation_weight_field: str | None,
    *,
    batch_size: int,
    seed: int,
) -> DataLoader:
    selected, probabilities = class_distribution(
        rows,
        label,
        allocation_weight_field,
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        probabilities,
        num_samples=len(selected),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        Rows(selected),
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda batch: batch,
    )


def next_batch(iterator, loader: DataLoader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def disable_dropout(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.Dropout):
            child.p = 0.0


def score_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    *,
    max_length: int,
    device: str,
) -> torch.Tensor:
    encoded = tokenize_prompt_completion_batch(
        tokenizer,
        prompts,
        completions,
        max_length=max_length,
        device=device,
    )
    logits = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        use_cache=False,
    ).logits
    scores, _ = completion_scores_from_logits(logits, encoded["labels"])
    return scores


def main() -> None:
    args = parse_args()
    raw_config = read_yaml(args.config)
    for override_path in args.override_config:
        deep_update(raw_config, read_yaml(override_path))
    method = raw_config["method"]
    training = raw_config["training"]
    assignment = ReferenceAssignment.from_name(method["reference_assignment"])
    allocation_weight_field = method.get("source_allocation_weight_field")
    if allocation_weight_field is not None and not isinstance(allocation_weight_field, str):
        raise TypeError("source_allocation_weight_field must be a string or null")
    model_name = args.model or raw_config["models"].get("initial_policy")
    if not model_name:
        raise ValueError(
            "Supply the SFT initial-policy checkpoint with --model or "
            "models.initial_policy in the config."
        )
    set_seed(int(training.get("seed", 42)))

    rows = read_jsonl(args.data)
    validate_rows(
        rows,
        requires_consensus=assignment is not ReferenceAssignment.POLICY_ANCHORED,
        allocation_weight_field=allocation_weight_field,
    )
    branch_batch_size = int(training.get("batch_size", 1))
    if branch_batch_size <= 0:
        raise ValueError("training batch_size must be positive")
    seed = int(training.get("seed", 42))
    desirable_loader = class_loader(
        rows,
        "D",
        allocation_weight_field,
        batch_size=branch_batch_size,
        seed=seed,
    )
    undesirable_loader = class_loader(
        rows,
        "U",
        allocation_weight_field,
        batch_size=branch_batch_size,
        seed=seed + 1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_name = str(training.get("dtype", "bfloat16"))
    dtype = torch.float32 if args.device == "cpu" else getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=False,
    ).to(args.device)
    disable_dropout(model)

    lora = training.get("lora", {})
    if int(lora.get("rank", 0)) > 0:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(lora["rank"]),
                lora_alpha=int(lora.get("alpha", 2 * int(lora["rank"]))),
                lora_dropout=float(lora.get("dropout", 0.0)),
                target_modules=lora.get("target_modules", "all-linear"),
            ),
        )

    config = CGPOConfig(
        beta=float(method["beta"]),
        desirable_weight=float(method["desirable_weight"]),
        undesirable_weight=float(method["undesirable_weight"]),
        undesirable_warmup_start=int(method.get("undesirable_warmup_start", 0)),
        undesirable_warmup_steps=int(method.get("undesirable_warmup_steps", 0)),
        reference_assignment=assignment,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
    )
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    epochs = int(training.get("epochs", 1))
    max_length = int(training.get("max_length", 2048))
    steps_per_epoch = int(
        training.get(
            "steps_per_epoch",
            max(len(desirable_loader), len(undesirable_loader)),
        )
    )
    if steps_per_epoch <= 0:
        raise ValueError("training steps_per_epoch must be positive")
    total_updates = math.ceil(steps_per_epoch * epochs / accumulation)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    update_step = 0
    for _ in range(epochs):
        desirable_iterator = iter(desirable_loader)
        undesirable_iterator = iter(undesirable_loader)
        for _ in range(steps_per_epoch):
            desirable_batch, desirable_iterator = next_batch(
                desirable_iterator,
                desirable_loader,
            )
            undesirable_batch, undesirable_iterator = next_batch(
                undesirable_iterator,
                undesirable_loader,
            )
            batch = desirable_batch + undesirable_batch
            prompts = [str(row["prompt"]) for row in batch]
            completions = [str(row["completion"]) for row in batch]
            policy_scores = score_batch(
                model,
                tokenizer,
                prompts,
                completions,
                max_length=max_length,
                device=args.device,
            )
            initial_scores = torch.tensor(
                [row["initial_logp"] for row in batch],
                device=args.device,
                dtype=policy_scores.dtype,
            )
            if assignment is ReferenceAssignment.POLICY_ANCHORED:
                consensus_scores = initial_scores
            else:
                consensus_scores = torch.tensor(
                    [row["aligned_consensus"] for row in batch],
                    device=args.device,
                    dtype=policy_scores.dtype,
                )
            labels = torch.tensor(
                [row["label"] == "D" for row in batch],
                device=args.device,
                dtype=torch.bool,
            )

            # The anchor is policy based for every reference assignment. Its
            # statistic is detached before it enters the pointwise objective.
            with torch.no_grad():
                anchor_policy_scores = score_batch(
                    model,
                    tokenizer,
                    prompts,
                    [str(row["anchor_completion"]) for row in batch],
                    max_length=max_length,
                    device=args.device,
                )
                anchor_initial_scores = torch.tensor(
                    [row["anchor_initial_logp"] for row in batch],
                    device=args.device,
                    dtype=policy_scores.dtype,
                )
                z0 = (anchor_policy_scores - anchor_initial_scores).mean().detach()

            output = cgpo_loss(
                policy_scores,
                initial_scores,
                consensus_scores,
                labels,
                z0,
                step=update_step,
                config=config,
            )
            (output.loss / accumulation).backward()
            micro_step += 1
            if micro_step % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                print(
                    f"step={update_step}/{total_updates} "
                    f"loss={output.loss.item():.6f} "
                    f"u_coeff={output.undesirable_coefficient:.3f}"
                )

    if micro_step % accumulation:
        optimizer.step()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)


if __name__ == "__main__":
    main()
