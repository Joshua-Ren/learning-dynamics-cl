from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed

from plasticity_loss_sft.data import (
    filter_prompt_completion_for_context,
    load_instruction_dataset,
    to_prompt_completion_dataset,
)
from plasticity_loss_sft.modeling import (
    get_lm_head_weight,
    inspect_model_access,
    load_model_and_tokenizer,
    supports_assistant_token_mask,
)
from plasticity_loss_sft.runtime import bf16_supported, gpu_report, package_versions, reset_peak_memory


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SFT setup without changing the objective.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_name", default="trl-lib/Capybara")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_limit", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no_bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_dataset = load_instruction_dataset(args.dataset_name, args.split, args.dataset_limit, args.seed)
    prompt_completion = to_prompt_completion_dataset(raw_dataset, tokenizer)
    prompt_completion = filter_prompt_completion_for_context(
        prompt_completion,
        tokenizer,
        args.max_seq_length,
    )
    sample = prompt_completion[0]

    full_text = sample["prompt"] + sample["completion"]
    prompt_ids = tokenizer(sample["prompt"], add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_seq_length,
        return_tensors="pt",
    )
    prompt_token_count = min(len(prompt_ids), encoded["input_ids"].shape[1])
    labels = encoded["input_ids"].clone()
    labels[:, :prompt_token_count] = IGNORE_INDEX

    prompt_mask_ok = bool(torch.all(labels[:, :prompt_token_count] == IGNORE_INDEX).item())
    assistant_mask_ok = bool(torch.any(labels[:, prompt_token_count:] != IGNORE_INDEX).item())
    unmasked_token_count = int(torch.sum(labels != IGNORE_INDEX).item())

    use_bf16 = bf16_supported() and not args.no_bf16
    model, _ = load_model_and_tokenizer(args.model_name, use_bf16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    reset_peak_memory()

    batch = {key: value.to(device) for key, value in encoded.items()}
    batch["labels"] = labels.to(device)
    outputs = model(**batch, output_hidden_states=True, use_cache=False)
    loss = outputs.loss
    loss.backward()

    access_report = inspect_model_access(model)
    lm_head_grad = get_lm_head_weight(model).grad
    hidden_states = outputs.hidden_states
    result = {
        "package_versions": package_versions(),
        "model_name": args.model_name,
        "chat_template_present": bool(tokenizer.chat_template),
        "assistant_token_mask_supported": supports_assistant_token_mask(tokenizer),
        "loss_masking_mode": "native_prompt_completion_loss",
        "prompt_preview": sample["prompt"][:500],
        "completion_preview": sample["completion"][:300],
        "prompt_token_count": prompt_token_count,
        "sequence_token_count": int(encoded["input_ids"].shape[1]),
        "unmasked_assistant_token_count": unmasked_token_count,
        "prompt_labels_all_ignored": prompt_mask_ok,
        "assistant_labels_present": assistant_mask_ok,
        "manual_loss": float(loss.detach().cpu()),
        "hidden_state_count": len(hidden_states) if hidden_states is not None else 0,
        "first_hidden_state_shape": list(hidden_states[0].shape) if hidden_states else None,
        "last_hidden_state_shape": list(hidden_states[-1].shape) if hidden_states else None,
        "lm_head_grad_present": lm_head_grad is not None,
        "lm_head_grad_norm": float(lm_head_grad.detach().float().norm().cpu())
        if lm_head_grad is not None
        else None,
        "model_access": access_report.__dict__,
        "gpu": gpu_report().__dict__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
