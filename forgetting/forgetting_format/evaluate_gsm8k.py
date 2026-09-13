"""Evaluate GSM8K answer accuracy and the ``####`` answer-format retention.

The evaluator deliberately uses the same two raw prompt variants as GSM8K SFT:
``Question ... Answer:`` and ``Problem ... result:``.  It reports both a
strict score (the generated answer must contain ``#### <correct number>``) and
a relaxed numeric score (the final generated number is correct even when the
marker was forgotten).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import write_jsonl
from prepare_datasets import GSM8K_PROMPT_FORMATS, _format_gsm8k_prompt, _load_gsm8k


HASH_NUMBER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K content accuracy and #### answer-format accuracy."
    )
    parser.add_argument("--model_name", required=True, help="Model name or local checkpoint path")
    parser.add_argument("--tokenizer_path", default=None, help="Optional original tokenizer name or path")
    parser.add_argument("--output_path", required=True, help="Prediction JSONL path")
    parser.add_argument(
        "--format_variant", required=True, choices=sorted(GSM8K_PROMPT_FORMATS),
        help="Prompt template used at evaluation time",
    )
    parser.add_argument("--data_path", default="openai/gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    parser.add_argument("--chat_template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chat_template_model", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return value


def load_model_and_tokenizer(args: argparse.Namespace, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if args.chat_template and tokenizer.chat_template is None:
        if not args.chat_template_model:
            raise ValueError(
                "Tokenizer has no chat template. Pass --chat_template_model or --no-chat_template."
            )
        template_tokenizer = AutoTokenizer.from_pretrained(
            args.chat_template_model,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer.chat_template = template_tokenizer.chat_template
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.bf16 and device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def build_model_prompt(tokenizer, prompt: str, chat_template: bool) -> str:
    if not chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def normalize_number(value: str) -> str:
    """Canonicalize integer/decimal strings so e.g. ``1,200`` equals ``1200``."""
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return ""
    if not number.is_finite():
        return ""
    if number == 0:
        return "0"
    return format(number.normalize(), "f").rstrip("0").rstrip(".") if "." in format(number.normalize(), "f") else format(number.normalize(), "f")


def extract_answer(text: str, require_hash: bool) -> str:
    matches = HASH_NUMBER_RE.findall(text) if require_hash else NUMBER_RE.findall(text)
    return normalize_number(matches[-1]) if matches else ""


def generate_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    dataset, source = _load_gsm8k(
        args.data_path, args.dataset_config, args.split, args.cache_dir
    )
    rows = list(dataset)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No GSM8K examples to evaluate")
    if any("question" not in row or "answer" not in row for row in rows):
        raise KeyError("GSM8K records must contain question and answer fields")

    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args, device)
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [
            build_model_prompt(
                tokenizer, _format_gsm8k_prompt(str(row["question"]), args.format_variant), args.chat_template
            )
            for row in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = generated[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        for offset, (row, prompt, response) in enumerate(zip(batch, prompts, responses)):
            target = extract_answer(str(row["answer"]), require_hash=True)
            strict_prediction = extract_answer(response, require_hash=True)
            relaxed_prediction = strict_prediction or extract_answer(response, require_hash=False)
            predictions.append(
                {
                    "source_row": start + offset,
                    "question": str(row["question"]),
                    "prompt": prompt,
                    "target": target,
                    "response": response,
                    "has_hash_marker": bool(HASH_NUMBER_RE.search(response)),
                    "strict_prediction": strict_prediction,
                    "relaxed_prediction": relaxed_prediction,
                    "strict_correct": bool(target) and strict_prediction == target,
                    "content_correct": bool(target) and relaxed_prediction == target,
                }
            )
    return predictions, source


def summarize(records: list[dict[str, Any]], source: str, args: argparse.Namespace) -> dict[str, Any]:
    count = len(records)
    strict_correct = sum(bool(record["strict_correct"]) for record in records)
    content_correct = sum(bool(record["content_correct"]) for record in records)
    hash_marker = sum(bool(record["has_hash_marker"]) for record in records)
    unparsed = sum(not bool(record["relaxed_prediction"]) for record in records)
    return {
        "run_name": args.run_name or Path(args.output_path).stem,
        "model_name": args.model_name,
        "dataset_source": source,
        "split": args.split,
        "format_variant": args.format_variant,
        "num_examples": count,
        "strict_hash_accuracy": strict_correct / count if count else 0.0,
        "relaxed_numeric_accuracy": content_correct / count if count else 0.0,
        "hash_marker_ratio": hash_marker / count if count else 0.0,
        "accuracy_lost_to_hash_format_ratio": (content_correct - strict_correct) / count if count else 0.0,
        "unparsed_response_ratio": unparsed / count if count else 0.0,
        "chat_template": args.chat_template,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records, source = generate_records(args)
    write_jsonl(output_path, records)
    summary = summarize(records, source, args)
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
