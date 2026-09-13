"""Generate MMLU answers and measure format-sensitive as well as relaxed accuracy."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import read_jsonl, write_jsonl


VALID_LETTERS = {"A", "B", "C", "D"}
LETTER_ONLY_RE = re.compile(r"^[\s\[\(]*([ABCD])[\s\]\)\.,:;!?]*$", re.IGNORECASE)
LABELLED_LETTER_RE = re.compile(
    r"(?i)\b(?:final\s+answer|answer|result|option|choice)\s*(?:is\s*)?[:\-]?\s*[\[\(]?\s*([ABCD])\b"
)
VERBAL_LETTER_RE = re.compile(r"(?i)\b(?:the\s+)?(?:correct\s+)?answer\s+is\s+([ABCD])\b")
STANDALONE_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prepared MMLU data and preserve each raw generation for format analysis."
    )
    parser.add_argument("--eval_file", required=True, help="mmlu.jsonl from prepare_datasets.py")
    parser.add_argument("--model_name", required=True, help="Base model name or local path")
    parser.add_argument("--output_path", required=True, help="Prediction JSONL path")
    parser.add_argument("--adapter_path", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    parser.add_argument("--chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat_template_model", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return value


def resolve_tokenizer_path(args: argparse.Namespace) -> str:
    if args.tokenizer_path:
        return args.tokenizer_path
    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    if adapter_path and (adapter_path / "tokenizer_config.json").exists():
        return str(adapter_path)
    return args.model_name


def load_model_and_tokenizer(args: argparse.Namespace, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_tokenizer_path(args),
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
        if template_tokenizer.chat_template is None:
            raise ValueError(f"{args.chat_template_model} has no chat template")
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
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.to(device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def build_model_prompt(tokenizer, prompt: str, chat_template: bool) -> str:
    if not chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def classify_answer_format(response: str) -> str:
    text = response.strip()
    if not text:
        return "empty"
    if "###" in text:
        return "gsm8k_hash"
    if re.match(r"(?i)^\s*result\s*[:\-]", text):
        return "result_label"
    if re.match(r"(?i)^\s*(?:final\s+answer|answer)\s*(?:is\s*)?[:\-]", text):
        return "answer_label"
    if LETTER_ONLY_RE.fullmatch(text):
        return "letter_only"
    if VERBAL_LETTER_RE.search(text[:160]):
        return "verbal_answer"
    return "other"


def extract_predicted_letter(response: str) -> str:
    text = response.strip()
    direct = LETTER_ONLY_RE.fullmatch(text)
    if direct:
        return direct.group(1).upper()

    for pattern in (LABELLED_LETTER_RE, VERBAL_LETTER_RE):
        match = pattern.search(text[:160])
        if match:
            return match.group(1).upper()

    # This makes relaxed accuracy tolerant to e.g. "I choose C", while strict
    # accuracy below still requires the MMLU instruction's letter-only format.
    match = STANDALONE_LETTER_RE.search(text[:80])
    if match:
        return match.group(1).upper()
    return ""


def metadata_value(record: dict[str, Any], key: str, default: Any = None) -> Any:
    metadata = record.get("metadata")
    return metadata.get(key, default) if isinstance(metadata, dict) else default


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    count = len(records)
    format_counts = Counter(str(record["answer_format"]) for record in records)
    canonical = sum(bool(record["format_is_canonical"]) for record in records)
    relaxed_correct = sum(bool(record["content_correct"]) for record in records)
    strict_correct = sum(bool(record["strict_correct"]) for record in records)
    correct_noncanonical = sum(
        bool(record["content_correct"]) and not bool(record["format_is_canonical"])
        for record in records
    )
    return {
        "run_name": args.run_name or Path(args.output_path).stem,
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "tokenizer_path": resolve_tokenizer_path(args),
        "eval_file": str(Path(args.eval_file).resolve()),
        "num_examples": count,
        "relaxed_accuracy": relaxed_correct / count if count else 0.0,
        "strict_accuracy": strict_correct / count if count else 0.0,
        "canonical_letter_only_ratio": canonical / count if count else 0.0,
        "format_error_ratio": 1.0 - canonical / count if count else 0.0,
        "correct_but_noncanonical_ratio": correct_noncanonical / count if count else 0.0,
        "accuracy_lost_to_format_ratio": (relaxed_correct - strict_correct) / count if count else 0.0,
        "answer_format_counts": dict(sorted(format_counts.items())),
        "chat_template": args.chat_template,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
    }


def generate_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_records = list(read_jsonl(args.eval_file))
    if args.max_samples is not None:
        source_records = source_records[: args.max_samples]
    if not source_records:
        raise ValueError(f"No MMLU records in {args.eval_file}")

    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args, device)
    predictions: list[dict[str, Any]] = []

    for start in range(0, len(source_records), args.batch_size):
        batch = source_records[start : start + args.batch_size]
        prompts = [
            build_model_prompt(tokenizer, str(record["prompt"]), args.chat_template) for record in batch
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

        for record, response in zip(batch, responses):
            target = str(record["target"]).strip().upper()
            prediction = extract_predicted_letter(response)
            answer_format = classify_answer_format(response)
            is_canonical = answer_format == "letter_only"
            content_correct = prediction == target
            predictions.append(
                {
                    "example_id": record["example_id"],
                    "subject": metadata_value(record, "subject", "unknown"),
                    "target": target,
                    "predicted_letter": prediction,
                    "response": response,
                    "answer_format": answer_format,
                    "format_is_canonical": is_canonical,
                    "content_correct": content_correct,
                    "strict_correct": content_correct and is_canonical,
                }
            )
    return predictions


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = generate_records(args)
    write_jsonl(output_path, records)
    summary = summarize(records, args)
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
