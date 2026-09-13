#!/usr/bin/env python
"""Build the no-memory GSM8K baseline used to define self-wrong subsets."""
import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


NUMBER_RE = re.compile(
    r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the no-memory greedy GSM8K baseline used to define self-wrong subsets."
    )
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3 thinking mode. Disabled by default for comparability with non-reasoning baselines.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_answer(text: str) -> str:
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    matches = NUMBER_RE.findall(text)
    return matches[-1].strip() if matches else text.strip()


def normalize_number(text: str):
    value = text.strip().replace(",", "").replace("$", "").replace("%", "")
    value = value.strip().rstrip(".")
    try:
        if "/" in value:
            left, right = value.split("/", 1)
            return Fraction(Decimal(left.strip())) / Fraction(Decimal(right.strip()))
        return Decimal(value)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return value.lower()


def answers_match(prediction: str, gold: str) -> bool:
    return normalize_number(prediction) == normalize_number(gold)


def build_prompt(tokenizer, question: str, enable_thinking: bool) -> str:
    user_content = (
        "Solve the GSM8K math problem. Show concise reasoning and finish with exactly one final line "
        "in the format #### <number>.\n\n"
        f"Question:\n{question}"
    )
    messages = [
        {
            "role": "system",
            "content": "You are a careful math problem solver. Always end with #### followed by the final numeric answer.",
        },
        {"role": "user", "content": user_content},
    ]
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if "qwen3" in tokenizer.name_or_path.lower():
        template_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def write_result(
    path: Path,
    args: argparse.Namespace,
    data: Dict[str, object],
    records: List[Dict[str, object]],
) -> None:
    correct = sum(int(record["correct"]) for record in records)
    wrong_indices = [int(record["index"]) for record in records if not record["correct"]]
    result = {
        "summary": {
            "method": "no-memory greedy GSM8K answer generation",
            "model_name": args.model_name,
            "data_path": str(args.data_path.resolve()),
            "source_path": data.get("source_path"),
            "split": data.get("split", "train"),
            "num_queries": len(records),
            "correct": correct,
            "accuracy": correct / len(records) if records else 0.0,
            "wrong": len(wrong_indices),
            "wrong_indices": wrong_indices,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "chat_template": True,
            "enable_thinking": args.enable_thinking,
            "decoding": "greedy",
        },
        "per_query": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_path}")

    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    data = json.loads(args.data_path.read_text(encoding="utf-8"))
    rows = list(data["records"])
    records: List[Dict[str, object]] = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="Evaluating GSM8K"):
        batch = rows[start : start + args.batch_size]
        questions = [str(row.get("translations", {}).get("en", row["source"])["question"]) for row in batch]
        answers = [str(row.get("translations", {}).get("en", row["source"])["answer"]) for row in batch]
        prompts = [build_prompt(tokenizer, question, args.enable_thinking) for question in questions]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        ).to(device)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        responses = tokenizer.batch_decode(generated[:, input_length:], skip_special_tokens=True)

        for row, question, answer, prompt, response in zip(
            batch, questions, answers, prompts, responses
        ):
            gold = extract_answer(answer)
            prediction = extract_answer(response)
            records.append(
                {
                    "index": int(row["index"]),
                    "question": question,
                    "gold_answer": answer,
                    "gold_final": gold,
                    "prompt": prompt,
                    "response": response,
                    "pred_final": prediction,
                    "correct": answers_match(prediction, gold),
                }
            )
        write_result(args.output_path, args, data, records)

    print(json.dumps(json.loads(args.output_path.read_text(encoding="utf-8"))["summary"], indent=2))


if __name__ == "__main__":
    main()
