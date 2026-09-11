from __future__ import annotations

import argparse
import gc
import json
import math
import re
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .common import read_jsonl, write_json, write_jsonl
from .prompts import MMLU_CHOICES


CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
HASH_CHOICE_RE = re.compile(r"####\s*([ABCD])\b", re.IGNORECASE)


def extract_choice(prediction: str) -> str:
    hash_matches = HASH_CHOICE_RE.findall(prediction)
    if hash_matches:
        return hash_matches[-1].upper()
    matches = CHOICE_RE.findall(prediction)
    return matches[-1].upper() if matches else ""


def encode_empty_assistant_prompt(tokenizer: object, template: object, prompt: str) -> torch.Tensor:
    messages = template.mm_plugin.process_messages(
        [{"role": "user", "content": prompt}], [], [], [], None
    )
    prompt_ids, _ = template.encode_oneturn(
        tokenizer, messages + [{"role": "assistant", "content": ""}], None, None
    )
    if not prompt_ids:
        raise ValueError("The selected chat template produced an empty prompt.")
    return torch.tensor(prompt_ids, dtype=torch.long)


def resolve_choice_token_ids(tokenizer: object, template: object, prompt: str) -> dict[str, dict[str, Any]]:
    messages = template.mm_plugin.process_messages(
        [{"role": "user", "content": prompt}], [], [], [], None
    )
    _, empty_response_ids = template.encode_oneturn(
        tokenizer, messages + [{"role": "assistant", "content": ""}], None, None
    )
    resolved: dict[str, dict[str, Any]] = {}
    for letter in MMLU_CHOICES:
        attempts = []
        chosen = None
        for text in (letter, f" {letter}"):
            _, response_ids = template.encode_oneturn(
                tokenizer, messages + [{"role": "assistant", "content": text}], None, None
            )
            if empty_response_ids and response_ids[-len(empty_response_ids) :] == empty_response_ids:
                content_ids = response_ids[: -len(empty_response_ids)]
            else:
                content_ids = tokenizer.encode(text, add_special_tokens=False)
            attempt = {
                "text": text,
                "ids": [int(token_id) for token_id in content_ids],
                "decoded": tokenizer.decode(content_ids, skip_special_tokens=False),
            }
            attempts.append(attempt)
            if chosen is None and len(content_ids) == 1:
                chosen = attempt
        if chosen is None:
            raise ValueError(f"No one-token continuation found for choice {letter}: {attempts}")
        resolved[letter] = {"token_id": chosen["ids"][0], "text": chosen["text"], "attempts": attempts}
    return resolved


@torch.inference_mode()
def choice_statistics(chat_model: object, prompt: str, target: str, cutoff_len: int) -> dict[str, Any]:
    engine = chat_model.engine
    model = engine.model
    tokenizer = engine.tokenizer
    template = engine.template
    input_ids = encode_empty_assistant_prompt(tokenizer, template, prompt)
    if input_ids.numel() > cutoff_len:
        input_ids = input_ids[-cutoff_len:]
    batch = input_ids.unsqueeze(0).to(next(model.parameters()).device)
    logits = model(input_ids=batch, attention_mask=torch.ones_like(batch)).logits[0, -1].float()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    token_info = resolve_choice_token_ids(tokenizer, template, prompt)
    choice_probs = {letter: float(probs[token_info[letter]["token_id"]].item()) for letter in MMLU_CHOICES}
    choice_mass = sum(choice_probs.values())
    argmax = max(MMLU_CHOICES, key=choice_probs.get)
    normalized = {letter: choice_probs[letter] / choice_mass for letter in MMLU_CHOICES}
    entropy = -sum(value * math.log(value) for value in normalized.values() if value > 0)
    result: dict[str, Any] = {
        "choice_mass": choice_mass,
        "choice_argmax": argmax,
        "choice_argmax_correct": int(argmax == target),
        "choice_entropy_normalized": entropy,
    }
    for letter in MMLU_CHOICES:
        token_id = token_info[letter]["token_id"]
        result[f"choice_probability_{letter}"] = choice_probs[letter]
        result[f"choice_log_probability_{letter}"] = float(log_probs[token_id].item())
        result[f"choice_token_id_{letter}"] = token_id
        result[f"choice_token_text_{letter}"] = token_info[letter]["text"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generative and first-token MMLU evaluation.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_kind", choices=("base", "sft", "eaft"), required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    from llamafactory.chat import ChatModel

    chat_model = ChatModel(
        {
            "model_name_or_path": args.model_name_or_path,
            "template": args.template,
            "infer_backend": "huggingface",
            "trust_remote_code": True,
            "finetuning_type": "full",
            "stage": "sft",
            "cache_dir": args.cache_dir,
            "cutoff_len": args.cutoff_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "resize_vocab": True,
        }
    )

    records = read_jsonl(args.input_file)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    output_rows = []
    for record in tqdm(records, desc=f"Evaluating {args.model_kind}"):
        responses = chat_model.chat(
            [{"role": "user", "content": record["prompt"]}],
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        prediction = responses[0].response_text if responses else ""
        parsed = extract_choice(prediction)
        target = str(record["target"]).strip().upper()
        output_rows.append(
            {
                **record,
                "model_kind": args.model_kind,
                "prediction": prediction,
                "parsed_prediction": parsed,
                "generation_correct": int(parsed == target),
                **choice_statistics(chat_model, record["prompt"], target, args.cutoff_len),
            }
        )

    write_jsonl(args.output_file, output_rows)
    write_json(
        str(args.output_file) + ".summary.json",
        {
            "model_kind": args.model_kind,
            "model_name_or_path": args.model_name_or_path,
            "template": args.template,
            "num_examples": len(output_rows),
            "decoding": "greedy",
        },
    )
    del chat_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({"output_file": args.output_file, "num_examples": len(output_rows)}, indent=2))


if __name__ == "__main__":
    main()
