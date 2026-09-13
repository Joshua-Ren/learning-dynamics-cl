#!/usr/bin/env python
"""Run retrieval and downstream GSM8K generation in one pass."""

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from forvalue_streaming_ghrh import (
    compute_train_representations,
    get_lm_head_weight,
    normalize_readout_channels,
    parse_gh_embedding_layers,
    resolve_gh_embedding_layers,
    score_test_streaming,
)
from utils import GRPO_dataset


LANGUAGES = ("zh", "fr", "ko", "es")
LABELS = {
    "en": {"question": "Question", "answer": "Answer"},
    "zh": {"question": "问题", "answer": "答案"},
    "fr": {"question": "Question", "answer": "Réponse"},
    "ko": {"question": "문제", "answer": "정답"},
    "es": {"question": "Pregunta", "answer": "Respuesta"},
}

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Agent-memory GSM8K pipeline: retrieve top-1 translated memory with "
            "ForValue RH/GH/Both, then answer English questions and score accuracy."
        )
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Translated self-wrong GSM8K dataset (see attribution/agent_memory/data).",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--channels", nargs="+", default=["rh", "gh", "both"], choices=("rh", "gh", "both"))
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=768, help="Max tokens for retrieval examples.")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Max tokens for agent prompts.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4, help="Retrieval representation batch size.")
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--prediction_topk", type=int, default=16)
    parser.add_argument("--train_score_chunk", type=int, default=8)
    parser.add_argument("--embed_device", type=str, default="auto")
    parser.add_argument("--score_device", type=str, default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3 thinking mode. Disabled by default to match the filtering baseline.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retrieval_query_field",
        choices=("qa", "question"),
        default="qa",
        help=(
            "Text used for the English query when computing retrieval/value scores. "
            "`qa` keeps the previous question+answer behavior; `question` uses only "
            "the English question while translated memory candidates still use question+answer."
        ),
    )
    parser.add_argument(
        "--gh_embedding_layers",
        nargs="+",
        default=["all"],
        help="GH hidden layers. Default all resolves according to --gh_layer_index_mode.",
    )
    parser.add_argument("--gh_layer_index_mode", choices=("bottom", "top"), default="bottom")
    parser.add_argument(
        "--gh_use_input_layernorm",
        "--gh_input_layernorm",
        dest="gh_use_input_layernorm",
        action="store_true",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def format_gsm8k(fields: Dict[str, str], language: str = "en") -> str:
    labels = LABELS.get(language, LABELS["en"])
    return f"{labels['question']}: {fields['question']}\n{labels['answer']}: {fields['answer']}"


def format_gsm8k_question(fields: Dict[str, str], language: str = "en") -> str:
    labels = LABELS.get(language, LABELS["en"])
    return f"{labels['question']}: {fields['question']}"


def extract_gold_answer(answer: str) -> str:
    if "####" in answer:
        answer = answer.rsplit("####", 1)[1]
    matches = NUMBER_RE.findall(answer)
    return matches[-1].strip() if matches else answer.strip()


def extract_prediction_answer(response: str) -> str:
    text = response
    if "####" in text:
        text = text.rsplit("####", 1)[1]
    matches = NUMBER_RE.findall(text)
    return matches[-1].strip() if matches else text.strip()


def normalize_number(text: str):
    value = text.strip().replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    try:
        if "/" in value:
            left, right = value.split("/", 1)
            return Fraction(Decimal(left.strip())) / Fraction(Decimal(right.strip()))
        return Decimal(value)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return value.lower()


def answers_match(prediction: str, gold: str) -> bool:
    return normalize_number(prediction) == normalize_number(gold)


def build_gsm8k_records(
    data: Dict[str, object],
    max_queries: int | None,
    retrieval_query_field: str = "qa",
):
    rows = list(data["records"])
    if max_queries is not None:
        rows = rows[: max(0, min(max_queries, len(rows)))]

    dataset_name = data.get("source_dataset", "openai/gsm8k")
    train_records: List[Dict[str, str]] = []
    train_meta: List[Dict[str, object]] = []
    test_records: List[Dict[str, str]] = []
    test_meta: List[Dict[str, object]] = []

    for row in rows:
        source_id = f"{dataset_name}:{row['index']}"
        english_fields = row.get("translations", {}).get("en", row["source"])
        if retrieval_query_field == "question":
            query_text = format_gsm8k_question(english_fields, "en")
        else:
            query_text = format_gsm8k(english_fields, "en")
        test_records.append({"text": query_text})
        test_meta.append(
            {
                "test_index": len(test_meta),
                "dataset": dataset_name,
                "index": row["index"],
                "source_id": source_id,
                "question": str(english_fields["question"]),
                "answer": str(english_fields["answer"]),
                "gold_final": extract_gold_answer(str(english_fields["answer"])),
                "retrieval_query_field": retrieval_query_field,
            }
        )

        for language in LANGUAGES:
            fields = row["translations"][language]
            text = format_gsm8k(fields, language)
            train_records.append({"text": text})
            train_meta.append(
                {
                    "train_index": len(train_meta),
                    "dataset": dataset_name,
                    "index": row["index"],
                    "source_id": source_id,
                    "language": language,
                    "question": str(fields["question"]),
                    "answer": str(fields["answer"]),
                    "text": text,
                }
            )

    return train_records, train_meta, test_records, test_meta


def load_model_and_tokenizer(args: argparse.Namespace, embed_device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    dtype = torch.float16 if embed_device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.to(embed_device)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def build_top1_records(
    score_matrix: np.ndarray,
    train_meta: List[Dict[str, object]],
    test_meta: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for test_index, score_row in enumerate(score_matrix):
        train_index = int(np.argsort(score_row)[::-1][0])
        top1 = dict(train_meta[train_index])
        top1["rank"] = 1
        top1["score"] = float(score_row[train_index])
        top1["same_source"] = top1["source_id"] == test_meta[test_index]["source_id"]
        records.append({"query": dict(test_meta[test_index]), "top1": top1})
    return records


def format_agent_prompt(
    tokenizer,
    query: Dict[str, object],
    memory: Dict[str, object],
    use_chat_template: bool,
    enable_thinking: bool = False,
) -> str:
    user_content = (
        "A retrieved memory item is provided. It may be a translated worked example from GSM8K. "
        "Use it only if it helps; do not copy an unrelated answer.\n\n"
        f"Retrieved memory language: {memory['language']}\n"
        f"Retrieved memory question:\n{memory['question']}\n\n"
        f"Retrieved memory answer:\n{memory['answer']}\n\n"
        "Now solve the English question. Show concise reasoning and finish with exactly one final line "
        "in the format #### <number>.\n\n"
        f"English question:\n{query['question']}"
    )
    if not use_chat_template:
        return user_content + "\nAnswer:"
    messages = [
        {
            "role": "system",
            "content": "You are a careful math problem solver. Always end with #### followed by the final numeric answer.",
        },
        {"role": "user", "content": user_content},
    ]
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in tokenizer.name_or_path.lower():
        template_kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
) -> List[str]:
    tokenizer.padding_side = "left"
    responses: List[str] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating answers"):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        ).to(device)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[:, input_length:]
        responses.extend(tokenizer.batch_decode(completion_ids, skip_special_tokens=True))
        del inputs, generated, completion_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    tokenizer.padding_side = "right"
    return responses


def summarize_channel(records: List[Dict[str, object]], num_memory_candidates: int) -> Dict[str, object]:
    retrieval_correct = [int(item["top1"]["same_source"]) for item in records]
    answer_correct = [int(item["correct"]) for item in records]
    same_records = [item for item in records if item["top1"]["same_source"]]
    diff_records = [item for item in records if not item["top1"]["same_source"]]

    def acc(items: Iterable[Dict[str, object]]) -> float | None:
        items = list(items)
        if not items:
            return None
        return float(np.mean([int(item["correct"]) for item in items]))

    return {
        "num_queries": len(records),
        "num_memory_candidates": int(num_memory_candidates),
        "top1_retrieval_accuracy": float(np.mean(retrieval_correct)) if records else 0.0,
        "top1_retrieval_correct": int(np.sum(retrieval_correct)),
        "answer_accuracy": float(np.mean(answer_correct)) if records else 0.0,
        "answer_correct": int(np.sum(answer_correct)),
        "answer_accuracy_when_top1_same_source": acc(same_records),
        "answer_accuracy_when_top1_different_source": acc(diff_records),
    }


def write_result(path: Path, result: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_path}")

    embed_device = resolve_device(args.embed_device)
    score_device = resolve_device(args.score_device)
    use_chat_template = not args.no_chat_template

    data = json.loads(args.data_path.read_text(encoding="utf-8"))
    train_records, train_meta, test_records, test_meta = build_gsm8k_records(
        data,
        args.max_queries,
        args.retrieval_query_field,
    )
    print(f"memory translations: {len(train_records)}")
    print(f"english queries: {len(test_records)}")

    tokenizer, model = load_model_and_tokenizer(args, embed_device)
    any_gh = any(channel in {"gh", "both"} for channel in args.channels)
    retrieval_train_channels = ("rh", "gh") if any_gh else ("rh",)
    lm_head_weight = get_lm_head_weight(model) if any_gh else None
    gh_embedding_layers = None
    if any_gh:
        gh_embedding_layers = resolve_gh_embedding_layers(
            parse_gh_embedding_layers(args.gh_embedding_layers),
            model,
            args.gh_layer_index_mode,
        )
        print(f"GH embedding layers: {gh_embedding_layers}")
        print(f"GH layer index mode: {args.gh_layer_index_mode}")
        print(f"GH use input layernorm: {args.gh_use_input_layernorm}")

    train_loader = DataLoader(
        GRPO_dataset(Dataset.from_list(train_records), tokenizer, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        GRPO_dataset(Dataset.from_list(test_records), tokenizer, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"building train representations for: {', '.join(retrieval_train_channels)}")
    train_representations = compute_train_representations(
        dataloader_train=train_loader,
        model=model,
        embed_device=embed_device,
        prediction_topk=args.prediction_topk,
        vocab_mode="topk_unique",
        lowest_likelihood_ratio=1.0,
        global_vocab_ids_cpu=None,
        readout_channels=retrieval_train_channels,
        lm_head_weight=lm_head_weight,
        gh_embedding_layers=gh_embedding_layers,
        gh_layer_index_mode=args.gh_layer_index_mode,
        gh_use_input_layernorm=args.gh_use_input_layernorm,
        compute_proposed=True,
    )

    result: Dict[str, object] = {
        "summary": {
            "method": "agent memory top1 retrieval + greedy GSM8K answer generation",
            "model_name": args.model_name,
            "data_path": str(args.data_path),
            "num_queries": len(test_records),
            "num_memory_candidates": len(train_records),
            "memory_languages": list(LANGUAGES),
            "channels": list(args.channels),
            "gh_embedding_layers": gh_embedding_layers,
            "gh_layer_index_mode": args.gh_layer_index_mode if any_gh else None,
            "gh_use_input_layernorm": bool(args.gh_use_input_layernorm) if any_gh else False,
            "max_length": args.max_length,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "prediction_topk": args.prediction_topk,
            "retrieval_query_field": args.retrieval_query_field,
            "chat_template": use_chat_template,
            "enable_thinking": args.enable_thinking,
            "decoding": "greedy",
        },
        "channels": {},
    }

    for channel in args.channels:
        readout_channels = normalize_readout_channels(None, channel)
        print(f"scoring channel: {channel} ({', '.join(readout_channels)})")
        scores = score_test_streaming(
            dataloader_test=test_loader,
            model=model,
            train_representations=train_representations,
            embed_device=embed_device,
            score_device=score_device,
            prediction_topk=args.prediction_topk,
            train_score_chunk=args.train_score_chunk,
            vocab_mode="topk_unique",
            lowest_likelihood_ratio=1.0,
            global_vocab_ids_cpu=None,
            readout_channels=readout_channels,
            lm_head_weight=lm_head_weight if "gh" in readout_channels else None,
            gh_embedding_layers=gh_embedding_layers if "gh" in readout_channels else None,
            gh_layer_index_mode=args.gh_layer_index_mode,
            gh_use_input_layernorm=args.gh_use_input_layernorm,
            approximate_proposed=False,
        )
        channel_records = build_top1_records(
            score_matrix=scores.detach().cpu().float().numpy(),
            train_meta=train_meta,
            test_meta=test_meta,
        )
        del scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        prompts = [
            format_agent_prompt(
                tokenizer,
                item["query"],
                item["top1"],
                use_chat_template,
                args.enable_thinking,
            )
            for item in channel_records
        ]
        responses = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=embed_device,
            batch_size=args.generation_batch_size,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
        )

        for item, prompt, response in zip(channel_records, prompts, responses):
            gold = str(item["query"]["gold_final"])
            pred = extract_prediction_answer(response)
            item["prompt"] = prompt
            item["response"] = response
            item["pred_final"] = pred
            item["correct"] = answers_match(pred, gold)

        result["channels"][channel] = {
            "summary": summarize_channel(channel_records, len(train_meta)),
            "per_query": channel_records,
        }
        result["summary"][f"{channel}_answer_accuracy"] = result["channels"][channel]["summary"]["answer_accuracy"]
        result["summary"][f"{channel}_top1_retrieval_accuracy"] = result["channels"][channel]["summary"]["top1_retrieval_accuracy"]
        write_result(args.output_path, result)
        print(json.dumps(result["channels"][channel]["summary"], ensure_ascii=False, indent=2))
        print(f"updated {args.output_path}")

    write_result(args.output_path, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
