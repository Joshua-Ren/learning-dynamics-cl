#!/usr/bin/env python
"""Generate GSM8K answers from saved deterministic top-1 selections."""
import argparse
import json
from pathlib import Path
from typing import Dict, List

from run_gsm8k_memory_agent_top1 import (
    LANGUAGES,
    answers_match,
    build_gsm8k_records,
    extract_prediction_answer,
    format_agent_prompt,
    generate_responses,
    load_model_and_tokenizer,
    resolve_device,
    summarize_channel,
)


def parse_channel_path(raw: str):
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected channel=path")
    channel, path = raw.split("=", 1)
    return channel, Path(path)


def build_top1_records(value_data, train_meta, test_meta):
    records = []
    for test_index, query_value in enumerate(value_data["per_query"]):
        value_item = query_value["top4"][0]
        train_index = int(value_item["train_index"])
        memory = dict(train_meta[train_index])
        memory["rank"] = 1
        memory["score"] = float(value_item["score"])
        memory["same_source"] = memory["source_id"] == test_meta[test_index]["source_id"]
        records.append({"query": dict(test_meta[test_index]), "top1": memory})
    return records


def main():
    parser = argparse.ArgumentParser(description="Run GSM8K top1 agent memory from saved selection JSONs.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--selection", action="append", type=parse_channel_path, required=True, help="channel=selection_json")
    parser.add_argument("--retrieval_query_field", choices=("qa", "question"), default="question")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen3 thinking mode. Disabled by default to match the filtering baseline.",
    )
    args = parser.parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.data_path.read_text(encoding="utf-8"))
    _, train_meta, _, test_meta = build_gsm8k_records(data, None, args.retrieval_query_field)

    model_args = argparse.Namespace(model_name=args.model_name, local_files_only=args.local_files_only)
    device = resolve_device("auto")
    tokenizer, model = load_model_and_tokenizer(model_args, device)

    result: Dict[str, object] = {
        "summary": {
            "method": "agent memory top1 retrieval from saved selection outputs + greedy GSM8K answer generation",
            "model_name": args.model_name,
            "data_path": str(args.data_path),
            "num_queries": len(test_meta),
            "num_memory_candidates": len(train_meta),
            "memory_languages": list(LANGUAGES),
            "channels": [channel for channel, _ in args.selection],
            "selection_paths": {channel: str(path) for channel, path in args.selection},
            "retrieval_query_field": args.retrieval_query_field,
            "max_new_tokens": args.max_new_tokens,
            "chat_template": True,
            "enable_thinking": args.enable_thinking,
            "decoding": "greedy",
            "max_prompt_length": args.max_prompt_length,
            "generation_batch_size": args.generation_batch_size,
        },
        "channels": {},
    }

    for channel, path in args.selection:
        value_data = json.loads(path.read_text(encoding="utf-8"))
        if len(value_data.get("per_query", [])) != len(test_meta):
            raise ValueError(
                f"{path} has {len(value_data.get('per_query', []))} queries; "
                f"dataset has {len(test_meta)}"
            )
        records = build_top1_records(value_data, train_meta, test_meta)
        prompts = [
            format_agent_prompt(
                tokenizer,
                item["query"],
                item["top1"],
                True,
                args.enable_thinking,
            )
            for item in records
        ]
        responses = generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            batch_size=args.generation_batch_size,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
        )
        for item, prompt, response in zip(records, prompts, responses):
            gold = str(item["query"]["gold_final"])
            pred = extract_prediction_answer(response)
            item["prompt"] = prompt
            item["response"] = response
            item["pred_final"] = pred
            item["correct"] = answers_match(pred, gold)

        summary = summarize_channel(records, len(train_meta))
        summary["responses_missing_final_marker"] = int(sum("####" not in item.get("response", "") for item in records))
        result["channels"][channel] = {"summary": summary, "per_query": records}
        result["summary"][f"{channel}_answer_accuracy"] = summary["answer_accuracy"]
        result["summary"][f"{channel}_top1_retrieval_accuracy"] = summary["top1_retrieval_accuracy"]
        result["summary"][f"{channel}_responses_missing_final_marker"] = summary["responses_missing_final_marker"]
        args.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(channel, json.dumps(summary, ensure_ascii=False, indent=2))

    args.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
