from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean
from typing import Any

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

TASKS = ("gsm8k", "mbpp", "dolly_qa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed SFT/probe subsets for plasticity experiments.")
    parser.add_argument("--output_dir", default="data/prepared_subsets")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--train_size", type=int, default=1000)
    parser.add_argument("--probe_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--allow_smaller",
        action="store_true",
        help="Allow tasks with fewer than train_size + probe_size examples; probe size is kept when possible.",
    )
    parser.add_argument("--print_examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    selected_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("tasks", {})
    else:
        manifest = {"tasks": {}}
    manifest.update(
        {
            "seed": args.seed,
            "requested_train_size": args.train_size,
            "requested_probe_size": args.probe_size,
            "tokenizer_name": args.tokenizer_name,
        }
    )
    for task in selected_tasks:
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}. Valid tasks: {', '.join(TASKS)}")
        records = load_task_records(task, args.cache_dir)
        train_records, probe_records = select_subsets(
            records,
            train_size=args.train_size,
            probe_size=args.probe_size,
            seed=args.seed,
            task=task,
            allow_smaller=args.allow_smaller,
        )
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(task_dir / "train.jsonl", train_records)
        write_jsonl(task_dir / "probe.jsonl", probe_records)
        selected_ids = {
            "seed": args.seed,
            "source_count": len(records),
            "requested_train_size": args.train_size,
            "requested_probe_size": args.probe_size,
            "actual_train_size": len(train_records),
            "actual_probe_size": len(probe_records),
            "train_ids": [record["example_id"] for record in train_records],
            "probe_ids": [record["example_id"] for record in probe_records],
        }
        write_json(task_dir / "selected_ids.json", selected_ids)

        task_stats = {
            "train": compute_stats(train_records, tokenizer),
            "probe": compute_stats(probe_records, tokenizer),
        }
        write_json(task_dir / "stats.json", task_stats)
        manifest["tasks"][task] = {
            "source_count": len(records),
            "train_path": str(task_dir / "train.jsonl"),
            "probe_path": str(task_dir / "probe.jsonl"),
            "selected_ids_path": str(task_dir / "selected_ids.json"),
            "stats_path": str(task_dir / "stats.json"),
            "stats": task_stats,
        }
        print_task_report(task, train_records, probe_records, task_stats, args.print_examples)

    write_json(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")


def load_task_records(task: str, cache_dir: str | None) -> list[dict[str, Any]]:
    if task == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache_dir)
        return [gsm8k_record(row, idx) for idx, row in enumerate(dataset)]
    if task == "dolly_qa":
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train", cache_dir=cache_dir)
        records = []
        for idx, row in enumerate(dataset):
            if row.get("category") == "closed_qa":
                records.append(dolly_qa_record(row, idx))
        return records
    if task == "mbpp":
        parts = []
        for split in ("train", "validation", "test", "prompt"):
            part = load_dataset("google-research-datasets/mbpp", "full", split=split, cache_dir=cache_dir)
            parts.append(part.map(lambda row, split=split: {"source_split": split}))
        dataset = concatenate_datasets(parts)
        return [mbpp_record(row, idx) for idx, row in enumerate(dataset)]
    raise ValueError(task)


def gsm8k_record(row: Mapping[str, Any], idx: int) -> dict[str, Any]:
    example_id = stable_id("gsm8k", "train", idx, row["question"])
    return conversation_record(
        task="gsm8k",
        example_id=example_id,
        user=str(row["question"]),
        assistant=str(row["answer"]),
        metadata={"source": "openai/gsm8k/main:train", "source_index": idx},
    )


def dolly_qa_record(row: Mapping[str, Any], idx: int) -> dict[str, Any]:
    prompt_parts = [str(row["instruction"])]
    context = str(row.get("context") or "").strip()
    if context:
        prompt_parts.extend(["", context])
    example_id = stable_id("dolly_qa", idx, row["instruction"], context)
    return conversation_record(
        task="dolly_qa",
        example_id=example_id,
        user="\n".join(prompt_parts),
        assistant=str(row["response"]),
        metadata={
            "source": "databricks/databricks-dolly-15k:train",
            "source_index": idx,
            "category": row.get("category"),
        },
    )


def mbpp_record(row: Mapping[str, Any], idx: int) -> dict[str, Any]:
    prompt_parts = [str(row["text"])]
    tests = row.get("test_list") or []
    if tests:
        prompt_parts.extend(["", "Tests:"])
        prompt_parts.extend(str(test) for test in tests)
    setup = str(row.get("test_setup_code") or "").strip()
    if setup:
        prompt_parts.extend(["", "Test setup:", setup])
    task_id = row.get("task_id", idx)
    example_id = stable_id("mbpp", row.get("source_split", "unknown"), task_id, row["text"])
    return conversation_record(
        task="mbpp",
        example_id=example_id,
        user="\n".join(prompt_parts),
        assistant=str(row["code"]),
        metadata={
            "source": "google-research-datasets/mbpp/full",
            "source_split": row.get("source_split"),
            "source_index": idx,
            "task_id": task_id,
        },
    )


def conversation_record(
    task: str,
    example_id: str,
    user: str,
    assistant: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task": task,
        "example_id": example_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": dict(metadata),
    }


def select_subsets(
    records: list[dict[str, Any]],
    train_size: int,
    probe_size: int,
    seed: int,
    task: str,
    allow_smaller: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    needed = train_size + probe_size
    if len(records) < needed:
        if not allow_smaller:
            raise RuntimeError(
                f"Task {task} has only {len(records)} examples, but {needed} are required "
                f"for {train_size} train + {probe_size} probe with no overlap."
            )
        if len(records) <= probe_size:
            actual_probe_size = max(0, len(records) // 5)
        else:
            actual_probe_size = probe_size
        actual_train_size = len(records) - actual_probe_size
    else:
        actual_train_size = train_size
        actual_probe_size = probe_size

    indices = list(range(len(records)))
    rng = random.Random(f"{seed}:{task}")
    rng.shuffle(indices)
    train_indices = indices[:actual_train_size]
    probe_indices = indices[actual_train_size : actual_train_size + actual_probe_size]
    train_records = [records[index] | {"subset": "train"} for index in train_indices]
    probe_records = [records[index] | {"subset": "probe"} for index in probe_indices]
    assert set(record["example_id"] for record in train_records).isdisjoint(
        record["example_id"] for record in probe_records
    )
    return train_records, probe_records


def compute_stats(records: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> dict[str, float | int]:
    prompt_lengths = []
    assistant_lengths = []
    for record in records:
        user = record["messages"][0]["content"]
        assistant = record["messages"][1]["content"]
        prompt_lengths.append(len(tokenizer(user, add_special_tokens=False)["input_ids"]))
        assistant_lengths.append(len(tokenizer(assistant, add_special_tokens=False)["input_ids"]))
    return {
        "count": len(records),
        "avg_prompt_tokens": mean(prompt_lengths) if prompt_lengths else 0.0,
        "avg_assistant_tokens": mean(assistant_lengths) if assistant_lengths else 0.0,
        "max_prompt_tokens": max(prompt_lengths) if prompt_lengths else 0,
        "max_assistant_tokens": max(assistant_lengths) if assistant_lengths else 0,
    }


def print_task_report(
    task: str,
    train_records: list[dict[str, Any]],
    probe_records: list[dict[str, Any]],
    stats: Mapping[str, Any],
    print_examples: int,
) -> None:
    print("=" * 80)
    print(f"Task: {task}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    for subset_name, records in (("train", train_records), ("probe", probe_records)):
        print(f"Examples from {task}/{subset_name}:")
        for record in records[:print_examples]:
            user = record["messages"][0]["content"].replace("\n", " ")
            assistant = record["messages"][1]["content"].replace("\n", " ")
            print(f"- {record['example_id']}")
            print(f"  user: {user[:300]}")
            print(f"  assistant: {assistant[:300]}")


def stable_id(*parts: object) -> str:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return digest


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
