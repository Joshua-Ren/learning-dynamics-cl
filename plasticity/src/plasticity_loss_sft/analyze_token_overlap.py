from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


TASKS = ("gsm8k", "mbpp", "dolly_qa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze assistant-response token overlap across prepared task subsets."
    )
    parser.add_argument("--subsets_dir", default="data/prepared_subsets")
    parser.add_argument("--output_dir", default="analysis/token_overlap")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--table_k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subsets_dir = Path(args.subsets_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    counters: dict[str, Counter[int]] = {}
    top_tokens: dict[str, list[dict[str, Any]]] = {}
    top_sets: dict[str, set[int]] = {}
    totals: dict[str, dict[str, int]] = {}

    for task in tasks:
        train_path = subsets_dir / task / "train.jsonl"
        responses = load_assistant_responses(train_path)
        counter = count_response_tokens(responses, tokenizer)
        counters[task] = counter
        top_tokens[task] = serialize_top_tokens(counter, tokenizer, args.top_k)
        top_sets[task] = {entry["token_id"] for entry in top_tokens[task]}
        totals[task] = {"examples": len(responses), "tokens": sum(counter.values())}
        write_json(output_dir / f"{task}_top_{args.top_k}_tokens.json", top_tokens[task])

    pairwise = compute_pairwise(top_sets)
    shared_ids = set.intersection(*(top_sets[task] for task in tasks)) if tasks else set()
    readable_tables = build_readable_tables(tasks, counters, top_sets, shared_ids, tokenizer, args.table_k)
    report = {
        "tokenizer_name": args.tokenizer_name,
        "subsets_dir": str(subsets_dir),
        "readable_table_path": str(output_dir / "readable_tables.md"),
        "top_k": args.top_k,
        "tasks": tasks,
        "totals": totals,
        "pairwise": pairwise,
        "all_three_intersection_size": len(shared_ids),
        "readable_tables": readable_tables,
        "top_token_files": {
            task: str(output_dir / f"{task}_top_{args.top_k}_tokens.json") for task in tasks
        },
    }
    write_json(output_dir / "overlap_report.json", report)
    write_markdown(output_dir / "readable_tables.md", report, tokenizer)
    print_report(report)


def load_assistant_responses(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared train split: {path}")

    responses = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages") or []
            assistant_messages = [
                message for message in messages if message.get("role") == "assistant"
            ]
            if not assistant_messages:
                raise ValueError(f"No assistant message in {path}:{line_number}")
            responses.append(str(assistant_messages[-1]["content"]))
    return responses


def count_response_tokens(
    responses: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> Counter[int]:
    counter: Counter[int] = Counter()
    for response in responses:
        token_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        counter.update(int(token_id) for token_id in token_ids)
    return counter


def serialize_top_tokens(
    counter: Counter[int],
    tokenizer: PreTrainedTokenizerBase,
    top_k: int,
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "token_id": token_id,
            "token": decode_token(tokenizer, token_id),
            "count": count,
        }
        for rank, (token_id, count) in enumerate(counter.most_common(top_k), start=1)
    ]


def compute_pairwise(top_sets: dict[str, set[int]]) -> dict[str, dict[str, float | int]]:
    results = {}
    for left, right in combinations(top_sets, 2):
        intersection = top_sets[left] & top_sets[right]
        union = top_sets[left] | top_sets[right]
        results[f"{left}__{right}"] = {
            "intersection_size": len(intersection),
            "jaccard_similarity": len(intersection) / len(union) if union else 0.0,
        }
    return results


def build_readable_tables(
    tasks: list[str],
    counters: dict[str, Counter[int]],
    top_sets: dict[str, set[int]],
    shared_ids: set[int],
    tokenizer: PreTrainedTokenizerBase,
    table_k: int,
) -> dict[str, list[dict[str, Any]]]:
    tables = {
        "shared_by_all": [
            token_row(tokenizer, token_id, counters)
            for token_id in sorted(
                shared_ids,
                key=lambda candidate: sum(counters[task][candidate] for task in tasks),
                reverse=True,
            )[:table_k]
        ]
    }
    for task in tasks:
        other_tasks = [candidate for candidate in tasks if candidate != task]
        task_specific = sorted(
            top_sets[task],
            key=lambda token_id: specificity_score(token_id, task, other_tasks, counters),
            reverse=True,
        )
        tables[f"specific_to_{task}"] = [
            token_row(tokenizer, token_id, counters, focus_task=task)
            for token_id in task_specific[:table_k]
        ]
    return tables


def specificity_score(
    token_id: int,
    task: str,
    other_tasks: list[str],
    counters: dict[str, Counter[int]],
) -> tuple[float, int]:
    max_other_count = max((counters[other][token_id] for other in other_tasks), default=0)
    return (counters[task][token_id] / (1 + max_other_count), counters[task][token_id])


def token_row(
    tokenizer: PreTrainedTokenizerBase,
    token_id: int,
    counters: dict[str, Counter[int]],
    focus_task: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "token_id": token_id,
        "token": decode_token(tokenizer, token_id),
        "counts": {task: counters[task][token_id] for task in counters},
    }
    if focus_task is not None:
        other_counts = [count for task, count in row["counts"].items() if task != focus_task]
        row["specificity_ratio"] = counters[focus_task][token_id] / (1 + max(other_counts, default=0))
    return row


def decode_token(tokenizer: PreTrainedTokenizerBase, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> None:
    del tokenizer
    lines = [
        "# Token Overlap Report",
        "",
        f"Tokenizer: `{report['tokenizer_name']}`",
        f"Top-k: `{report['top_k']}`",
        "",
        "## Pairwise Top-Token Overlap",
        "",
        "| Pair | Intersection | Jaccard |",
        "| --- | ---: | ---: |",
    ]
    for pair, values in report["pairwise"].items():
        lines.append(
            f"| `{pair}` | {values['intersection_size']} | "
            f"{values['jaccard_similarity']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"All-three intersection size: `{report['all_three_intersection_size']}`",
            "",
        ]
    )

    for name, rows in report["readable_tables"].items():
        lines.extend(
            [
                f"## {name}",
                "",
                "| Token ID | Token | GSM8K | MBPP | Dolly QA | Specificity |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            counts = row["counts"]
            specificity = row.get("specificity_ratio")
            specificity_text = "" if specificity is None else f"{specificity:.2f}"
            lines.append(
                f"| {row['token_id']} | `{markdown_token(row['token'])}` | "
                f"{counts.get('gsm8k', 0)} | {counts.get('mbpp', 0)} | "
                f"{counts.get('dolly_qa', 0)} | {specificity_text} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_token(token: str) -> str:
    return token.replace("\\", "\\\\").replace("`", "\\`").replace("\n", "\\n").replace("\r", "\\r")


def print_report(report: dict[str, Any]) -> None:
    print("Pairwise top-token overlap:")
    for pair, values in report["pairwise"].items():
        print(
            f"- {pair}: intersection={values['intersection_size']}, "
            f"jaccard={values['jaccard_similarity']:.4f}"
        )
    print(f"All-three intersection size: {report['all_three_intersection_size']}")
    print("Top-token JSON files:")
    for task, path in report["top_token_files"].items():
        print(f"- {task}: {path}")
    print(f"Readable table: {report['readable_table_path']}")


if __name__ == "__main__":
    main()
