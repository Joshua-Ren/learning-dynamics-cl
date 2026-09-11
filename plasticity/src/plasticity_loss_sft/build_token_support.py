from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any


TASKS = ("gsm8k", "mbpp", "dolly_qa")
DEFAULT_FILTER_MIN = 1e-4
DEFAULT_FILTER_MAX = 1.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical raw and gradient-filtered top-k token support files."
    )
    parser.add_argument("--raw_overlap_dir", default="analysis/token_overlap")
    parser.add_argument("--gu_norm_dir", default="analysis/gu_norms_train")
    parser.add_argument("--analysis_output_dir", default="analysis/token_support")
    parser.add_argument("--canonical_output_dir", default="data/token_support/qwen25_1p5b")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--filter_min", type=float, default=DEFAULT_FILTER_MIN)
    parser.add_argument("--filter_max", type=float, default=DEFAULT_FILTER_MAX)
    parser.add_argument("--table_k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    analysis_output_dir = Path(args.analysis_output_dir)
    canonical_output_dir = Path(args.canonical_output_dir)
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    canonical_output_dir.mkdir(parents=True, exist_ok=True)

    raw_support = load_raw_support(
        raw_overlap_dir=Path(args.raw_overlap_dir),
        tasks=tasks,
        tokenizer_name=args.tokenizer_name,
        top_k=args.top_k,
    )
    gradient_support = build_gradient_filtered_support(
        gu_norm_dir=Path(args.gu_norm_dir),
        tasks=tasks,
        tokenizer_name=args.tokenizer_name,
        top_k=args.top_k,
        filter_min=args.filter_min,
        filter_max=args.filter_max,
    )

    raw_report = add_overlap_report(raw_support, table_k=args.table_k)
    gradient_report = add_overlap_report(gradient_support, table_k=args.table_k)

    write_json(canonical_output_dir / "raw_top300.json", raw_report)
    write_json(canonical_output_dir / "gradient_filtered_top300.json", gradient_report)
    write_json(analysis_output_dir / "raw_top300.json", raw_report)
    write_json(analysis_output_dir / "gradient_filtered_top300.json", gradient_report)
    write_markdown(analysis_output_dir / "gradient_filtered_top300.md", gradient_report)

    print("Gradient-filtered top-token overlap:")
    print_pairwise(gradient_report["overlap"])
    print(f"All-three intersection size: {gradient_report['overlap']['all_tasks_intersection_size']}")
    print(f"Canonical raw support: {canonical_output_dir / 'raw_top300.json'}")
    print(f"Canonical gradient-filtered support: {canonical_output_dir / 'gradient_filtered_top300.json'}")
    print(f"Markdown summary: {analysis_output_dir / 'gradient_filtered_top300.md'}")


def load_raw_support(
    raw_overlap_dir: Path,
    tasks: list[str],
    tokenizer_name: str,
    top_k: int,
) -> dict[str, Any]:
    task_tokens = {}
    for task in tasks:
        path = raw_overlap_dir / f"{task}_top_{top_k}_tokens.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        task_tokens[task] = normalize_top_rows(rows, top_k)
    return {
        "metadata": {
            "support_type": "raw_frequency_topk",
            "tokenizer_name": tokenizer_name,
            "top_k": top_k,
            "filtering_rule": "none",
            "source_files": {
                task: str(raw_overlap_dir / f"{task}_top_{top_k}_tokens.json") for task in tasks
            },
        },
        "tasks": task_tokens,
    }


def build_gradient_filtered_support(
    gu_norm_dir: Path,
    tasks: list[str],
    tokenizer_name: str,
    top_k: int,
    filter_min: float,
    filter_max: float,
) -> dict[str, Any]:
    task_tokens = {}
    filter_counts = {}
    for task in tasks:
        path = gu_norm_dir / f"{task}_train_gu_norm_tokens.jsonl"
        counter, token_strings, total_count, kept_count = count_filtered_tokens(
            path=path,
            filter_min=filter_min,
            filter_max=filter_max,
        )
        task_tokens[task] = [
            {
                "rank": rank,
                "token_id": token_id,
                "token": token_strings[token_id],
                "count": count,
            }
            for rank, (token_id, count) in enumerate(counter.most_common(top_k), start=1)
        ]
        filter_counts[task] = {
            "source_occurrences": total_count,
            "kept_occurrences": kept_count,
            "dropped_occurrences": total_count - kept_count,
        }
    return {
        "metadata": {
            "support_type": "gradient_filtered_frequency_topk",
            "tokenizer_name": tokenizer_name,
            "top_k": top_k,
            "filtering_rule": f"{filter_min:g} < gu_norm_sq < {filter_max:g}",
            "filter_min_exclusive": filter_min,
            "filter_max_exclusive": filter_max,
            "source_files": {
                task: str(gu_norm_dir / f"{task}_train_gu_norm_tokens.jsonl") for task in tasks
            },
            "filter_counts": filter_counts,
        },
        "tasks": task_tokens,
    }


def count_filtered_tokens(
    path: Path,
    filter_min: float,
    filter_max: float,
) -> tuple[Counter[int], dict[int, str], int, int]:
    counter: Counter[int] = Counter()
    token_strings: dict[int, str] = {}
    total_count = 0
    kept_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total_count += 1
            row = json.loads(line)
            gu_norm_sq = float(row["gu_norm_sq"])
            if not filter_min < gu_norm_sq < filter_max:
                continue
            kept_count += 1
            token_id = int(row["token_id"])
            counter[token_id] += 1
            token_strings.setdefault(token_id, str(row["token"]))
    return counter, token_strings, total_count, kept_count


def normalize_top_rows(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    normalized = []
    for rank, row in enumerate(rows[:top_k], start=1):
        normalized.append(
            {
                "rank": rank,
                "token_id": int(row["token_id"]),
                "token": str(row["token"]),
                "count": int(row["count"]),
            }
        )
    return normalized


def add_overlap_report(support: dict[str, Any], table_k: int) -> dict[str, Any]:
    tasks = support["tasks"]
    top_sets = {task: {row["token_id"] for row in rows} for task, rows in tasks.items()}
    counters = {
        task: Counter({row["token_id"]: row["count"] for row in rows}) for task, rows in tasks.items()
    }
    tokens = {
        row["token_id"]: row["token"] for rows in tasks.values() for row in rows
    }
    pairwise = {}
    for left, right in combinations(top_sets, 2):
        intersection = top_sets[left] & top_sets[right]
        union = top_sets[left] | top_sets[right]
        pairwise[f"{left}__{right}"] = {
            "intersection_size": len(intersection),
            "jaccard_similarity": len(intersection) / len(union) if union else 0.0,
        }
    all_shared = set.intersection(*top_sets.values()) if top_sets else set()
    support["overlap"] = {
        "pairwise": pairwise,
        "all_tasks_intersection_size": len(all_shared),
        "shared_by_all": token_table_rows(
            token_ids=sorted(
                all_shared,
                key=lambda token_id: sum(counters[task][token_id] for task in counters),
                reverse=True,
            )[:table_k],
            counters=counters,
            tokens=tokens,
        ),
        "task_specific": {
            task: task_specific_rows(task, counters, top_sets, tokens, table_k) for task in tasks
        },
    }
    return support


def task_specific_rows(
    task: str,
    counters: dict[str, Counter[int]],
    top_sets: dict[str, set[int]],
    tokens: dict[int, str],
    table_k: int,
) -> list[dict[str, Any]]:
    other_tasks = [candidate for candidate in counters if candidate != task]
    ranked = sorted(
        top_sets[task],
        key=lambda token_id: (
            counters[task][token_id] / (1 + max(counters[other][token_id] for other in other_tasks)),
            counters[task][token_id],
        ),
        reverse=True,
    )
    return token_table_rows(ranked[:table_k], counters, tokens, focus_task=task)


def token_table_rows(
    token_ids: list[int],
    counters: dict[str, Counter[int]],
    tokens: dict[int, str],
    focus_task: str | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for token_id in token_ids:
        counts = {task: counters[task][token_id] for task in counters}
        row: dict[str, Any] = {
            "token_id": token_id,
            "token": tokens.get(token_id, ""),
            "counts": counts,
        }
        if focus_task is not None:
            other_counts = [count for task, count in counts.items() if task != focus_task]
            row["specificity_ratio"] = counts[focus_task] / (1 + max(other_counts, default=0))
        rows.append(row)
    return rows


def write_markdown(path: Path, support: dict[str, Any]) -> None:
    metadata = support["metadata"]
    overlap = support["overlap"]
    lines = [
        "# Gradient-Filtered Top-300 Token Support",
        "",
        f"Tokenizer: `{metadata['tokenizer_name']}`",
        f"Top-k: `{metadata['top_k']}`",
        f"Filtering rule: `{metadata['filtering_rule']}`",
        "",
        "## Filter Counts",
        "",
        "| Task | Source occurrences | Kept | Dropped | Kept fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task, counts in metadata["filter_counts"].items():
        kept_fraction = counts["kept_occurrences"] / counts["source_occurrences"]
        lines.append(
            f"| `{task}` | {counts['source_occurrences']} | {counts['kept_occurrences']} | "
            f"{counts['dropped_occurrences']} | {kept_fraction:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Overlap",
            "",
            "| Pair | Intersection | Jaccard |",
            "| --- | ---: | ---: |",
        ]
    )
    for pair, values in overlap["pairwise"].items():
        lines.append(
            f"| `{pair}` | {values['intersection_size']} | "
            f"{values['jaccard_similarity']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"All-task intersection size: `{overlap['all_tasks_intersection_size']}`",
            "",
        ]
    )
    append_token_table(lines, "Shared By All", overlap["shared_by_all"])
    for task, rows in overlap["task_specific"].items():
        append_token_table(lines, f"Specific To {task}", rows)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_token_table(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            f"## {title}",
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


def print_pairwise(overlap: dict[str, Any]) -> None:
    for pair, values in overlap["pairwise"].items():
        print(
            f"- {pair}: intersection={values['intersection_size']}, "
            f"jaccard={values['jaccard_similarity']:.4f}"
        )


def markdown_token(token: str) -> str:
    return token.replace("\\", "\\\\").replace("`", "\\`").replace("\n", "\\n").replace("\r", "\\r")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
