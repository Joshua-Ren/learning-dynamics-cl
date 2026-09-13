"""Compare per-example MMLU predictions to quantify answer-format migration."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a base MMLU run against GSM8K format conditions. "
            "Each comparison must be NAME=PATH_TO_PREDICTIONS_JSONL."
        )
    )
    parser.add_argument("--baseline", required=True, help="Base-model MMLU prediction JSONL")
    parser.add_argument("--comparisons", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_markdown", default=None)
    return parser.parse_args()


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty comparison name in {value!r}")
        if name in result:
            raise ValueError(f"Duplicate comparison name {name!r}")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        result[name] = path
    return result


def load_by_id(path: str | Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        example_id = str(record.get("example_id", ""))
        if not example_id:
            raise KeyError(f"{path}: record lacks example_id")
        if example_id in loaded:
            raise ValueError(f"{path}: duplicate example_id {example_id}")
        required = {"answer_format", "content_correct", "strict_correct"}
        missing = required - set(record)
        if missing:
            raise KeyError(
                f"{path}: record {example_id} lacks {sorted(missing)}; "
                "use evaluate_mmlu.py to produce the predictions."
            )
        loaded[example_id] = record
    if not loaded:
        raise ValueError(f"No predictions in {path}")
    return loaded


def canonical(record: dict[str, Any]) -> bool:
    return bool(record.get("format_is_canonical", record["answer_format"] == "letter_only"))


def content_correct(record: dict[str, Any]) -> bool:
    return bool(record["content_correct"])


def strict_correct(record: dict[str, Any]) -> bool:
    return bool(record["strict_correct"])


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_run(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(records.values())
    formats = Counter(str(record["answer_format"]) for record in values)
    total = len(values)
    canonical_count = sum(canonical(record) for record in values)
    content_count = sum(content_correct(record) for record in values)
    strict_count = sum(strict_correct(record) for record in values)
    content_noncanonical = sum(
        content_correct(record) and not canonical(record) for record in values
    )
    return {
        "num_examples": total,
        "relaxed_accuracy": ratio(content_count, total),
        "strict_accuracy": ratio(strict_count, total),
        "canonical_letter_only_ratio": ratio(canonical_count, total),
        "format_error_ratio": 1.0 - ratio(canonical_count, total),
        "correct_but_noncanonical_ratio": ratio(content_noncanonical, total),
        "accuracy_lost_to_format_ratio": ratio(content_count - strict_count, total),
        "answer_format_counts": dict(sorted(formats.items())),
    }


def compare_pair(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        only_baseline = len(baseline_ids - candidate_ids)
        only_candidate = len(candidate_ids - baseline_ids)
        raise ValueError(
            "Prediction files do not contain the same MMLU examples: "
            f"{only_baseline} only in baseline, {only_candidate} only in comparison."
        )

    ids = sorted(baseline_ids)
    total = len(ids)
    transitions = Counter()
    format_changed = 0
    canonical_to_noncanonical = 0
    noncanonical_to_canonical = 0
    same_letter_format_changed = 0
    baseline_strict_to_format_only_failure = 0
    baseline_strict_to_content_failure = 0
    baseline_strict_lost = 0

    for example_id in ids:
        base = baseline[example_id]
        trial = candidate[example_id]
        base_format = str(base["answer_format"])
        trial_format = str(trial["answer_format"])
        transitions[f"{base_format} -> {trial_format}"] += 1
        changed = base_format != trial_format
        format_changed += int(changed)
        canonical_to_noncanonical += int(canonical(base) and not canonical(trial))
        noncanonical_to_canonical += int(not canonical(base) and canonical(trial))
        same_letter_format_changed += int(
            changed
            and bool(str(base.get("predicted_letter", "")))
            and base.get("predicted_letter") == trial.get("predicted_letter")
        )
        format_only_failure = (
            strict_correct(base) and content_correct(trial) and not canonical(trial)
        )
        baseline_strict_to_format_only_failure += int(format_only_failure)
        baseline_strict_to_content_failure += int(
            strict_correct(base) and not content_correct(trial)
        )
        baseline_strict_lost += int(strict_correct(base) and not strict_correct(trial))

    base_summary = summarize_run(baseline)
    trial_summary = summarize_run(candidate)
    base_canonical = sum(canonical(record) for record in baseline.values())
    base_strict = sum(strict_correct(record) for record in baseline.values())
    return {
        "num_examples": total,
        "answer_format_changed_ratio": ratio(format_changed, total),
        "canonical_to_noncanonical_ratio": ratio(canonical_to_noncanonical, total),
        "canonical_to_noncanonical_given_baseline_canonical": ratio(
            canonical_to_noncanonical, base_canonical
        ),
        "noncanonical_to_canonical_ratio": ratio(noncanonical_to_canonical, total),
        "same_predicted_letter_but_format_changed_ratio": ratio(
            same_letter_format_changed, total
        ),
        "baseline_strict_correct_to_format_only_failure_ratio": ratio(
            baseline_strict_to_format_only_failure, total
        ),
        "baseline_strict_correct_to_content_failure_ratio": ratio(
            baseline_strict_to_content_failure, total
        ),
        "format_mediated_share_of_baseline_strict_losses": ratio(
            baseline_strict_to_format_only_failure, baseline_strict_lost
        ),
        "baseline_strict_loss_ratio": ratio(baseline_strict_lost, base_strict),
        "relaxed_accuracy_delta": (
            trial_summary["relaxed_accuracy"] - base_summary["relaxed_accuracy"]
        ),
        "strict_accuracy_delta": trial_summary["strict_accuracy"] - base_summary["strict_accuracy"],
        "format_error_ratio_delta": (
            trial_summary["format_error_ratio"] - base_summary["format_error_ratio"]
        ),
        "answer_format_transitions": dict(sorted(transitions.items())),
    }


def by_subject(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for example_id, record in baseline.items():
        grouped_ids[str(record.get("subject", "unknown"))].append(example_id)

    results = {}
    for subject, ids in sorted(grouped_ids.items()):
        base_group = {example_id: baseline[example_id] for example_id in ids}
        candidate_group = {example_id: candidate[example_id] for example_id in ids}
        pair = compare_pair(base_group, candidate_group)
        results[subject] = {
            "num_examples": len(ids),
            "strict_accuracy_delta": pair["strict_accuracy_delta"],
            "relaxed_accuracy_delta": pair["relaxed_accuracy_delta"],
            "answer_format_changed_ratio": pair["answer_format_changed_ratio"],
            "canonical_to_noncanonical_ratio": pair["canonical_to_noncanonical_ratio"],
        }
    return results


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# MMLU answer-format comparison",
        "",
        "Strict accuracy counts an answer only when it is both correct and exactly in the requested letter-only format. Relaxed accuracy extracts an A/B/C/D answer from the response.",
        "",
        "## Per-run metrics",
        "",
        "| Run | Relaxed accuracy | Strict accuracy | Format-error ratio | Correct but noncanonical |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in payload["run_summaries"].items():
        lines.append(
            f"| {name} | {summary['relaxed_accuracy']:.4f} | {summary['strict_accuracy']:.4f} | "
            f"{summary['format_error_ratio']:.4f} | {summary['correct_but_noncanonical_ratio']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Change from baseline",
            "",
            "| Condition | Format changed | Canonical → noncanonical | Strict Δ | Relaxed Δ | Format-mediated strict losses |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, comparison in payload["comparisons"].items():
        lines.append(
            f"| {name} | {comparison['answer_format_changed_ratio']:.4f} | "
            f"{comparison['canonical_to_noncanonical_ratio']:.4f} | "
            f"{comparison['strict_accuracy_delta']:+.4f} | "
            f"{comparison['relaxed_accuracy_delta']:+.4f} | "
            f"{comparison['format_mediated_share_of_baseline_strict_losses']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    comparison_paths = parse_named_paths(args.comparisons)
    baseline = load_by_id(args.baseline)

    payload: dict[str, Any] = {
        "baseline_path": str(Path(args.baseline).resolve()),
        "run_summaries": {"baseline": summarize_run(baseline)},
        "comparisons": {},
        "by_subject": {},
    }
    for name, path in comparison_paths.items():
        candidate = load_by_id(path)
        payload["run_summaries"][name] = summarize_run(candidate)
        payload["comparisons"][name] = compare_pair(baseline, candidate)
        payload["by_subject"][name] = by_subject(baseline, candidate)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    output_markdown = (
        Path(args.output_markdown)
        if args.output_markdown
        else output_json.with_suffix(".md")
    )
    with output_markdown.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown(payload))
    print(f"Saved comparison JSON to {output_json}")
    print(f"Saved comparison table to {output_markdown}")


if __name__ == "__main__":
    main()
