from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, write_csv, write_json
from .prompts import MMLU_CHOICES


EOS_MARKERS = ("<|endoftext|>", "<|im_end|>", "</s>", "<eos>", "<EOS>")
CONDITIONS = ("base", "sft", "eaft")


def stripped_response(text: str) -> str:
    value = (text or "").strip()
    changed = True
    while changed:
        changed = False
        for marker in EOS_MARKERS:
            if value.endswith(marker):
                value = value[: -len(marker)].strip()
                changed = True
    return value


def index_records(path: str) -> dict[str, dict[str, Any]]:
    return {row["example_id"]: row for row in read_jsonl(path)}


def summarize_group(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    total = len(rows)
    exact = sum(stripped_response(row[f"{condition}_prediction"]) in MMLU_CHOICES for row in rows)
    hashes = sum("####" in row[f"{condition}_prediction"] for row in rows)
    generation_correct = sum(int(row[f"{condition}_generation_correct"]) for row in rows)
    likelihood_correct = sum(int(row[f"{condition}_choice_argmax_correct"]) for row in rows)
    choice_mass = sum(float(row[f"{condition}_choice_mass"]) for row in rows)
    return {
        "num_examples": total,
        "exact_count": exact,
        "exact_rate": exact / total,
        "non_if_count": total - exact,
        "non_if_rate": (total - exact) / total,
        "hash_count": hashes,
        "hash_rate": hashes / total,
        "generation_correct_count": generation_correct,
        "generation_accuracy": generation_correct / total,
        "likelihood_correct_count": likelihood_correct,
        "likelihood_accuracy": likelihood_correct / total,
        "mean_choice_mass": choice_mass / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize whole-dataset and subject-wise MMLU behavior.")
    parser.add_argument("--base_file", required=True)
    parser.add_argument("--sft_file", required=True)
    parser.add_argument("--eaft_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    indexed = {
        "base": index_records(args.base_file),
        "sft": index_records(args.sft_file),
        "eaft": index_records(args.eaft_file),
    }
    ids = set(indexed["base"])
    if any(set(indexed[condition]) != ids for condition in CONDITIONS[1:]):
        raise ValueError("Base, SFT, and EAFT files must contain identical example IDs.")

    merged = []
    for example_id in sorted(ids):
        base = indexed["base"][example_id]
        row: dict[str, Any] = {
            "example_id": example_id,
            "subject": base["metadata"]["subject"],
            "target": base["target"],
        }
        for condition in CONDITIONS:
            source = indexed[condition][example_id]
            for field in (
                "prediction",
                "parsed_prediction",
                "generation_correct",
                "choice_argmax",
                "choice_argmax_correct",
                "choice_mass",
            ):
                row[f"{condition}_{field}"] = source[field]
        merged.append(row)

    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        by_subject[row["subject"]].append(row)

    subject_rows = []
    for subject, rows in sorted(by_subject.items()):
        output: dict[str, Any] = {"subject": subject, "num_examples": len(rows)}
        for condition in CONDITIONS:
            stats = summarize_group(rows, condition)
            output.update({f"{condition}_{key}": value for key, value in stats.items() if key != "num_examples"})
        subject_rows.append(output)

    overall_rows = []
    for condition in CONDITIONS:
        overall_rows.append({"condition": condition, **summarize_group(merged, condition)})

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "merged_examples.csv", merged)
    write_csv(output_dir / "subject_behavior.csv", subject_rows)
    write_csv(output_dir / "overall_behavior.csv", overall_rows)
    write_json(output_dir / "overall_behavior.json", {row["condition"]: row for row in overall_rows})


if __name__ == "__main__":
    main()
