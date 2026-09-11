from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .common import write_csv


DEFAULT_SCORE_COLUMNS = (
    "baseline_last_hidden_cosine_mean",
    "baseline_last_hidden_dot_mean",
    "ifmass_ch1_mean",
    "ifmass_ch2_mean",
    "ifmass_total_mean",
    "ifmass_ch1_mean_abs_mean",
    "ifmass_ch2_mean_abs_mean",
    "ifmass_total_mean_abs_mean",
)


def read_csv(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate subject-wise erosion with pre-finetuning scores.")
    parser.add_argument("--behavior_file", required=True)
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--condition", choices=("sft", "eaft"), default="sft")
    parser.add_argument("--score_columns", default=None, help="Comma-separated columns; defaults to paper scores and baselines.")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    from scipy.stats import pearsonr, spearmanr

    behavior = {row["subject"]: row for row in read_csv(args.behavior_file)}
    scores = {row["subject"]: row for row in read_csv(args.score_file)}
    subjects = sorted(set(behavior) & set(scores))
    if len(subjects) != 57:
        raise ValueError(f"Expected 57 shared MMLU subjects, found {len(subjects)}.")

    requested = (
        [item.strip() for item in args.score_columns.split(",") if item.strip()]
        if args.score_columns
        else list(DEFAULT_SCORE_COLUMNS)
    )
    missing = [column for column in requested if column not in scores[subjects[0]]]
    if missing:
        raise ValueError(f"Score columns not found: {missing}")

    rows = []
    for behavior_name in ("non_if_rate", "hash_rate"):
        behavior_column = f"{args.condition}_{behavior_name}"
        y = [float(behavior[subject][behavior_column]) for subject in subjects]
        for score_column in requested:
            x = [float(scores[subject][score_column]) for subject in subjects]
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            rows.append(
                {
                    "condition": args.condition,
                    "behavior": behavior_name,
                    "score": score_column,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_r": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "num_subjects": len(subjects),
                }
            )
    if any(not math.isfinite(row["pearson_r"]) or not math.isfinite(row["spearman_r"]) for row in rows):
        raise ValueError("A requested score or behavior column is constant; correlation is undefined.")
    write_csv(args.output_file, rows)


if __name__ == "__main__":
    main()
