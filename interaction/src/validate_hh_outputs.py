import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate HH analysis CSV/JSON outputs.")
    parser.add_argument("--output_dir", type=str, default="./out/hh_analysis_debug")
    parser.add_argument(
        "--allow_legacy_raw",
        action="store_true",
        help="Allow old pair_metrics.csv files that have no metric_variant column.",
    )
    return parser.parse_args()


def load_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_csv(rows, allow_legacy_raw=False):
    if not rows:
        raise AssertionError("pair_metrics.csv has no rows.")

    if "metric_variant" not in rows[0]:
        if not allow_legacy_raw:
            raise AssertionError(
                "pair_metrics.csv has no metric_variant column. This looks like "
                "an older run. Rerun run_hh_analysis.py with the current code to "
                "produce both raw and centered rows, or pass --allow_legacy_raw "
                "to validate finite cosine values only."
            )
        variants = {"legacy_raw"}
    else:
        variants = {row.get("metric_variant") for row in rows}
        if not allow_legacy_raw and ("raw" not in variants or "centered" not in variants):
            raise AssertionError(
                "Expected raw and centered metric_variant rows. Got "
                f"{sorted(variants)}. Rerun run_hh_analysis.py with the current "
                "code so pair_metrics.csv contains both variants."
            )

    if allow_legacy_raw and variants == {None}:
        variants = {"legacy_raw"}

    clipped_count = 0
    for row in rows:
        cosine = float(row["cosine"])
        if not math.isfinite(cosine):
            raise AssertionError(f"Non-finite cosine found: {cosine}")
        clipped = min(1.0, max(-1.0, cosine))
        if clipped != cosine:
            clipped_count += 1
            row["cosine"] = str(clipped)

    return clipped_count


def validate_summary(summary, allow_legacy_raw=False):
    if allow_legacy_raw and "raw" not in summary and "centered" not in summary:
        summary = {"legacy_raw": summary}

    required_variants = ["raw", "centered"]
    if allow_legacy_raw and "legacy_raw" in summary:
        required_variants = ["legacy_raw"]

    for variant in required_variants:
        if variant not in summary:
            raise AssertionError(f"Missing {variant} summary.")
        for layer, stats in summary[variant].items():
            count = stats["count"]
            if count <= 0:
                raise AssertionError(f"{variant} layer {layer} has non-positive count: {count}")
            for key in ["dot_negative_fraction", "cosine_negative_fraction"]:
                value = stats[key]
                if value is None or value < 0.0 or value > 1.0:
                    raise AssertionError(f"{variant} layer {layer} invalid {key}: {value}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = load_rows(output_dir / "pair_metrics.csv")
    summary = load_summary(output_dir / "summary.json")

    clipped_count = validate_csv(rows, allow_legacy_raw=args.allow_legacy_raw)
    validate_summary(summary, allow_legacy_raw=args.allow_legacy_raw)

    print("HH output validation passed")
    print(f"rows: {len(rows)}")
    print(f"clipped_cosine_values: {clipped_count}")
    if "metric_variant" in rows[0]:
        variants = sorted({row.get("metric_variant") for row in rows})
    else:
        variants = ["legacy_raw"]
    print(f"metric_variants: {variants}")


if __name__ == "__main__":
    main()
