from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from plasticity_loss_sft.compare_readout_geometry import compute_model_results
from plasticity_loss_sft.readout_geometry import load_token_support
from plasticity_loss_sft.run_readout_geometry import resolve_device, resolve_dtype


DEFAULT_SUPPORT_FILES = (
    "data/token_support/qwen25_1p5b/raw_top300.json",
    "data/token_support/qwen25_1p5b/gradient_filtered_top300.json",
)
METRIC_FIELDS = ("trace", "effective_rank", "hoyer_concentration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare readout geometry for a base model and multiple staged checkpoints."
    )
    parser.add_argument(
        "--model_specs",
        nargs="+",
        required=True,
        help="Model specs as label=path_or_hf_name. The first spec is used as the comparison baseline.",
    )
    parser.add_argument("--support_files", nargs="+", default=list(DEFAULT_SUPPORT_FILES))
    parser.add_argument("--output_dir", default="analysis/readout_geometry/long_seq_sft_100epoch")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = parse_model_specs(args.model_specs)
    support_paths = [Path(path) for path in args.support_files]
    supports = {path.stem: load_token_support(path) for path in support_paths}
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    model_results: dict[str, Any] = {}
    for spec in model_specs:
        print(f"Computing readout geometry for {spec['label']}: {spec['model_name_or_path']}")
        model_results[spec["label"]] = compute_model_results(
            model_name_or_path=spec["model_name_or_path"],
            model_label=spec["label"],
            supports=supports,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
        )

    baseline_label = model_specs[0]["label"]
    model_order = [spec["label"] for spec in model_specs]
    comparison_rows = build_comparison_rows(model_results, model_order)
    report = {
        "baseline_label": baseline_label,
        "comparison_reference": "previous_stage",
        "model_order": model_order,
        "device": str(device),
        "dtype": str(dtype),
        "support_files": [str(path) for path in support_paths],
        "models": model_results,
        "comparison": comparison_rows,
    }
    write_json(output_dir / "readout_geometry_stage_comparison.json", report)
    write_comparison_csv(output_dir / "readout_geometry_stage_comparison.csv", comparison_rows)
    write_comparison_markdown(output_dir / "readout_geometry_stage_comparison.md", comparison_rows)
    print_comparison(comparison_rows)
    print(f"Wrote comparison: {output_dir / 'readout_geometry_stage_comparison.json'}")


def parse_model_specs(values: list[str]) -> list[dict[str, str]]:
    specs = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected model spec label=path_or_hf_name, got {value!r}")
        label, model_name_or_path = value.split("=", 1)
        label = label.strip()
        model_name_or_path = model_name_or_path.strip()
        if not label or not model_name_or_path:
            raise ValueError(f"Invalid empty model spec component in {value!r}")
        if label in seen:
            raise ValueError(f"Duplicate model label: {label}")
        seen.add(label)
        specs.append({"label": label, "model_name_or_path": model_name_or_path})
    if not specs:
        raise ValueError("At least one model spec is required.")
    return specs


def build_comparison_rows(model_results: dict[str, Any], model_order: list[str]) -> list[dict[str, Any]]:
    rows = []
    for index, model_label in enumerate(model_order):
        results = model_results[model_label]
        reference_label = model_label if index == 0 else model_order[index - 1]
        reference = model_results[reference_label]
        for support_name, support in results["supports"].items():
            reference_support = reference["supports"][support_name]
            for task, metrics in support["tasks"].items():
                reference_metrics = reference_support["tasks"][task]
                row: dict[str, Any] = {
                    "model_label": model_label,
                    "reference_label": reference_label,
                    "support_name": support_name,
                    "support_type": support["support_type"],
                    "task": task,
                    "token_count": metrics["token_count"],
                    "hidden_dim": metrics["hidden_dim"],
                    "spectrum_path": metrics["spectrum_path"],
                    "reference_spectrum_path": reference_metrics["spectrum_path"],
                }
                for field in METRIC_FIELDS:
                    value = float(metrics[field])
                    reference_value = float(reference_metrics[field])
                    row[field] = value
                    row[f"reference_{field}"] = reference_value
                    row[f"relative_change_{field}"] = relative_change(value, reference_value)
                validate_row(row)
                rows.append(row)
    return rows


def relative_change(value: float, reference_value: float) -> float:
    if reference_value == 0.0:
        return math.nan
    return (value - reference_value) / reference_value


def validate_row(row: dict[str, Any]) -> None:
    for field in METRIC_FIELDS:
        value = float(row[field])
        reference_value = float(row[f"reference_{field}"])
        change = float(row[f"relative_change_{field}"])
        if not math.isfinite(value) or not math.isfinite(reference_value) or not math.isfinite(change):
            raise RuntimeError(f"Non-finite value for {row['model_label']}/{row['support_name']}/{row['task']}/{field}")
        expected = relative_change(value, reference_value)
        if not math.isclose(change, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Relative change mismatch for {row['model_label']}/{row['support_name']}/{row['task']}/{field}")


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model_label",
        "reference_label",
        "support_name",
        "support_type",
        "task",
        "token_count",
        "hidden_dim",
    ]
    for field in METRIC_FIELDS:
        fieldnames.extend([field, f"reference_{field}", f"relative_change_{field}"])
    fieldnames.extend(["spectrum_path", "reference_spectrum_path"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def write_comparison_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Readout Geometry Stage Comparison",
        "",
        "Relative columns compare each row to the immediately previous model stage. The base rows compare to themselves.",
        "",
        "| Model | Reference | Support | Task | Trace | Eff. Rank | Hoyer | rel Trace | rel Eff. Rank | rel Hoyer |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model_label']}` | `{row['reference_label']}` | `{row['support_name']}` | `{row['task']}` | "
            f"{row['trace']:.6f} | {row['effective_rank']:.6f} | "
            f"{row['hoyer_concentration']:.6f} | "
            f"{row['relative_change_trace']:.8f} | "
            f"{row['relative_change_effective_rank']:.8f} | "
            f"{row['relative_change_hoyer_concentration']:.8f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_comparison(rows: list[dict[str, Any]]) -> None:
    print("model,reference,support,task,trace,effective_rank,hoyer,rel_trace,rel_effective_rank,rel_hoyer")
    for row in rows:
        print(
            f"{row['model_label']},{row['reference_label']},{row['support_name']},{row['task']},"
            f"{row['trace']:.6f},{row['effective_rank']:.6f},{row['hoyer_concentration']:.6f},"
            f"{row['relative_change_trace']:.8f},{row['relative_change_effective_rank']:.8f},"
            f"{row['relative_change_hoyer_concentration']:.8f}"
        )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
