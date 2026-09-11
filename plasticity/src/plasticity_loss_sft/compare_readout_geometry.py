from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from plasticity_loss_sft.modeling import get_lm_head_weight
from plasticity_loss_sft.readout_geometry import (
    compute_readout_geometry,
    load_token_support,
    validate_metrics,
)
from plasticity_loss_sft.run_readout_geometry import resolve_device, resolve_dtype


DEFAULT_SUPPORT_FILES = (
    "data/token_support/qwen25_1p5b/raw_top300.json",
    "data/token_support/qwen25_1p5b/gradient_filtered_top300.json",
)
METRIC_FIELDS = ("trace", "effective_rank", "hoyer_concentration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare readout geometry for base and SFT models.")
    parser.add_argument("--base_model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--trained_model_path", required=True)
    parser.add_argument("--support_files", nargs="+", default=list(DEFAULT_SUPPORT_FILES))
    parser.add_argument("--output_dir", default="analysis/readout_geometry/single_task_gsm8k")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    support_paths = [Path(path) for path in args.support_files]
    supports = {path.stem: load_token_support(path) for path in support_paths}

    base_results = compute_model_results(
        model_name_or_path=args.base_model_name,
        model_label="base",
        supports=supports,
        output_dir=output_dir,
        device=device,
        dtype=dtype,
    )
    trained_results = compute_model_results(
        model_name_or_path=args.trained_model_path,
        model_label="gsm8k_trained",
        supports=supports,
        output_dir=output_dir,
        device=device,
        dtype=dtype,
    )

    comparison_rows = build_comparison_rows(base_results, trained_results)
    report = {
        "base_model_name": args.base_model_name,
        "trained_model_path": args.trained_model_path,
        "device": str(device),
        "dtype": str(dtype),
        "support_files": [str(path) for path in support_paths],
        "models": {
            "base": base_results,
            "gsm8k_trained": trained_results,
        },
        "comparison": comparison_rows,
    }
    write_json(output_dir / "single_task_gsm8k_readout_comparison.json", report)
    write_comparison_csv(output_dir / "single_task_gsm8k_readout_comparison.csv", comparison_rows)
    write_comparison_markdown(output_dir / "single_task_gsm8k_readout_comparison.md", comparison_rows)
    print_comparison(comparison_rows)
    print(f"Wrote comparison: {output_dir / 'single_task_gsm8k_readout_comparison.json'}")


def compute_model_results(
    model_name_or_path: str,
    model_label: str,
    supports: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    lm_head_weight = get_lm_head_weight(model).detach()

    spectra_dir = output_dir / "spectra" / model_label
    spectra_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "lm_head_shape": list(lm_head_weight.shape),
        "supports": {},
    }
    for support_name, support in supports.items():
        results["supports"][support_name] = {
            "support_type": support.support_type,
            "tokenizer_name": support.tokenizer_name,
            "top_k": support.top_k,
            "filtering_rule": support.filtering_rule,
            "tasks": {},
        }
        for task, token_ids in support.task_token_ids.items():
            metrics, spectrum = compute_readout_geometry(
                lm_head_weight=lm_head_weight,
                token_ids=token_ids,
                support_name=support_name,
                support_type=support.support_type,
                task=task,
            )
            validate_metrics(metrics)
            spectrum_path = spectra_dir / f"{support_name}_{task}_eigenvalues.json"
            write_json(spectrum_path, spectrum)
            row = metrics.to_dict()
            row["spectrum_path"] = str(spectrum_path)
            results["supports"][support_name]["tasks"][task] = row

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return results


def build_comparison_rows(base_results: dict[str, Any], trained_results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for support_name, base_support in base_results["supports"].items():
        trained_support = trained_results["supports"][support_name]
        for task, base_metrics in base_support["tasks"].items():
            trained_metrics = trained_support["tasks"][task]
            row: dict[str, Any] = {
                "support_name": support_name,
                "support_type": base_support["support_type"],
                "task": task,
                "token_count": base_metrics["token_count"],
                "hidden_dim": base_metrics["hidden_dim"],
                "base_spectrum_path": base_metrics["spectrum_path"],
                "trained_spectrum_path": trained_metrics["spectrum_path"],
            }
            for field in METRIC_FIELDS:
                base_value = float(base_metrics[field])
                trained_value = float(trained_metrics[field])
                row[f"base_{field}"] = base_value
                row[f"trained_{field}"] = trained_value
                row[f"relative_change_{field}"] = relative_change(trained_value, base_value)
            validate_comparison_row(row)
            rows.append(row)
    return rows


def relative_change(trained_value: float, base_value: float) -> float:
    if base_value == 0.0:
        return math.nan
    return (trained_value - base_value) / base_value


def validate_comparison_row(row: dict[str, Any]) -> None:
    for field in METRIC_FIELDS:
        base_value = float(row[f"base_{field}"])
        trained_value = float(row[f"trained_{field}"])
        change = float(row[f"relative_change_{field}"])
        if not math.isfinite(base_value) or not math.isfinite(trained_value) or not math.isfinite(change):
            raise RuntimeError(f"Non-finite comparison value for {row['support_name']}/{row['task']}/{field}")
        expected = relative_change(trained_value, base_value)
        if not math.isclose(change, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Relative change mismatch for {row['support_name']}/{row['task']}/{field}")


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "support_name",
        "support_type",
        "task",
        "token_count",
        "hidden_dim",
    ]
    for field in METRIC_FIELDS:
        fieldnames.extend([f"base_{field}", f"trained_{field}", f"relative_change_{field}"])
    fieldnames.extend(["base_spectrum_path", "trained_spectrum_path"])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def write_comparison_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Single-Task GSM8K Readout Geometry Comparison",
        "",
        "| Support | Task | Metric | Base | Trained | Relative Change |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        for field in METRIC_FIELDS:
            lines.append(
                f"| `{row['support_name']}` | `{row['task']}` | `{field}` | "
                f"{row[f'base_{field}']:.6f} | {row[f'trained_{field}']:.6f} | "
                f"{row[f'relative_change_{field}']:.8f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_comparison(rows: list[dict[str, Any]]) -> None:
    print("support,task,metric,base,trained,relative_change")
    for row in rows:
        for field in METRIC_FIELDS:
            print(
                f"{row['support_name']},{row['task']},{field},"
                f"{row[f'base_{field}']:.6f},{row[f'trained_{field}']:.6f},"
                f"{row[f'relative_change_{field}']:.8f}"
            )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
