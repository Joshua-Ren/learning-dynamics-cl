from __future__ import annotations

import argparse
import csv
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


DEFAULT_SUPPORT_FILES = (
    "data/token_support/qwen25_1p5b/raw_top300.json",
    "data/token_support/qwen25_1p5b/gradient_filtered_top300.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run readout geometry metrics on token supports.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--support_files", nargs="+", default=list(DEFAULT_SUPPORT_FILES))
    parser.add_argument("--output_dir", default="analysis/readout_geometry/qwen25_1p5b")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--determinism_check", action="store_true")
    parser.add_argument("--determinism_tol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    spectra_dir = output_dir / "spectra"
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    lm_head_weight = get_lm_head_weight(model).detach()

    support_paths = [Path(path) for path in args.support_files]
    support_results: dict[str, Any] = {}
    summary_rows = []

    for support_path in support_paths:
        support = load_token_support(support_path)
        support_name = support_path.stem
        support_results[support_name] = {
            "support_file": str(support_path),
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
            if args.determinism_check:
                repeat_metrics, repeat_spectrum = compute_readout_geometry(
                    lm_head_weight=lm_head_weight,
                    token_ids=token_ids,
                    support_name=support_name,
                    support_type=support.support_type,
                    task=task,
                )
                validate_determinism(metrics.to_dict(), repeat_metrics.to_dict(), args.determinism_tol)
                validate_spectrum_determinism(spectrum, repeat_spectrum, args.determinism_tol)

            spectrum_path = spectra_dir / f"{support_name}_{task}_eigenvalues.json"
            write_json(spectrum_path, spectrum)
            row = metrics.to_dict()
            row["spectrum_path"] = str(spectrum_path)
            support_results[support_name]["tasks"][task] = row
            summary_rows.append(row)

    report = {
        "model_name": args.model_name,
        "device": str(device),
        "dtype": str(dtype),
        "lm_head_shape": list(lm_head_weight.shape),
        "support_files": [str(path) for path in support_paths],
        "determinism_check": args.determinism_check,
        "supports": support_results,
    }
    write_json(output_dir / "readout_geometry_report.json", report)
    write_summary_csv(output_dir / "readout_geometry_summary.csv", summary_rows)
    write_summary_markdown(output_dir / "readout_geometry_summary.md", summary_rows)
    print_summary(summary_rows)
    print(f"Wrote report: {output_dir / 'readout_geometry_report.json'}")


def validate_determinism(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> None:
    fields = (
        "trace",
        "effective_rank",
        "hoyer_concentration",
        "min_eigenvalue",
        "max_eigenvalue",
    )
    for field in fields:
        if not math.isclose(float(left[field]), float(right[field]), rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError(f"Determinism check failed for {field}: {left[field]} vs {right[field]}")


def validate_spectrum_determinism(left: list[float], right: list[float], tolerance: float) -> None:
    if len(left) != len(right):
        raise RuntimeError(f"Spectrum length changed across repeated computation: {len(left)} vs {len(right)}")
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
        if not math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError(
                f"Spectrum determinism check failed at {index}: {left_value} vs {right_value}"
            )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "support_name",
        "support_type",
        "task",
        "token_count",
        "hidden_dim",
        "trace",
        "effective_rank",
        "hoyer_concentration",
        "min_eigenvalue",
        "max_eigenvalue",
        "spectrum_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Readout Geometry Summary",
        "",
        "| Support | Task | Tokens | Hidden | Trace | Eff. Rank | Hoyer | Max Eig. |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['support_name']}` | `{row['task']}` | {row['token_count']} | "
            f"{row['hidden_dim']} | {row['trace']:.6f} | {row['effective_rank']:.6f} | "
            f"{row['hoyer_concentration']:.6f} | {row['max_eigenvalue']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("support,task,tokens,hidden,trace,effective_rank,hoyer,max_eigenvalue")
    for row in rows:
        print(
            f"{row['support_name']},{row['task']},{row['token_count']},{row['hidden_dim']},"
            f"{row['trace']:.6f},{row['effective_rank']:.6f},"
            f"{row['hoyer_concentration']:.6f},{row['max_eigenvalue']:.6f}"
        )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
