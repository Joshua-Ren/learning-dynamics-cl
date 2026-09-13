"""Attribute template-induced approximate-score changes to h and prediction error.

The input is ``pairwise_matrices.npz`` written by analyze_template_similarity.py.
For each aligned MMLU/GSM pair, let S = H * E, where H is h_dot and E is
error_dot.  Moving from GSM Q/A (0) to GSM P/R (1), the exact symmetric
decomposition is:

    S1 - S0 = (H1 - H0) * (E0 + E1) / 2
            + (E1 - E0) * (H0 + H1) / 2

The two terms are the Shapley/symmetric contributions from H and E.  They are
computed before averaging across pairs, which correctly preserves covariance
between h_dot and error_dot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose pairwise approximate-score deltas into h_dot and error_dot effects."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="An analyze_template_similarity.py output directory containing pairwise_matrices.npz.",
    )
    parser.add_argument(
        "--output_dir", default=None, help="Defaults to <input_dir>/score_change_decomposition."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    return parser.parse_args()


def condition_arrays(payload: np.lib.npyio.NpzFile, mmlu_template: str, gsm_template: str):
    prefix = f"mmlu_{mmlu_template}_x_gsm_{gsm_template}"
    try:
        return payload[f"{prefix}__h_dot"], payload[f"{prefix}__error_dot"]
    except KeyError as exc:
        expected = f"{prefix}__h_dot / {prefix}__error_dot"
        raise KeyError(f"Missing {expected} in NPZ input.") from exc


def decompose(h0: np.ndarray, e0: np.ndarray, h1: np.ndarray, e1: np.ndarray):
    if not (h0.shape == e0.shape == h1.shape == e1.shape):
        raise ValueError("All four paired matrices must have the same shape.")
    score_delta = h1 * e1 - h0 * e0
    h_contribution = (h1 - h0) * (e0 + e1) / 2.0
    error_contribution = (e1 - e0) * (h0 + h1) / 2.0
    residual = score_delta - h_contribution - error_contribution
    if not np.allclose(residual, 0.0, rtol=1e-10, atol=1e-5):
        raise AssertionError(f"Decomposition residual too large: {np.abs(residual).max():.3e}")
    return {
        "score_delta_pr_minus_qa": score_delta,
        "h_dot_contribution": h_contribution,
        "error_dot_contribution": error_contribution,
    }


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def cluster_bootstrap(
    matrices: dict[str, np.ndarray], seed: int, count: int
) -> dict[str, dict[str, float]]:
    results = {name: {"mean": float(matrix.mean())} for name, matrix in matrices.items()}
    if count <= 0:
        return results
    shapes = {matrix.shape for matrix in matrices.values()}
    if len(shapes) != 1:
        raise ValueError("Bootstrap inputs must have a common matrix shape.")
    rows, columns = next(iter(shapes))
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(count, dtype=np.float64) for name in matrices}
    for draw_index in range(count):
        row_indices = rng.integers(0, rows, size=rows)
        column_indices = rng.integers(0, columns, size=columns)
        for name, matrix in matrices.items():
            draws[name][draw_index] = matrix[np.ix_(row_indices, column_indices)].mean()
    for name, values in draws.items():
        results[name]["ci95_low"] = float(np.quantile(values, 0.025))
        results[name]["ci95_high"] = float(np.quantile(values, 0.975))
    return results


def scientific(value: float) -> str:
    return f"{value:.5e}"


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Approximate-score change decomposition",
        "",
        "Direction is GSM P/R minus GSM Q/A. A negative total means GSM P/R lowers the score.",
        "The h and error terms are computed per MMLU–GSM pair before averaging.",
        "",
        "| MMLU template | Quantity | Mean contribution | 95% cluster-bootstrap CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for template, values in results["conditions"].items():
        for quantity in (
            "score_delta_pr_minus_qa",
            "h_dot_contribution",
            "error_dot_contribution",
        ):
            stats = values["bootstrap"][quantity]
            lines.append(
                f"| MMLU {template.upper()} | {quantity} | {scientific(stats['mean'])} | "
                f"[{scientific(stats['ci95_low'])}, {scientific(stats['ci95_high'])}] |"
            )
        lines.append("|  |  |  |  |")
    lines.extend(
        [
            "",
            "Formula: ΔS = ΔS_h + ΔS_e, where "
            "ΔS_h=(H_PR−H_QA)(E_QA+E_PR)/2 and "
            "ΔS_e=(E_PR−E_QA)(H_QA+H_PR)/2.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap_samples must be non-negative.")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "score_change_decomposition"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / "pairwise_matrices.npz"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    all_matrices: dict[str, np.ndarray] = {}
    condition_results: dict[str, Any] = {}
    with np.load(input_path) as payload:
        for mmlu_template in ("qa", "pr"):
            h0, e0 = condition_arrays(payload, mmlu_template, "qa")
            h1, e1 = condition_arrays(payload, mmlu_template, "pr")
            parts = decompose(h0, e0, h1, e1)
            condition_results[mmlu_template] = {
                "summary": {name: summary(values) for name, values in parts.items()},
                "bootstrap": cluster_bootstrap(
                    parts,
                    seed=args.seed + (0 if mmlu_template == "qa" else 1000),
                    count=args.bootstrap_samples,
                ),
            }
            for name, values in parts.items():
                all_matrices[f"mmlu_{mmlu_template}__{name}"] = values

    results = {
        "input_dir": str(input_dir),
        "direction": "GSM P/R minus GSM Q/A",
        "definition": {
            "score_delta_pr_minus_qa": "H_PR * E_PR - H_QA * E_QA",
            "h_dot_contribution": "(H_PR-H_QA) * (E_QA+E_PR) / 2",
            "error_dot_contribution": "(E_PR-E_QA) * (H_QA+H_PR) / 2",
            "identity": "score_delta_pr_minus_qa = h_dot_contribution + error_dot_contribution",
        },
        "conditions": condition_results,
    }
    np.savez_compressed(output_dir / "decomposition_matrices.npz", **all_matrices)
    (output_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output_dir / "report.md", results)
    for template, values in condition_results.items():
        print(f"MMLU {template.upper()} (GSM P/R minus Q/A):")
        for name, stats in values["bootstrap"].items():
            print(
                f"  {name}: {stats['mean']:.6e} "
                f"CI=[{stats['ci95_low']:.6e}, {stats['ci95_high']:.6e}]"
            )
    print(f"Wrote decomposition to {output_dir}")


if __name__ == "__main__":
    main()
