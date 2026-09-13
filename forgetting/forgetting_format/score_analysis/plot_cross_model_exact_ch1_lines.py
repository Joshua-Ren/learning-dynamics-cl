"""Plot mean exact-CH1 scores across the four counterfactual states for all models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = ARTIFACTS / "score_analysis_cross_model_exact_ch1_answer_first_n50_seed42"
DEFAULT_INPUTS = (
    ARTIFACTS
    / "score_analysis_qwen25_base_mmlu5000_exact_ch1_component_attribution_answer_first_n50_seed42"
    / "summary.json",
    ARTIFACTS
    / "score_analysis_llama32_3b_base_mmlu5000_exact_ch1_component_attribution_answer_first_n50_seed42"
    / "summary.json",
    ARTIFACTS
    / "score_analysis_qwen3_4b_base_mmlu5000_exact_ch1_component_attribution_answer_first_n50_seed42"
    / "summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--input_jsons",
        nargs=3,
        default=[str(path) for path in DEFAULT_INPUTS],
        metavar=("QWEN25", "LLAMA32", "QWEN3"),
        help="Three exact-CH1 summary.json files in Qwen2.5, Llama-3.2, Qwen3 order.",
    )
    return parser.parse_args()


def means(path: Path) -> list[float]:
    values = json.loads(path.read_text(encoding="utf-8"))["conditions"]["qa"]["bootstrap"]
    # This ordering exactly follows plot_exact_ch1_counterfactual_path.py.
    return [
        abs(values["score_qa"]["mean"]),
        abs(values["error_only"]["mean"]),
        abs(values["hidden_only"]["mean"]),
        abs(values["score_pr"]["mean"]),
    ]


def main() -> None:
    args = parse_args()
    models = (
        ("Qwen2.5-1.5B", "#4C78A8", "o"),
        ("Llama-3.2-3B", "#E45756", "s"),
        ("Qwen3-4B", "#54A24B", "D"),
    )
    x = np.arange(4)
    labels = (
        "Original\nQ/A",
        "Error → P/R\nHidden = Q/A",
        "Hidden → P/R\nError = Q/A",
        "Original\nP/R",
    )
    fig, axis = plt.subplots(figsize=(7.0, 5.4))
    for (name, color, marker), path in zip(models, args.input_jsons, strict=True):
        values = means(Path(path))
        axis.plot(
            x,
            values,
            label=name,
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.35,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=3,
        )
        for x_value, score in zip(x, values, strict=True):
            axis.annotate(
                f"{score:+.1f}",
                (x_value, score),
                xytext=(0, 8 if score >= 0 else -13),
                textcoords="offset points",
                ha="center",
                va="bottom" if score >= 0 else "top",
                color=color,
                fontsize=8.6,
                weight="bold",
            )
    axis.axhline(0, color="#444444", linewidth=1.0, zorder=1)
    axis.grid(axis="y", linestyle=":", alpha=0.48, zorder=0)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Absolute CH1 score (mean)")
    axis.set_title("Absolute CH1 counterfactual path at the MMLU Q/A gold-answer token")
    axis.legend(frameon=False, loc="best")
    # fig.text(
    #     0.5,
    #     0.015,
    #     "MMLU is fixed to the gold answer-token position; GSM8K uses all teacher-forced positions. "
    #     "Means only (no confidence intervals).",
    #     ha="center",
    #     fontsize=9.4,
    # )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_dir / "cross_model_exact_ch1_counterfactual_lines_mean.png",
        output_dir / "cross_model_exact_ch1_counterfactual_lines_mean.pdf",
    )
    for output in outputs:
        fig.savefig(output, dpi=220, bbox_inches="tight")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
