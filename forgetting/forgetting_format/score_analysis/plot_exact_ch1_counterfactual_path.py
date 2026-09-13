"""Plot the four exact-CH1 GSM Q/A-to-P/R counterfactual scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "score_analysis_qwen25_base_mmlu5000_exact_ch1_component_attribution_answer_first_n50_seed42"
    / "summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", default=str(DEFAULT_INPUT))
    parser.add_argument("--output_png", default=None)
    parser.add_argument("--output_pdf", default=None)
    return parser.parse_args()


def output_path(input_json: Path, requested: str | None, suffix: str) -> Path:
    return (
        Path(requested)
        if requested
        else input_json.parent / f"exact_ch1_counterfactual_path.{suffix}"
    )


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json)
    values = json.loads(input_json.read_text(encoding="utf-8"))["conditions"]["qa"]["bootstrap"]

    # All four terms are exact vocabulary-aligned CH1 scores, with MMLU fixed
    # to Q/A and restricted to its gold answer-prediction position.
    entries = (
        ("score_qa", "Original\nQ/A", "#7F7F7F"),
        ("error_only", "Error → P/R\nHidden = Q/A", "#54A24B"),
        ("hidden_only", "Hidden → P/R\nError = Q/A", "#F58518"),
        ("score_pr", "Original\nP/R", "#4C78A8"),
    )
    means = np.array([values[key]["mean"] for key, _, _ in entries])
    lows = np.array([values[key]["mean"] for key, _, _ in entries]) + 1
    highs = np.array([values[key]["mean"] for key, _, _ in entries])
    errors = np.vstack((means - lows, highs - means))
    positions = np.arange(len(entries))

    fig, axis = plt.subplots(figsize=(5, 5))
    axis.bar(
        positions,
        means,
        color=[color for _, _, color in entries],
        width=0.5,
        edgecolor="white",
        linewidth=1.25,
        zorder=3,
    )
    # axis.errorbar(
    #     positions,
    #     means,
    #     yerr=errors,
    #     fmt="none",
    #     color="#202020",
    #     capsize=5,
    #     linewidth=1.5,
    #     zorder=4,
    # )
    axis.axhline(0, color="#3F3F3F", linewidth=1.05, zorder=2)
    axis.grid(axis="y", linestyle=":", alpha=0.48, zorder=0)
    axis.set_xticks(positions, [label for _, label, _ in entries])
    axis.set_ylabel("Exact CH1 score")
    axis.set_title(
        "Exact CH1 counterfactual path at the MMLU Q/A answer token",
        fontsize=10,
    )
    lower, upper = lows.min(), highs.max()
    margin = max(abs(lower), abs(upper)) * 0.5
    axis.set_ylim(lower - margin, upper + margin)
    for position, mean in zip(positions, means, strict=True):
        offset = max((upper - lower) * 0.025, 3.0)
        axis.text(
            position,
            mean + (offset if mean >= 0 else -offset),
            f"{mean:.2f}",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=10.5,
            weight="bold",
        )
    # fig.text(
    #     0.5,
    #     0.01,
    #     "MMLU uses only the teacher-forced gold answer token; GSM8K uses all positions. "
    #     "Error bars: 95% cluster-bootstrap CI. Higher / less-negative CH1 predicts a more protective update.",
    #     ha="center",
    #     fontsize=9.2,
    # )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for target in (
        output_path(input_json, args.output_png, "png"),
        # output_path(input_json, args.output_pdf, "pdf"),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=220, bbox_inches="tight")
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
