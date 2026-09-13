"""Plot the exact-CH1 H versus prediction-error counterfactual attribution."""

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


def resolve_output(input_json: Path, supplied: str | None, suffix: str) -> Path:
    if supplied:
        return Path(supplied)
    return input_json.parent / f"exact_ch1_h_error_attribution.{suffix}"


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json)
    summary = json.loads(input_json.read_text(encoding="utf-8"))
    quantities = (
        ("score_delta_pr_minus_qa", "Total exact CH1", "#4C78A8"),
        ("hidden_contribution", "Hidden state H", "#F58518"),
        ("prediction_error_contribution", "Prediction error E", "#54A24B"),
    )
    panels = (("qa", "Primary: MMLU Q/A"), ("pr", "Control: MMLU P/R"))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), sharey=True)
    all_limits = []
    for template, _ in panels:
        values = summary["conditions"][template]["bootstrap"]
        for key, _, _ in quantities:
            item = values[key]
            all_limits.extend((item["ci95_low"], item["ci95_high"]))
    padding = max(abs(value) for value in all_limits) * 0.12
    y_min, y_max = min(all_limits) - padding, max(all_limits) + padding

    for axis, (template, title) in zip(axes, panels, strict=True):
        values = summary["conditions"][template]["bootstrap"]
        positions = np.arange(len(quantities))
        means = np.array([values[key]["mean"] for key, _, _ in quantities])
        low = np.array([values[key]["ci95_low"] for key, _, _ in quantities])
        high = np.array([values[key]["ci95_high"] for key, _, _ in quantities])
        errors = np.vstack((means - low, high - means))
        axis.bar(
            positions,
            means,
            color=[color for _, _, color in quantities],
            width=0.66,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        axis.errorbar(
            positions, means, yerr=errors, fmt="none", color="#222222", capsize=4, linewidth=1.4, zorder=4
        )
        axis.axhline(0, color="#444444", linewidth=1.0, zorder=2)
        axis.set_title(title, fontsize=12, weight="bold")
        axis.set_xticks(positions, [label for _, label, _ in quantities], rotation=17, ha="right")
        axis.set_ylim(y_min, y_max)
        axis.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
        for x, mean in zip(positions, means, strict=True):
            offset = max(abs(y_max - y_min) * 0.025, 2.0)
            axis.text(
                x,
                mean + (offset if mean >= 0 else -offset),
                f"{mean:+.1f}",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=10,
            )
    axes[0].set_ylabel("Exact CH1 change: GSM P/R − GSM Q/A")
    fig.suptitle(
        "Exact CH1 attribution at the MMLU gold-answer token\n"
        "Positive = less predicted MMLU answer-likelihood decrease; error bars = 95% cluster-bootstrap CI",
        fontsize=13,
        y=1.03,
    )
    fig.text(
        0.5,
        -0.05,
        "Exact Shapley attribution: total change = hidden-state contribution + prediction-error contribution. "
        "MMLU uses one answer position; GSM8K uses all positions.",
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout()
    for output in (resolve_output(input_json, args.output_png, "png"), resolve_output(input_json, args.output_pdf, "pdf")):
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
