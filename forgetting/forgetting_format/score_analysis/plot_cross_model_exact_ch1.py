"""Create one cross-model exact-CH1 answer-token counterfactual figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DEFAULT_OUTPUT_DIR = ARTIFACTS / "score_analysis_cross_model_exact_ch1_answer_first_n50_seed42"
MODEL_INPUTS = (
    (
        "Qwen2.5-1.5B",
        ARTIFACTS
        / "score_analysis_qwen25_base_mmlu5000_exact_ch1_component_attribution_answer_first_n50_seed42"
        / "summary.json",
    ),
    (
        "Llama-3.2-3B",
        ARTIFACTS
        / "score_analysis_llama32_3b_base_mmlu5000_exact_ch1_component_attribution_answer_first_n50_seed42"
        / "summary.json",
    ),
    (
        "Qwen3-4B",
        ARTIFACTS
        / "score_analysis_qwen3_4b_base_mmlu5000_exact_ch1_component_attribution_answer_first_n50_seed42"
        / "summary.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--input_jsons",
        nargs=3,
        default=[str(path) for _, path in MODEL_INPUTS],
        metavar=("QWEN25", "LLAMA32", "QWEN3"),
        help="Three summary.json files in Qwen2.5, Llama-3.2, Qwen3 order.",
    )
    return parser.parse_args()


def load_model(name: str, path: Path) -> dict:
    values = json.loads(path.read_text(encoding="utf-8"))["conditions"]["qa"]["bootstrap"]
    return {
        "name": name,
        "source": str(path),
        "score_qa": values["score_qa"]["mean"],
        "error_only": values["error_only"]["mean"],
        "hidden_only": values["hidden_only"]["mean"],
        "score_pr": values["score_pr"]["mean"],
        "total": values["score_delta_pr_minus_qa"]["mean"],
        "hidden": values["hidden_contribution"]["mean"],
        "error": values["prediction_error_contribution"]["mean"],
    }


def main() -> None:
    args = parse_args()
    models = [load_model(name, Path(path)) for (name, _), path in zip(MODEL_INPUTS, args.input_jsons, strict=True)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "baseline": "#7F7F7F",
        "error": "#54A24B",
        "hidden": "#F58518",
        "full": "#4C78A8",
        "total": "#4C78A8",
    }
    fig = plt.figure(figsize=(13.6, 8.2))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.35, 1.0), hspace=0.40, wspace=0.24)
    top_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    bottom_axis = fig.add_subplot(grid[1, :])

    all_deltas = []
    for item in models:
        baseline = item["score_qa"]
        all_deltas.extend(
            [0.0, item["error_only"] - baseline, item["hidden_only"] - baseline, item["score_pr"] - baseline]
        )
    y_padding = 3.5
    y_min, y_max = min(all_deltas) - y_padding, max(all_deltas) + y_padding

    for axis, item in zip(top_axes, models, strict=True):
        baseline = item["score_qa"]
        values = {
            "baseline": 0.0,
            "error": item["error_only"] - baseline,
            "hidden": item["hidden_only"] - baseline,
            "full": item["score_pr"] - baseline,
        }
        raw_scores = {
            "baseline": item["score_qa"],
            "error": item["error_only"],
            "hidden": item["hidden_only"],
            "full": item["score_pr"],
        }
        x = {"baseline": 0.0, "error": 1.0, "hidden": 1.0, "full": 2.0}
        # Two alternative counterfactual paths; neither middle node is a step in the other path.
        axis.plot([x["baseline"], x["error"], x["full"]], [values["baseline"], values["error"], values["full"]], color=colors["error"], linewidth=2.4, zorder=2)
        axis.plot([x["baseline"], x["hidden"], x["full"]], [values["baseline"], values["hidden"], values["full"]], color=colors["hidden"], linewidth=2.4, zorder=2)
        for key in ("baseline", "error", "hidden", "full"):
            axis.scatter(x[key], values[key], color=colors[key], s=82, zorder=4, edgecolor="white", linewidth=1.0)
            offset = 1.15 if key != "hidden" else -1.45
            vertical = "bottom" if key != "hidden" else "top"
            axis.text(
                x[key],
                values[key] + offset,
                f"S={raw_scores[key]:+.1f}",
                ha="center",
                va=vertical,
                fontsize=9.3,
                weight="bold",
            )
        axis.axhline(0, color="#444444", linewidth=0.9, zorder=1)
        axis.set_xlim(-0.28, 2.28)
        axis.set_ylim(y_min, y_max)
        axis.grid(axis="y", linestyle=":", alpha=0.45)
        axis.set_title(item["name"], fontsize=12, weight="bold")
        axis.set_xticks([0, 1, 2], ["Original\nQ/A", "Counterfactual\nP/R component", "Original\nP/R"])
        axis.tick_params(axis="x", labelsize=9)
    top_axes[0].set_ylabel("Δ exact CH1 vs original Q/A")

    model_positions = np.arange(len(models))
    width = 0.22
    contributions = (("total", "Total ΔS", colors["total"]), ("hidden", "Hidden ΔH", colors["hidden"]), ("error", "Prediction error ΔE", colors["error"]))
    for offset, (key, label, color) in zip((-width, 0.0, width), contributions, strict=True):
        values = [item[key] for item in models]
        bars = bottom_axis.bar(model_positions + offset, values, width, label=label, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        for bar, value in zip(bars, values, strict=True):
            bottom_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.75 if value >= 0 else -0.75),
                f"{value:+.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9.4,
            )
    contribution_extent = max(abs(item[key]) for item in models for key, _, _ in contributions)
    bottom_axis.set_ylim(-contribution_extent - 5, contribution_extent + 5)
    bottom_axis.axhline(0, color="#444444", linewidth=0.9, zorder=1)
    bottom_axis.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
    bottom_axis.set_xticks(model_positions, [item["name"] for item in models])
    bottom_axis.set_ylabel("Exact CH1 change: P/R − Q/A")
    bottom_axis.legend(loc="upper right", ncol=3, frameon=False)

    fig.suptitle(
        "Cross-model exact CH1 template effect at the MMLU gold-answer token",
        fontsize=15,
        weight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.012,
        "Top: baseline-centered counterfactual paths; labels show raw exact CH1 scores. "
        "Bottom: exact Shapley attribution, ΔS = ΔH + ΔE. "
        "All 95% cluster-bootstrap intervals include zero; these are point-estimate comparisons.",
        ha="center",
        fontsize=9.7,
    )
    handles = [
        Line2D([0], [0], color=colors["error"], linewidth=2.4, label="Swap error only"),
        Line2D([0], [0], color=colors["hidden"], linewidth=2.4, label="Swap hidden only"),
        Patch(facecolor=colors["full"], label="Full P/R"),
    ]
    top_axes[2].legend(handles=handles, loc="lower right", frameon=True, fontsize=8.5)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.115, top=0.885, hspace=0.42, wspace=0.24)
    figure_paths = (
        output_dir / "cross_model_exact_ch1_counterfactual_attribution.png",
        output_dir / "cross_model_exact_ch1_counterfactual_attribution.pdf",
    )
    for target in figure_paths:
        fig.savefig(target, dpi=220, bbox_inches="tight")
        print(f"Wrote {target}")
    (output_dir / "cross_model_exact_ch1_summary.json").write_text(
        json.dumps({"models": models}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
