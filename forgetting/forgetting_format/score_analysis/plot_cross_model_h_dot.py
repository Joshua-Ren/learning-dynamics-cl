"""Overlay the CH2 layerwise raw h-dot template effect for three models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_OUTPUT = ARTIFACTS / "score_analysis_cross_model_ch2_n50_seed42/per_layer_h_dot_delta_across_models"
MODEL_CSVS = {
    "Qwen2.5-1.5B": ARTIFACTS
    / "score_analysis_qwen25_base_mmlu5000_mmlu_answer_first_per_layer_input_rmsnorm_n50_seed42"
    / "primary_delta_by_layer.csv",
    "Llama-3.2-3B": ARTIFACTS
    / "score_analysis_llama32_3b_base_mmlu5000_mmlu_answer_first_per_layer_input_rmsnorm_n50_seed42"
    / "primary_delta_by_layer.csv",
    "Qwen3-4B": ARTIFACTS
    / "score_analysis_qwen3_4b_base_mmlu5000_mmlu_answer_first_per_layer_input_rmsnorm_n50_seed42"
    / "primary_delta_by_layer.csv",
}
COLORS = {"Qwen2.5-1.5B": "#0072B2", "Llama-3.2-3B": "#D55E00", "Qwen3-4B": "#009E73"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cross-model layerwise h-dot deltas.")
    parser.add_argument("--output_prefix", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dpi", type=int, default=350)
    return parser.parse_args()


def read_gh_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["readout"] == "GH hidden"]
    if not rows:
        raise ValueError(f"No GH hidden rows in {path}")
    return tuple(
        np.asarray([float(row[key]) for row in rows])
        for key in ("layer", "h_dot_delta", "h_dot_ci95_low", "h_dot_ci95_high")
    )


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(6.25, 4.25), constrained_layout=True)
    max_layer = 0
    for name, csv_path in MODEL_CSVS.items():
        layers, delta, lower, upper = read_gh_rows(csv_path)
        max_layer = max(max_layer, int(layers.max()))
        color = COLORS[name]
        axis.fill_between(layers, lower, upper, color=color, alpha=0.14, linewidth=0, zorder=2)
        axis.plot(
            layers,
            delta,
            label=name,
            color=color,
            linewidth=2.15,
            marker="o",
            markersize=3.4,
            zorder=3,
        )

    axis.axhline(0.0, color="0.2", linewidth=1.0, linestyle="--", zorder=1)
    axis.set_xlim(0.5, max_layer + 0.5)
    ticks = list(range(1, max_layer + 1, 5))
    if ticks[-1] != max_layer:
        ticks.append(max_layer)
    axis.set_xticks(ticks)
    axis.set_yscale("symlog", linthresh=100.0, linscale=0.8, base=10)
    axis.set_xlabel("CH2 layer (input RMSNorm)")
    axis.set_ylabel(r"$\Delta$ h dot product (symmetric log scale)")
    axis.set_title("Q/A -> P/R reduces CH2 hidden-state similarity across models", pad=9)
    axis.grid(axis="y", color="0.88", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper left", handlelength=2.4)
    # figure.text(
    #     0.5,
    #     -0.045,
    #     r"$\Delta = \mathrm{sim}$(MMLU Q/A, GSM Q/A) $-\ \mathrm{sim}$(MMLU Q/A, GSM P/R). "
    #     "Positive values indicate lower similarity under the P/R format; shaded bands are 95% bootstrap CIs.",
    #     ha="center",
    #     va="top",
    #     fontsize=8.8,
    # )
    for suffix, options in (
        (".pdf", {"bbox_inches": "tight"}),
        (".png", {"dpi": args.dpi, "bbox_inches": "tight"}),
    ):
        figure.savefig(output_prefix.with_suffix(suffix), **options)
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
