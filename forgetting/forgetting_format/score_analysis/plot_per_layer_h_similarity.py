"""Plot the layerwise h-similarity format-mismatch effect for CH2."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_RESULTS_DIR = ARTIFACTS / "score_analysis_qwen25_base_mmlu5000_mmlu_answer_first_per_layer_input_rmsnorm_n50_seed42"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot layerwise h dot-product and cosine deltas for CH2."
    )
    parser.add_argument(
        "--csv_path", default=str(DEFAULT_RESULTS_DIR / "primary_delta_by_layer.csv")
    )
    parser.add_argument(
        "--output_prefix",
        default=str(DEFAULT_RESULTS_DIR / "per_layer_h_similarity_delta"),
    )
    parser.add_argument("--dpi", type=int, default=350)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # CH2 uses only the normalized intermediate GH layers.  The final RH layer
    # is intentionally excluded because it is not an intermediate GH readout.
    rows = [row for row in rows if row["readout"] == "GH hidden"]
    if not rows:
        raise ValueError(f"No GH-hidden rows found in {path}")
    keys = (
        "layer",
        "h_dot_delta",
        "h_dot_ci95_low",
        "h_dot_ci95_high",
        "h_cosine_delta",
        "h_cosine_ci95_low",
        "h_cosine_ci95_high",
    )
    return {key: np.asarray([float(row[key]) for row in rows]) for key in keys}


def style_axis(axis: plt.Axes, title: str, ylabel: str, last_layer: int) -> None:
    axis.axhline(0.0, color="0.2", linewidth=1.0, linestyle="--", zorder=1)
    axis.set_title(title, fontsize=11, pad=8)
    axis.set_xlabel("GH layer (input RMSNorm)")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0.5, last_layer + 0.5)
    ticks = list(range(1, last_layer + 1, 5))
    if ticks[-1] != last_layer:
        ticks.append(last_layer)
    axis.set_xticks(ticks)
    axis.grid(axis="y", color="0.88", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    data = load_rows(csv_path)
    layers = data["layer"]
    last_layer = int(layers.max())

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
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.55), constrained_layout=True)
    color = "#2166AC"

    for axis, metric, lower, upper, title, ylabel, scale in (
        (
            axes[0],
            data["h_dot_delta"],
            data["h_dot_ci95_low"],
            data["h_dot_ci95_high"],
            "(a) Hidden-state dot product",
            r"$\Delta$ h dot product",
            1.0,
        ),
        (
            axes[1],
            data["h_cosine_delta"],
            data["h_cosine_ci95_low"],
            data["h_cosine_ci95_high"],
            "(b) Hidden-state cosine similarity",
            r"$\Delta$ cosine(h) ($\times 10^{-3}$)",
            1_000.0,
        ),
    ):
        metric = metric * scale
        lower = lower * scale
        upper = upper * scale
        axis.fill_between(layers, lower, upper, color=color, alpha=0.20, linewidth=0, zorder=2)
        axis.plot(layers, metric, color=color, linewidth=2.1, marker="o", markersize=3.6, zorder=3)
        style_axis(axis, title, ylabel, last_layer)

    figure.text(
        0.5,
        -0.045,
        r"$\Delta = \mathrm{sim}$(MMLU Q/A, GSM Q/A) $-\ \mathrm{sim}$(MMLU Q/A, GSM P/R). "
        "Positive values indicate lower similarity under the P/R format.",
        ha="center",
        va="top",
        fontsize=9,
    )
    for suffix, options in (
        (".pdf", {"bbox_inches": "tight"}),
        (".png", {"dpi": args.dpi, "bbox_inches": "tight"}),
    ):
        figure.savefig(output_prefix.with_suffix(suffix), **options)
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
