import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Section 3 validation figures.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing section3_pairs.csv and summary.json.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def ensure_output_dir(input_dir, output_dir):
    out_dir = Path(output_dir) if output_dir else Path(input_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def scatter(df, x, y, path, xlabel=None, ylabel=None):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.scatter(df[x], df[y], s=18, alpha=0.7, linewidths=0)
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.axvline(0.0, color="0.75", linewidth=0.8)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def heatmap(df, value_col, path, title):
    pivot = df.pivot_table(
        index="update_label_pos",
        columns="observe_label_pos",
        values=value_col,
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("observe label_pos")
    ax.set_ylabel("update label_pos")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cumulative_mass(df, path):
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for col, label in [("ch1", "CH1"), ("ch2", "CH2")]:
        vals = df[col].abs().sort_values(ascending=False)
        total = vals.sum()
        if total == 0:
            mass = vals * 0.0
        else:
            mass = vals.cumsum() / total
        x = range(1, len(mass) + 1)
        ax.plot(x, mass.values, marker=".", linewidth=1.4, label=label)
    ax.set_xlabel("ranked token pairs")
    ax.set_ylabel("cumulative absolute influence mass")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = ensure_output_dir(input_dir, args.output_dir)
    df = pd.read_csv(input_dir / "section3_pairs.csv")

    plt.rcParams.update(
        {
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
            "font.size": 10,
        }
    )

    scatter(
        df,
        "delta_logp",
        "first_order_exact",
        out_dir / "fig4a_delta_vs_first_order_exact.png",
        "measured delta log p",
        "first-order exact",
    )
    scatter(
        df,
        "first_order_exact",
        "approx",
        out_dir / "fig4b_first_order_exact_vs_approx.png",
        "first-order exact",
        "forward approximation",
    )
    scatter(
        df,
        "delta_logp",
        "approx",
        out_dir / "fig4c_delta_vs_approx.png",
        "measured delta log p",
        "forward approximation",
    )
    heatmap(df, "ch1", out_dir / "fig4d_ch1_heatmap.png", "CH1")
    heatmap(df, "ch2", out_dir / "fig4e_ch2_heatmap.png", "CH2")
    cumulative_mass(df, out_dir / "fig4f_cumulative_abs_mass.png")

    summary_path = input_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        stats = {
            "num_pairs": int(len(df)),
            "hoyer_sparsity_ch1": summary.get("hoyer_sparsity_ch1"),
            "density_ch1": summary.get("density_ch1"),
            "hoyer_sparsity_ch2": summary.get("hoyer_sparsity_ch2"),
            "density_ch2": summary.get("density_ch2"),
        }
        with open(out_dir / "figure_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    print(f"Saved Section 3 figures to {out_dir}")


if __name__ == "__main__":
    main()
