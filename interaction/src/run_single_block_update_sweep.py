import argparse
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from diagnose_per_layer_ch2 import metrics
from run_section3_validation import (
    build_observe_tokens,
    build_pairs,
    build_update_tokens,
    compute_after_logp,
    configure_precision,
    load_adapter_samples,
    load_model,
    model_dtype_report,
    resolve_torch_dtype,
    select_samples_by_domain,
    token_metadata,
)
from utils.ch1_ch2_metrics import _get_transformer_layers, _model_device
from utils.config_read import load_config
from utils.hh_dataset_adapters import DEFAULT_MMLU_SUBJECTS
from utils.one_step_forgetting import apply_single_token_update, clone_model_state, restore_model_state


COMPARISONS = {
    "A_actual_vs_exact": ("delta_logp_layer", "first_order_exact_layer"),
    "B_exact_vs_approx": ("first_order_exact_layer", "approx_layer"),
    "C_actual_vs_approx": ("delta_logp_layer", "approx_layer"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Single-block restricted one-step update sweep.")
    parser.add_argument("--source_run_config", type=Path, required=True)
    parser.add_argument("--per_layer_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=str, default="interaction/configs/train_basic.yaml")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--representative_layers", type=str, default="0,12,-1")
    return parser.parse_args()


def load_source_rows(source_run_config):
    csv_path = source_run_config.parent / "section3_pairs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing source Section 3 pair CSV: {csv_path}")
    return pd.read_csv(csv_path)


def finite_metrics(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    mask = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
    return metrics(x[mask], y[mask]) if mask.any() else {
        "n": 0,
        "pearson": None,
        "spearman_signed": None,
        "spearman_abs": None,
        "sign_agreement": None,
        "slope_y_on_x_through_origin": None,
        "mae": None,
        "mean_abs_exact": None,
        "mean_abs_approx": None,
    }


def parse_representative_layers(spec, num_layers):
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx = num_layers + idx
        if 0 <= idx < num_layers and idx not in out:
            out.append(idx)
    return out


def cast_model_to_requested_dtype(model, torch_dtype):
    requested = resolve_torch_dtype(torch_dtype)
    if requested == "auto":
        return model
    model.to(dtype=requested)
    bad = []
    for name, param in model.named_parameters():
        if param.is_floating_point() and param.dtype != requested:
            bad.append((name, str(param.dtype)))
            if len(bad) >= 5:
                break
    if bad:
        details = ", ".join(f"{name}:{dtype}" for name, dtype in bad)
        raise RuntimeError(f"Model precision cast failed for requested {requested}: {details}")
    return model


def rebuild_pairs(source, config, local_files_only):
    model_name = source["model_name_or_path"]
    max_length = config["model"]["max_length"]
    seed = source.get("seed", 42)
    pair_seed = source.get("pair_seed", seed)
    update_samples, update_adapter = load_adapter_samples(
        source["update_dataset"],
        source["update_split"],
        model_name,
        max_length,
        config,
        local_files_only,
        source.get("mmlu_subjects") or DEFAULT_MMLU_SUBJECTS,
    )
    observe_samples_all, _ = load_adapter_samples(
        source["observe_dataset"],
        source["observe_split"],
        model_name,
        max_length,
        config,
        local_files_only,
        source.get("mmlu_subjects") or DEFAULT_MMLU_SUBJECTS,
    )
    observe_samples = select_samples_by_domain(
        observe_samples_all,
        source["max_observe_samples_per_domain"],
        seed,
        source.get("max_observe_samples"),
    )
    update_tokens = build_update_tokens(
        update_samples,
        source["max_update_samples"],
        source["max_update_tokens_per_sample"],
    )
    observe_tokens = build_observe_tokens(observe_samples, source["max_observe_tokens_per_sample"])
    pairs = build_pairs(update_tokens, observe_tokens, source.get("max_pairs"), pair_seed)
    return update_tokens, observe_tokens, pairs, update_adapter.tokenizer


def run_block_updates(model, tokenizer, source_df, per_layer_df, update_tokens, observe_tokens, pairs, lr):
    layers = _get_transformer_layers(model)
    if layers is None:
        raise RuntimeError("Could not find transformer layers.")
    num_layers = len(layers)
    base_state = clone_model_state(model)
    device = _model_device(model)

    per_layer_lookup = per_layer_df.set_index(["pair_index", "layer"])
    rows = []
    pairs_by_update = {}
    for pair_index, (update_token_index, observe_token_index) in enumerate(pairs):
        pairs_by_update.setdefault(int(update_token_index), []).append((pair_index, int(observe_token_index)))
    try:
        for layer_idx, layer in enumerate(layers):
            block_params = [p for p in layer.parameters(recurse=True) if p.requires_grad]
            if not block_params:
                raise RuntimeError(f"Layer {layer_idx} has no trainable parameters.")
            optimizer = torch.optim.SGD(block_params, lr=lr)
            print(f"Layer {layer_idx}: {len(block_params)} tensors")

            for update_token_index in sorted(pairs_by_update):
                restore_model_state(model, base_state)
                model.zero_grad(set_to_none=True)
                update_sample_index, update_sample, update_pos = update_tokens[int(update_token_index)]

                pending = []
                for pair_index, observe_token_index in pairs_by_update[update_token_index]:
                    observe_sample_index, observe_sample, observe_pos = observe_tokens[int(observe_token_index)]
                    before = compute_after_logp(model, observe_sample, observe_pos)
                    layer_vals = per_layer_lookup.loc[(pair_index, layer_idx)]
                    exact_raw = float(layer_vals["exact_ch2_layer"])
                    approx_raw = float(layer_vals["approx_ch2_layer"])
                    pending.append({
                        "pair_index": pair_index,
                        "observe_token_index": observe_token_index,
                        "observe_sample_index": observe_sample_index,
                        "observe_sample": observe_sample,
                        "observe_pos": observe_pos,
                        "before": before,
                        "exact_raw": exact_raw,
                        "approx_raw": approx_raw,
                    })

                input_ids = update_sample["input_ids"].unsqueeze(0).to(device)
                attention_mask = update_sample["attention_mask"].unsqueeze(0).to(device)
                labels = update_sample["labels"].unsqueeze(0).to(device)
                update_loss = apply_single_token_update(
                    model=model,
                    optimizer=optimizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pos=update_pos,
                )
                for item in pending:
                    observe_sample = item["observe_sample"]
                    observe_pos = item["observe_pos"]
                    pair_index = item["pair_index"]
                    after = compute_after_logp(model, observe_sample, observe_pos)
                    delta = after - item["before"]
                    exact_raw = item["exact_raw"]
                    approx_raw = item["approx_raw"]
                    row = {
                        "pair_index": pair_index,
                        "layer": layer_idx,
                        "lr": lr,
                        "delta_logp_layer": delta,
                        "observe_logp_before": item["before"],
                        "observe_logp_after": after,
                        "update_loss": update_loss,
                        "first_order_exact_layer_raw": exact_raw,
                        "ch2_layer_raw": approx_raw,
                        "first_order_exact_layer": lr * exact_raw,
                        "approx_layer": lr * approx_raw,
                        "source_pair_first_order_exact": float(source_df.loc[pair_index, "first_order_exact"]) if "first_order_exact" in source_df else math.nan,
                        "source_pair_ch2_scaled": float(source_df.loc[pair_index, "ch2_scaled"]) if "ch2_scaled" in source_df else math.nan,
                        **token_metadata("update", update_sample, update_sample_index, update_pos, tokenizer),
                        **token_metadata("observe", observe_sample, item["observe_sample_index"], observe_pos, tokenizer),
                    }
                    rows.append(row)
    finally:
        restore_model_state(model, base_state)
    return pd.DataFrame(rows), num_layers


def summarize_by_layer(rows):
    summary_rows = []
    for layer_idx, sub in rows.groupby("layer"):
        for comparison, (x_col, y_col) in COMPARISONS.items():
            m = finite_metrics(sub[x_col], sub[y_col])
            summary_rows.append({
                "layer": int(layer_idx),
                "comparison": comparison,
                "x": x_col,
                "y": y_col,
                "valid_pairs": int(m.pop("n")),
                **m,
            })
    return pd.DataFrame(summary_rows)


def make_plots(out_dir, rows, summary, representative_layers):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    sub = summary[summary["comparison"] == "C_actual_vs_approx"]
    for col, label in [
        ("pearson", "Pearson"),
        ("spearman_signed", "signed Spearman"),
        ("spearman_abs", "abs Spearman"),
        ("sign_agreement", "sign agreement"),
    ]:
        ax.plot(sub["layer"], sub[col], marker="o", label=label)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("layer index")
    ax.set_ylabel("metric")
    ax.set_title("Single-block update: actual delta_logp vs CH2_l")
    ax.legend()
    p = fig_dir / "single_block_actual_vs_ch2_metrics.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, constrained_layout=True)
    metric_cols = [
        ("pearson", "Pearson"),
        ("spearman_signed", "signed Spearman"),
        ("spearman_abs", "abs Spearman"),
        ("sign_agreement", "sign agreement"),
    ]
    for ax, (col, label) in zip(axes.ravel(), metric_cols):
        for comparison, style in [
            ("A_actual_vs_exact", "--"),
            ("B_exact_vs_approx", ":"),
            ("C_actual_vs_approx", "-"),
        ]:
            ss = summary[summary["comparison"] == comparison]
            ax.plot(ss["layer"], ss[col], marker="o", linestyle=style, label=comparison.replace("_", " "))
        ax.axhline(0, color="gray", linewidth=0.8)
        if col == "sign_agreement":
            ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("layer index")
        ax.set_ylim(-1.05, 1.05)
    axes[0, 0].legend(fontsize=8)
    p = fig_dir / "single_block_comparison_ladder_metrics.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    for layer_idx in representative_layers:
        ss = rows[rows["layer"] == layer_idx]
        fig, ax = plt.subplots(figsize=(4.2, 4), constrained_layout=True)
        x = ss["delta_logp_layer"].to_numpy(dtype=float)
        y = ss["approx_layer"].to_numpy(dtype=float)
        ax.scatter(x, y, s=22, alpha=0.75)
        finite = np.isfinite(x) & np.isfinite(y)
        lim = float(np.nanmax(np.abs(np.concatenate([x[finite], y[finite]])))) if finite.any() else 1.0
        if lim == 0.0:
            lim = 1.0
        lim *= 1.05
        ax.plot([-lim, lim], [-lim, lim], color="black", linewidth=0.9, linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("actual delta_logp from block-only update")
        ax.set_ylabel("lr * CH2_l")
        ax.set_title(f"Layer {layer_idx}: actual vs CH2_l")
        p = fig_dir / f"single_block_scatter_layer_{layer_idx}.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(str(p))
    return paths


def write_report(out_dir, source, summary, plots, lr):
    c = summary[summary["comparison"] == "C_actual_vs_approx"]
    a = summary[summary["comparison"] == "A_actual_vs_exact"]
    b = summary[summary["comparison"] == "B_exact_vs_approx"]
    final_layer = int(c["layer"].max())
    final_c = c[c["layer"] == final_layer].iloc[0]
    all_c = c.sort_values("layer")
    poor_mid = all_c.loc[all_c["sign_agreement"].astype(float).idxmin()]
    paths = "\n".join(str(p) for p in plots)
    summary_csv = out_dir / "single_block_update_summary.csv"
    pair_csv = out_dir / "single_block_update_pairs.csv"

    report = f"""# Single-Block Restricted Update Sweep

Run directory:

```text
{out_dir}
```

Model: `{source['model_name_or_path']}`
Dataset combo: `{source['update_dataset']} -> {source['observe_dataset']}`
Learning rate: `{lr}`
Pairs per layer: `{int(c['valid_pairs'].median())}`

## Outputs

```text
{pair_csv}
{summary_csv}
{paths}
```

## Comparisons

`A_actual_vs_exact`: actual block-only `delta_logp_layer` vs `first_order_exact_layer_l`
`B_exact_vs_approx`: `first_order_exact_layer_l` vs `lr * ch2_layer_l`
`C_actual_vs_approx`: actual block-only `delta_logp_layer` vs `lr * ch2_layer_l`

## Key Result

Final layer C:

```text
Pearson:          {final_c['pearson']:.6g}
signed Spearman: {final_c['spearman_signed']:.6g}
abs Spearman:    {final_c['spearman_abs']:.6g}
sign agreement:  {final_c['sign_agreement']:.6g}
```

Worst sign-agreement layer for C:

```text
layer:            {int(poor_mid['layer'])}
Pearson:          {poor_mid['pearson']:.6g}
signed Spearman: {poor_mid['spearman_signed']:.6g}
abs Spearman:    {poor_mid['spearman_abs']:.6g}
sign agreement:  {poor_mid['sign_agreement']:.6g}
```

## Layer-Wise Metrics

See `{summary_csv}` for all A/B/C metrics by layer.

## Interpretation Checklist

1. If A is strong but B/C degrade in middle layers, the failure is structural CH2_l approximation error rather than finite-step Taylor error.
2. If A degrades, lower the learning rate before interpreting CH2_l.
3. If C is strong only for the top block and poor in middle/lower blocks, the restricted-update behavior matches the existing per-layer first-order diagnostic.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    configure_precision()
    source = json.loads(args.source_run_config.read_text(encoding="utf-8"))
    config = load_config(args.config)
    local_files_only = not args.allow_download
    lr = args.lr if args.lr is not None else float(source.get("lr", 1e-4))
    torch_dtype = args.torch_dtype or source.get("torch_dtype", "float32")
    attn_impl = args.attn_implementation if args.attn_implementation is not None else source.get("attn_implementation", "eager")
    device = args.device or source.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device_map = args.device_map if args.device_map is not None else source.get("device_map")

    seed = source.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    source_df = load_source_rows(args.source_run_config)
    per_layer_df = pd.read_csv(args.per_layer_csv)
    update_tokens, observe_tokens, pairs, tokenizer = rebuild_pairs(source, config, local_files_only)
    if len(pairs) != int(source_df.shape[0]):
        raise RuntimeError(f"Rebuilt {len(pairs)} pairs, but source CSV has {source_df.shape[0]} rows.")

    load_torch_dtype = "auto" if torch_dtype in ("float32", "float64") else torch_dtype
    model = load_model(
        source["model_name_or_path"],
        local_files_only=local_files_only,
        device=device,
        device_map=device_map,
        torch_dtype=load_torch_dtype,
        attn_implementation=attn_impl,
    )
    model = cast_model_to_requested_dtype(model, torch_dtype)
    rows, num_layers = run_block_updates(model, tokenizer, source_df, per_layer_df, update_tokens, observe_tokens, pairs, lr)
    summary = summarize_by_layer(rows)
    representative_layers = parse_representative_layers(args.representative_layers, num_layers)
    if num_layers - 1 not in representative_layers:
        representative_layers.append(num_layers - 1)
    plots = make_plots(args.output_dir, rows, summary, representative_layers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "single_block_update_pairs.csv", index=False)
    summary.to_csv(args.output_dir / "single_block_update_summary.csv", index=False)
    run_config = {
        "source_run_config": str(args.source_run_config),
        "per_layer_csv": str(args.per_layer_csv),
        "model_name_or_path": source["model_name_or_path"],
        "lr": lr,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_impl,
        "num_layers": num_layers,
        "num_pairs": len(pairs),
        "dtype_report": model_dtype_report(model),
        "plots": plots,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "single_block_update_summary.json").write_text(
        json.dumps({
            "run_config": run_config,
            "summary": summary.to_dict(orient="records"),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.output_dir, source, summary, plots, lr)
    print(json.dumps({"output_dir": str(args.output_dir), "num_layers": num_layers, "num_pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
