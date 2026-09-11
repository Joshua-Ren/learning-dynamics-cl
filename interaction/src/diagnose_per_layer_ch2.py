
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from run_section3_validation import (
    build_observe_tokens,
    build_pairs,
    build_update_tokens,
    configure_precision,
    load_adapter_samples,
    load_model,
    model_dtype_report,
    select_samples_by_domain,
)
from utils.ch1_ch2_metrics import (
    compute_readout_align_from_factors,
    extract_ch_factors,
    extract_normalized_block_inputs,
    _as_1d_tensor,
    _get_transformer_layers,
    _model_device,
    token_logprob,
)
from utils.config_read import load_config
from utils.hh_dataset_adapters import DEFAULT_MMLU_SUBJECTS


def sign_agreement(x, y, eps=0.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > eps) & (np.abs(y) > eps)
    if not mask.any():
        return None
    return float((np.sign(x[mask]) == np.sign(y[mask])).mean())


def spearman(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    return float(x.rank(method="average").corr(y.rank(method="average"), method="pearson"))


def through_origin_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return None
    denom = float(np.dot(x[mask], x[mask]))
    if denom == 0.0:
        return None
    return float(np.dot(x[mask], y[mask]) / denom)


def metrics(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    err = y - x
    return {
        "n": int(len(x)),
        "pearson": float(x.corr(y, method="pearson")),
        "spearman_signed": spearman(x, y),
        "spearman_abs": spearman(x.abs(), y.abs()),
        "sign_agreement": sign_agreement(x, y),
        "slope_y_on_x_through_origin": through_origin_slope(x, y),
        "mae": float(err.abs().mean()),
        "mean_abs_exact": float(x.abs().mean()),
        "mean_abs_approx": float(y.abs().mean()),
    }


def get_final_norm_params(model):
    candidates = [getattr(model, "model", None), getattr(model, "transformer", None), model]
    for container in candidates:
        if container is None:
            continue
        for name in ("norm", "ln_f", "final_layernorm", "final_norm"):
            mod = getattr(container, name, None)
            if mod is not None:
                return list(mod.parameters()), name
    return [], None


def layer_param_groups(model):
    layers = _get_transformer_layers(model)
    if layers is None:
        raise RuntimeError("Could not find transformer layers.")
    params = []
    param_layers = []
    for layer_idx, layer in enumerate(layers):
        for p in layer.parameters(recurse=True):
            if p.requires_grad:
                params.append(p)
                param_layers.append(layer_idx)
    return layers, params, param_layers


def exact_layer_interactions(model, obs_sample, obs_pos, update_sample, update_pos, params, param_layers, num_layers):
    model.eval()
    model.zero_grad(set_to_none=True)
    device = _model_device(model)
    obs_target = _as_1d_tensor(obs_sample["labels"], device, "labels")[int(obs_pos)]
    update_target = _as_1d_tensor(update_sample["labels"], device, "labels")[int(update_pos)]
    logp_o = token_logprob(model, obs_sample, int(obs_pos) - 1, obs_target, device=device)
    logp_u = token_logprob(model, update_sample, int(update_pos) - 1, update_target, device=device)
    grads_o = torch.autograd.grad(logp_o, params, retain_graph=True, create_graph=False, allow_unused=True)
    grads_u = torch.autograd.grad(logp_u, params, retain_graph=False, create_graph=False, allow_unused=True)
    vals = [0.0 for _ in range(num_layers)]
    for go, gu, layer_idx in zip(grads_o, grads_u, param_layers):
        if go is None or gu is None:
            continue
        vals[layer_idx] += float(torch.sum(go * gu).detach().cpu().item())
    model.zero_grad(set_to_none=True)
    return vals


def exact_param_interaction(model, obs_sample, obs_pos, update_sample, update_pos, params):
    if not params:
        return 0.0
    model.eval()
    model.zero_grad(set_to_none=True)
    device = _model_device(model)
    obs_target = _as_1d_tensor(obs_sample["labels"], device, "labels")[int(obs_pos)]
    update_target = _as_1d_tensor(update_sample["labels"], device, "labels")[int(update_pos)]
    logp_o = token_logprob(model, obs_sample, int(obs_pos) - 1, obs_target, device=device)
    logp_u = token_logprob(model, update_sample, int(update_pos) - 1, update_target, device=device)
    grads_o = torch.autograd.grad(logp_o, params, retain_graph=True, create_graph=False, allow_unused=True)
    grads_u = torch.autograd.grad(logp_u, params, retain_graph=False, create_graph=False, allow_unused=True)
    val = 0.0
    for go, gu in zip(grads_o, grads_u):
        if go is not None and gu is not None:
            val += float(torch.sum(go * gu).detach().cpu().item())
    model.zero_grad(set_to_none=True)
    return val


def approx_layer_interactions(model, obs_sample, obs_pos, update_sample, update_pos, num_layers):
    obs_factors = extract_ch_factors(model, obs_sample, layer=-1, label_positions=[obs_pos])
    update_factors = extract_ch_factors(model, update_sample, layer=-1, label_positions=[update_pos])
    readout_align = float(compute_readout_align_from_factors(model, obs_factors, update_factors)[0, 0].item())
    obs_inputs = extract_normalized_block_inputs(model, obs_sample, label_positions=[obs_pos], layer=-1)
    update_inputs = extract_normalized_block_inputs(model, update_sample, label_positions=[update_pos], layer=-1)
    if obs_inputs.shape[0] not in (num_layers, 2 * num_layers):
        raise RuntimeError(f"Unexpected number of normalized streams: {obs_inputs.shape[0]} for {num_layers} layers")
    vals = []
    for layer_idx in range(num_layers):
        if obs_inputs.shape[0] == 2 * num_layers:
            stream_ids = [2 * layer_idx, 2 * layer_idx + 1]
        else:
            stream_ids = [layer_idx]
        b = 0.0
        for sid in stream_ids:
            b += float(torch.dot(obs_inputs[sid, 0].float(), update_inputs[sid, 0].float()).item())
        vals.append(readout_align * b)
    return vals, readout_align


def load_source_rows(path):
    csv_path = path.parent / "section3_pairs.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def make_plots(out_dir, layer_summary, cumulative_summary):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    layers = [r["layer"] for r in layer_summary]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(layers, [r["pearson"] for r in layer_summary], marker="o", label="Pearson")
    ax.plot(layers, [r["spearman_signed"] for r in layer_summary], marker="o", label="signed Spearman")
    ax.plot(layers, [r["spearman_abs"] for r in layer_summary], marker="o", label="abs Spearman")
    ax.plot(layers, [r["sign_agreement"] for r in layer_summary], marker="o", label="sign agreement")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("layer index")
    ax.set_ylabel("metric")
    ax.set_title("Per-layer exact vs approx CH2")
    ax.legend()
    p = fig_dir / "per_layer_ch2_quality.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    depths = [r["top_k_layers"] for r in cumulative_summary]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(depths, [r["pearson"] for r in cumulative_summary], marker="o", label="Pearson")
    ax.plot(depths, [r["spearman_signed"] for r in cumulative_summary], marker="o", label="signed Spearman")
    ax.plot(depths, [r["spearman_abs"] for r in cumulative_summary], marker="o", label="abs Spearman")
    ax.plot(depths, [r["sign_agreement"] for r in cumulative_summary], marker="o", label="sign agreement")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("number of included top layers")
    ax.set_ylabel("metric")
    ax.set_title("Top-down cumulative CH2 quality")
    ax.legend()
    p = fig_dir / "top_down_cumulative_ch2_quality.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    return paths


def write_report(out_dir, summary):
    def row(r, first):
        return "| " + " | ".join(str(r[c]) if not isinstance(r[c], float) else f"{r[c]:.6g}" for c in first) + " |"
    layer_cols = ["layer", "pearson", "spearman_signed", "spearman_abs", "sign_agreement", "slope_y_on_x_through_origin", "mean_abs_exact", "mean_abs_approx"]
    cum_cols = ["top_k_layers", "lowest_layer", "pearson", "spearman_signed", "spearman_abs", "sign_agreement"]
    layer_rows = "\n".join(row(r, layer_cols) for r in summary["per_layer_summary"])
    cum_rows = "\n".join(row(r, cum_cols) for r in summary["top_down_cumulative_summary"])
    paths = "\n".join(summary["plots"])
    recon = summary["reconstruction"]
    canc = summary["cancellation"]
    report = f"""# Per-Layer CH2 Diagnostic

Run directory:

```text
{out_dir}
```

Model: `{summary['model_name_or_path']}`
Pairs: `{summary['num_pairs']}`
Layers: `{summary['num_layers']}`

## Reconstruction Checks

Exact per-layer sum uses all parameters inside each transformer block. The existing `ch2_exact_backbone` also includes non-embedding/non-readout backbone parameters outside blocks, mainly final norm, so that remainder is reported separately.

```text
max abs(source ch2_exact_backbone - sum layer exact):          {recon['max_abs_source_exact_minus_layer_exact']:.6e}
mean abs(source ch2_exact_backbone - sum layer exact):         {recon['mean_abs_source_exact_minus_layer_exact']:.6e}
max abs(source exact - layer exact - final norm exact):        {recon['max_abs_source_exact_minus_layer_exact_final_norm']:.6e}
mean abs(source exact - layer exact - final norm exact):       {recon['mean_abs_source_exact_minus_layer_exact_final_norm']:.6e}
max abs(source ch2 - sum layer approx):                       {recon['max_abs_source_approx_minus_layer_approx']:.6e}
mean abs(source ch2 - sum layer approx):                      {recon['mean_abs_source_approx_minus_layer_approx']:.6e}
```

## Per-Layer Exact vs Approx CH2

| layer | Pearson | signed Spearman | abs Spearman | sign agreement | slope | mean abs exact | mean abs approx |
|---:|---:|---:|---:|---:|---:|---:|---:|
{layer_rows}

## Top-Down Cumulative Quality

| top k layers | lowest included layer | Pearson | signed Spearman | abs Spearman | sign agreement |
|---:|---:|---:|---:|---:|---:|
{cum_rows}

## Cancellation Diagnostics

Exact cancellation ratio is `abs(sum_l exact_l) / sum_l abs(exact_l)`. Approx cancellation ratio uses the approximate layer contributions.

```text
exact cancellation ratio mean:       {canc['exact_ratio_mean']:.6g}
exact cancellation ratio median:     {canc['exact_ratio_median']:.6g}
approx cancellation ratio mean:      {canc['approx_ratio_mean']:.6g}
approx cancellation ratio median:    {canc['approx_ratio_median']:.6g}
sign-error exact ratio mean:         {canc['sign_error_exact_ratio_mean']:.6g}
sign-correct exact ratio mean:       {canc['sign_correct_exact_ratio_mean']:.6g}
sign-error count:                    {canc['sign_error_count']}
sign-correct count:                  {canc['sign_correct_count']}
```

## Plots

```text
{paths}
```

## Interpretation

The report distinguishes whether failures are layer-local or cumulative. If per-layer signed agreement is poor for many individual layers, this supports lower-layer/local approximation failure. If per-layer agreement is reasonable but cumulative agreement collapses as layers are summed, this supports cross-layer cancellation/calibration failure.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Per-layer exact vs approximate CH2 diagnostic.")
    parser.add_argument("--source_run_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=str, default="interaction/configs/train_basic.yaml")
    parser.add_argument("--allow_download", action="store_true")
    args = parser.parse_args()

    configure_precision()
    source = json.loads(args.source_run_config.read_text(encoding="utf-8"))
    source_df = load_source_rows(args.source_run_config)
    config = load_config(args.config)
    model_name = source["model_name_or_path"]
    max_length = config["model"]["max_length"]
    local_files_only = not args.allow_download
    device = source.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = source.get("seed", 42)
    pair_seed = source.get("pair_seed", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    update_samples, _ = load_adapter_samples(source["update_dataset"], source["update_split"], model_name, max_length, config, local_files_only, source.get("mmlu_subjects") or DEFAULT_MMLU_SUBJECTS)
    observe_samples_all, _ = load_adapter_samples(source["observe_dataset"], source["observe_split"], model_name, max_length, config, local_files_only, source.get("mmlu_subjects") or DEFAULT_MMLU_SUBJECTS)
    observe_samples = select_samples_by_domain(observe_samples_all, source["max_observe_samples_per_domain"], seed, source.get("max_observe_samples"))
    update_tokens = build_update_tokens(update_samples, source["max_update_samples"], source["max_update_tokens_per_sample"])
    observe_tokens = build_observe_tokens(observe_samples, source["max_observe_tokens_per_sample"])
    pairs = build_pairs(update_tokens, observe_tokens, source.get("max_pairs"), pair_seed)

    model = load_model(model_name, local_files_only, device, source.get("device_map"), source.get("torch_dtype", "float32"), source.get("attn_implementation", "eager"))
    layers, params, param_layers = layer_param_groups(model)
    final_norm_params, final_norm_name = get_final_norm_params(model)
    num_layers = len(layers)

    rows = []
    pair_rows = []
    for pair_index, (u_idx, o_idx) in enumerate(pairs):
        _, update_sample, update_pos = update_tokens[int(u_idx)]
        _, observe_sample, observe_pos = observe_tokens[int(o_idx)]
        exact_vals = exact_layer_interactions(model, observe_sample, observe_pos, update_sample, update_pos, params, param_layers, num_layers)
        final_norm_exact = exact_param_interaction(model, observe_sample, observe_pos, update_sample, update_pos, final_norm_params)
        approx_vals, readout_align = approx_layer_interactions(model, observe_sample, observe_pos, update_sample, update_pos, num_layers)
        source_exact = float(source_df.loc[pair_index, "ch2_exact_backbone"]) if source_df is not None else float("nan")
        source_approx = float(source_df.loc[pair_index, "ch2"]) if source_df is not None else float("nan")
        pair_rows.append({
            "pair_index": pair_index,
            "source_ch2_exact_backbone": source_exact,
            "source_ch2": source_approx,
            "sum_exact_layers": float(sum(exact_vals)),
            "final_norm_exact": final_norm_exact,
            "sum_exact_layers_plus_final_norm": float(sum(exact_vals) + final_norm_exact),
            "sum_approx_layers": float(sum(approx_vals)),
            "readout_align": readout_align,
        })
        for layer_idx, (ex, ap) in enumerate(zip(exact_vals, approx_vals)):
            rows.append({
                "pair_index": pair_index,
                "layer": layer_idx,
                "exact_ch2_layer": ex,
                "approx_ch2_layer": ap,
            })

    layer_df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_df.to_csv(out_dir / "per_layer_ch2_pairwise.csv", index=False)
    pair_df.to_csv(out_dir / "per_pair_ch2_layer_sums.csv", index=False)

    layer_summary = []
    for layer_idx, sub in layer_df.groupby("layer"):
        m = metrics(sub["exact_ch2_layer"], sub["approx_ch2_layer"])
        m.update({"layer": int(layer_idx)})
        layer_summary.append(m)
    layer_summary = sorted(layer_summary, key=lambda x: x["layer"])

    pivot_exact = layer_df.pivot(index="pair_index", columns="layer", values="exact_ch2_layer").sort_index(axis=1)
    pivot_approx = layer_df.pivot(index="pair_index", columns="layer", values="approx_ch2_layer").sort_index(axis=1)
    cumulative_summary = []
    for k in range(1, num_layers + 1):
        included = list(range(num_layers - k, num_layers))
        ex = pivot_exact[included].sum(axis=1)
        ap = pivot_approx[included].sum(axis=1)
        m = metrics(ex, ap)
        m.update({"top_k_layers": k, "lowest_layer": num_layers - k})
        cumulative_summary.append(m)

    exact_abs_sum = pivot_exact.abs().sum(axis=1)
    exact_abs_total = pivot_exact.sum(axis=1).abs()
    approx_abs_sum = pivot_approx.abs().sum(axis=1)
    approx_abs_total = pivot_approx.sum(axis=1).abs()
    exact_ratio = exact_abs_total / exact_abs_sum.replace(0, np.nan)
    approx_ratio = approx_abs_total / approx_abs_sum.replace(0, np.nan)
    sign_correct = np.sign(pivot_exact.sum(axis=1).to_numpy()) == np.sign(pivot_approx.sum(axis=1).to_numpy())
    cancellation_df = pd.DataFrame({
        "pair_index": pivot_exact.index,
        "exact_sum_abs_layers": exact_abs_sum,
        "exact_abs_sum_layers": exact_abs_total,
        "exact_cancellation_ratio": exact_ratio,
        "approx_sum_abs_layers": approx_abs_sum,
        "approx_abs_sum_layers": approx_abs_total,
        "approx_cancellation_ratio": approx_ratio,
        "sum_sign_correct": sign_correct,
    })
    cancellation_df.to_csv(out_dir / "ch2_cancellation_by_pair.csv", index=False)

    recon = {
        "max_abs_source_exact_minus_layer_exact": float((pair_df["source_ch2_exact_backbone"] - pair_df["sum_exact_layers"]).abs().max()),
        "mean_abs_source_exact_minus_layer_exact": float((pair_df["source_ch2_exact_backbone"] - pair_df["sum_exact_layers"]).abs().mean()),
        "max_abs_source_exact_minus_layer_exact_final_norm": float((pair_df["source_ch2_exact_backbone"] - pair_df["sum_exact_layers_plus_final_norm"]).abs().max()),
        "mean_abs_source_exact_minus_layer_exact_final_norm": float((pair_df["source_ch2_exact_backbone"] - pair_df["sum_exact_layers_plus_final_norm"]).abs().mean()),
        "max_abs_source_approx_minus_layer_approx": float((pair_df["source_ch2"] - pair_df["sum_approx_layers"]).abs().max()),
        "mean_abs_source_approx_minus_layer_approx": float((pair_df["source_ch2"] - pair_df["sum_approx_layers"]).abs().mean()),
        "final_norm_name": final_norm_name,
    }
    cancellation = {
        "exact_ratio_mean": float(exact_ratio.mean()),
        "exact_ratio_median": float(exact_ratio.median()),
        "approx_ratio_mean": float(approx_ratio.mean()),
        "approx_ratio_median": float(approx_ratio.median()),
        "sign_error_exact_ratio_mean": float(exact_ratio[~sign_correct].mean()) if (~sign_correct).any() else float("nan"),
        "sign_correct_exact_ratio_mean": float(exact_ratio[sign_correct].mean()) if sign_correct.any() else float("nan"),
        "sign_error_count": int((~sign_correct).sum()),
        "sign_correct_count": int(sign_correct.sum()),
    }
    plots = make_plots(out_dir, layer_summary, cumulative_summary)
    summary = {
        "model_name_or_path": model_name,
        "dtype_report": model_dtype_report(model),
        "source_run_config": str(args.source_run_config),
        "num_pairs": len(pairs),
        "num_layers": num_layers,
        "reconstruction": recon,
        "per_layer_summary": layer_summary,
        "top_down_cumulative_summary": cumulative_summary,
        "cancellation": cancellation,
        "plots": plots,
    }
    (out_dir / "per_layer_ch2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, summary)
    print(json.dumps({"output_dir": str(out_dir), "num_pairs": len(pairs), "num_layers": num_layers, "reconstruction": recon}, indent=2))


if __name__ == "__main__":
    main()
