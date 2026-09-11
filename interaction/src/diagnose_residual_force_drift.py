
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
import torch.nn.functional as F

from run_section3_validation import (
    configure_precision,
    load_adapter_samples,
    load_model,
    resolve_model_name,
    build_update_tokens,
    build_observe_tokens,
    build_pairs,
    select_samples_by_domain,
    model_dtype_report,
)
from utils.ch1_ch2_metrics import (
    _as_1d_tensor,
    _block_input_norm,
    _get_transformer_layers,
    _model_device,
    _model_compute_dtype,
    get_readout_weight,
)
from utils.config_read import load_config
from utils.hh_dataset_adapters import DEFAULT_MMLU_SUBJECTS, parse_csv_arg


def corr(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    return float(x.corr(y, method="pearson"))


def spearman(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    return float(x.rank(method="average").corr(y.rank(method="average"), method="pearson"))


def sign_agreement(x, y, eps=0.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > eps) & (np.abs(y) > eps)
    if not mask.any():
        return None
    return float((np.sign(x[mask]) == np.sign(y[mask])).mean())


def cos(a, b, eps=1e-30):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= eps:
        return np.nan
    return float(np.dot(a, b) / denom)


def pair_metrics(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return {
        "n": int(len(x)),
        "pearson": corr(x, y),
        "spearman_signed": spearman(x, y),
        "spearman_abs": spearman(np.abs(x), np.abs(y)),
        "sign_agreement": sign_agreement(x, y),
    }


def get_final_norm(model):
    candidates = [getattr(model, "model", None), getattr(model, "transformer", None), model]
    for container in candidates:
        if container is None:
            continue
        for name in ("norm", "ln_f", "final_layernorm", "final_norm"):
            mod = getattr(container, name, None)
            if mod is not None:
                return mod, name
    return None, None


def token_forward_backward(model, sample, label_pos):
    model.eval()
    model.zero_grad(set_to_none=True)
    device = _model_device(model)
    compute_dtype = _model_compute_dtype(model)
    label_pos = int(label_pos)
    logit_pos = label_pos - 1

    input_ids = _as_1d_tensor(sample["input_ids"], device, "input_ids").unsqueeze(0)
    labels = _as_1d_tensor(sample["labels"], device, "labels")
    attention_mask = sample.get("attention_mask")
    if attention_mask is not None:
        attention_mask = _as_1d_tensor(attention_mask, device, "attention_mask").unsqueeze(0)
    target_id = int(labels[label_pos].item())

    layers = _get_transformer_layers(model)
    if layers is None:
        raise RuntimeError("Could not find transformer layers for hook extraction.")

    norm_outputs = {}
    final_norm_output = {}
    handles = []

    def make_layer_hook(layer_idx):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            tensor.retain_grad()
            norm_outputs[layer_idx] = tensor
        return hook

    for layer_idx in range(len(layers)):
        norm = _block_input_norm(model, layer_idx)
        if norm is None:
            raise RuntimeError(f"Layer {layer_idx} has no recognized input RMSNorm.")
        handles.append(norm.register_forward_hook(make_layer_hook(layer_idx)))

    final_norm, final_norm_name = get_final_norm(model)
    if final_norm is not None:
        def final_hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            tensor.retain_grad()
            final_norm_output["tensor"] = tensor
        handles.append(final_norm.register_forward_hook(final_hook))

    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        )
        logits = outputs.logits[0, logit_pos].to(dtype=compute_dtype)
        probs = F.softmax(logits, dim=-1)
        g = -probs.detach().clone()
        g[target_id] += 1.0
        logp = F.log_softmax(logits, dim=-1)[target_id]
        logp.backward()

        readout_weight = get_readout_weight(model).detach().to(device=logits.device, dtype=compute_dtype)
        q = (g.to(device=logits.device, dtype=compute_dtype) @ readout_weight).detach().cpu()

        deltas = []
        streams = []
        for layer_idx in range(len(layers)):
            tensor = norm_outputs[layer_idx]
            grad = tensor.grad
            if grad is None:
                raise RuntimeError(f"Missing grad for normalized input layer {layer_idx}.")
            deltas.append(grad[0, logit_pos].detach().to(dtype=torch.float32, device="cpu"))
            streams.append(tensor[0, logit_pos].detach().to(dtype=torch.float32, device="cpu"))

        final_cos = None
        final_norm_ratio = None
        if "tensor" in final_norm_output:
            f = final_norm_output["tensor"]
            if f.grad is not None:
                final_grad = f.grad[0, logit_pos].detach().to(dtype=torch.float32, device="cpu")
                q_cpu = q.to(dtype=torch.float32)
                final_cos = cos(final_grad.numpy(), q_cpu.numpy())
                q_norm = float(q_cpu.norm().item())
                final_norm_ratio = float(final_grad.norm().item() / q_norm) if q_norm else None

        return {
            "q": q.to(dtype=torch.float32),
            "delta_layers": torch.stack(deltas, dim=0),
            "stream_layers": torch.stack(streams, dim=0),
            "target_id": target_id,
            "logp": float(logp.detach().cpu().item()),
            "final_norm_name": final_norm_name,
            "final_norm_grad_cos_q": final_cos,
            "final_norm_grad_norm_ratio_q": final_norm_ratio,
        }
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def build_force_records(model, update_tokens, observe_tokens, pairs):
    needed_update = sorted({int(u) for u, _ in pairs})
    needed_observe = sorted({int(o) for _, o in pairs})
    records = {"update": {}, "observe": {}}
    for idx in needed_update:
        sample_idx, sample, label_pos = update_tokens[idx]
        records["update"][idx] = token_forward_backward(model, sample, label_pos)
        records["update"][idx]["sample_index"] = int(sample_idx)
        records["update"][idx]["label_pos"] = int(label_pos)
    for idx in needed_observe:
        sample_idx, sample, label_pos = observe_tokens[idx]
        records["observe"][idx] = token_forward_backward(model, sample, label_pos)
        records["observe"][idx]["sample_index"] = int(sample_idx)
        records["observe"][idx]["label_pos"] = int(label_pos)
    return records


def summarize_single_token(records):
    all_records = list(records["update"].values()) + list(records["observe"].values())
    n_layers = int(all_records[0]["delta_layers"].shape[0])
    rows = []
    q_final_cos = []
    q_final_ratio = []
    for rec in all_records:
        q = rec["q"].numpy()
        q_norm = float(np.linalg.norm(q))
        if rec.get("final_norm_grad_cos_q") is not None:
            q_final_cos.append(rec["final_norm_grad_cos_q"])
        if rec.get("final_norm_grad_norm_ratio_q") is not None:
            q_final_ratio.append(rec["final_norm_grad_norm_ratio_q"])
        for layer in range(n_layers):
            d = rec["delta_layers"][layer].numpy()
            rows.append({
                "layer": layer,
                "cos_delta_q": cos(d, q),
                "norm_ratio_delta_q": float(np.linalg.norm(d) / q_norm) if q_norm else np.nan,
            })
    df = pd.DataFrame(rows)
    by_layer = []
    for layer, sub in df.groupby("layer"):
        by_layer.append({
            "layer": int(layer),
            "mean_cos_delta_q": float(sub["cos_delta_q"].mean()),
            "median_cos_delta_q": float(sub["cos_delta_q"].median()),
            "mean_norm_ratio_delta_q": float(sub["norm_ratio_delta_q"].mean()),
            "median_norm_ratio_delta_q": float(sub["norm_ratio_delta_q"].median()),
        })
    return {
        "by_layer": by_layer,
        "final_norm_grad_vs_q": {
            "mean_cos": float(np.nanmean(q_final_cos)) if q_final_cos else None,
            "median_cos": float(np.nanmedian(q_final_cos)) if q_final_cos else None,
            "mean_norm_ratio": float(np.nanmean(q_final_ratio)) if q_final_ratio else None,
            "median_norm_ratio": float(np.nanmedian(q_final_ratio)) if q_final_ratio else None,
        },
    }


def compute_pairwise(records, pairs):
    n_layers = int(next(iter(records["update"].values()))["delta_layers"].shape[0])
    pair_rows = []
    for pair_index, (u_idx, o_idx) in enumerate(pairs):
        u = records["update"][int(u_idx)]
        o = records["observe"][int(o_idx)]
        q_u = u["q"]
        q_o = o["q"]
        a_readout = float(torch.dot(q_o, q_u).item())
        for layer in range(n_layers):
            d_u = u["delta_layers"][layer]
            d_o = o["delta_layers"][layer]
            h_u = u["stream_layers"][layer]
            h_o = o["stream_layers"][layer]
            b_l = float(torch.dot(h_o, h_u).item())
            a_exact = float(torch.dot(d_o, d_u).item())
            pair_rows.append({
                "pair_index": int(pair_index),
                "update_token_index": int(u_idx),
                "observe_token_index": int(o_idx),
                "layer": int(layer),
                "A_readout": a_readout,
                "A_exact_l": a_exact,
                "B_l": b_l,
                "weighted_exact": a_exact * b_l,
                "weighted_readout": a_readout * b_l,
            })
    df = pd.DataFrame(pair_rows)
    by_layer = []
    for layer, sub in df.groupby("layer"):
        m = pair_metrics(sub["A_exact_l"], sub["A_readout"])
        wm = pair_metrics(sub["weighted_exact"], sub["weighted_readout"])
        m.update({
            "layer": int(layer),
            "weighted_pearson": wm["pearson"],
            "weighted_spearman_signed": wm["spearman_signed"],
            "weighted_spearman_abs": wm["spearman_abs"],
            "weighted_sign_agreement": wm["sign_agreement"],
            "mean_abs_B_l": float(sub["B_l"].abs().mean()),
        })
        by_layer.append(m)
    return df, by_layer


def compute_adjacent(records, pairs):
    all_records = list(records["update"].values()) + list(records["observe"].values())
    n_layers = int(all_records[0]["delta_layers"].shape[0])
    single = []
    for layer in range(n_layers - 1):
        vals = []
        ratios = []
        for rec in all_records:
            a = rec["delta_layers"][layer].numpy()
            b = rec["delta_layers"][layer + 1].numpy()
            vals.append(cos(a, b))
            nb = np.linalg.norm(b)
            ratios.append(float(np.linalg.norm(a) / nb) if nb else np.nan)
        single.append({
            "layer": int(layer),
            "next_layer": int(layer + 1),
            "mean_cos_delta_l_delta_next": float(np.nanmean(vals)),
            "median_cos_delta_l_delta_next": float(np.nanmedian(vals)),
            "mean_norm_ratio_l_to_next": float(np.nanmean(ratios)),
        })

    pair_rows = []
    for pair_index, (u_idx, o_idx) in enumerate(pairs):
        u = records["update"][int(u_idx)]
        o = records["observe"][int(o_idx)]
        for layer in range(n_layers - 1):
            a_l = float(torch.dot(o["delta_layers"][layer], u["delta_layers"][layer]).item())
            a_next = float(torch.dot(o["delta_layers"][layer + 1], u["delta_layers"][layer + 1]).item())
            pair_rows.append({"pair_index": int(pair_index), "layer": int(layer), "A_l": a_l, "A_next": a_next})
    df = pd.DataFrame(pair_rows)
    pair_summary = []
    for layer, sub in df.groupby("layer"):
        m = pair_metrics(sub["A_l"], sub["A_next"])
        m.update({"layer": int(layer), "next_layer": int(layer) + 1})
        pair_summary.append(m)
    return single, pair_summary


def make_plots(out_dir, pair_by_layer, adjacent_single):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    layers = [r["layer"] for r in pair_by_layer]
    sign = [r["sign_agreement"] for r in pair_by_layer]
    pearson = [r["pearson"] for r in pair_by_layer]
    spearman = [r["spearman_signed"] for r in pair_by_layer]
    abs_spear = [r["spearman_abs"] for r in pair_by_layer]

    paths = []
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(layers, sign, marker="o")
    ax.set_xlabel("layer index")
    ax.set_ylabel("sign agreement")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("A_exact_l vs A_readout sign agreement")
    p = fig_dir / "force_readout_sign_agreement_by_layer.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(layers, pearson, marker="o", label="Pearson")
    ax.plot(layers, spearman, marker="o", label="Spearman signed")
    ax.plot(layers, abs_spear, marker="o", label="Spearman abs")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("layer index")
    ax.set_ylabel("correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    ax.set_title("A_exact_l vs A_readout correlations")
    p = fig_dir / "force_readout_correlations_by_layer.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot([r["layer"] for r in adjacent_single], [r["mean_cos_delta_l_delta_next"] for r in adjacent_single], marker="o", label="mean")
    ax.plot([r["layer"] for r in adjacent_single], [r["median_cos_delta_l_delta_next"] for r in adjacent_single], marker="o", label="median")
    ax.set_xlabel("layer l")
    ax.set_ylabel("cos(delta_l, delta_{l+1})")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    ax.set_title("Adjacent layer force drift")
    p = fig_dir / "adjacent_delta_cos_by_layer.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))
    return paths


def write_report(out_dir, summary):
    pair_rows = "\n".join(
        f"| {r['layer']} | {r['pearson']:.4f} | {r['spearman_signed']:.4f} | {r['spearman_abs']:.4f} | {r['sign_agreement']:.4f} | {r['weighted_sign_agreement']:.4f} | {r['mean_abs_B_l']:.4g} |"
        for r in summary["pairwise_by_layer"]
    )
    single_rows = "\n".join(
        f"| {r['layer']} | {r['mean_cos_delta_q']:.4f} | {r['median_cos_delta_q']:.4f} | {r['mean_norm_ratio_delta_q']:.4g} | {r['median_norm_ratio_delta_q']:.4g} |"
        for r in summary["single_token_by_layer"]["by_layer"]
    )
    adj_rows = "\n".join(
        f"| {r['layer']}->{r['next_layer']} | {r['mean_cos_delta_l_delta_next']:.4f} | {r['median_cos_delta_l_delta_next']:.4f} | {r['mean_norm_ratio_l_to_next']:.4g} |"
        for r in summary["adjacent_single_token"]
    )
    plots = "\n".join(summary["plots"])
    final = summary["single_token_by_layer"]["final_norm_grad_vs_q"]
    report = f"""# Residual Force Drift Diagnostic

Run directory:

```text
{out_dir}
```

Model: `{summary['model_name_or_path']}`
Rows/pairs: `{summary['num_pairs']}`
Unique update tokens: `{summary['num_update_tokens']}`
Unique observe tokens: `{summary['num_observe_tokens']}`
Layers: `{summary['num_layers']}`

## Representation Convention

`q = W^T g`, where `g = one_hot(target) - softmax(logits)`, is the exact gradient with respect to the LM-head input representation. For Qwen, the final RMSNorm lies between the final raw residual stream and the LM head. Therefore `q` is directly comparable to the gradient at the final RMSNorm output, not to the raw post-block residual before final RMSNorm.

Layer forces `delta_l` here are exact gradients of token log-probability with respect to the normalized attention input stream of block `l`, i.e. the same normalized residual-stream location used by the current CH2 input-overlap factor for the attention stream.

Final RMSNorm output gradient vs `q` sanity check:

```text
mean cosine:       {final['mean_cos']:.12f}
median cosine:     {final['median_cos']:.12f}
mean norm ratio:   {final['mean_norm_ratio']:.12f}
median norm ratio: {final['median_norm_ratio']:.12f}
```

## Single-Token Force Preservation

| layer | mean cos(delta_l, q) | median cos(delta_l, q) | mean norm ratio | median norm ratio |
|---:|---:|---:|---:|---:|
{single_rows}

## Pairwise Interaction Geometry

Comparison per layer: `A_exact_l = <delta_l_observe, delta_l_update>` vs `A_readout = <q_observe, q_update>`.

| layer | Pearson | signed Spearman | abs Spearman | sign agreement | weighted sign agreement | mean abs B_l |
|---:|---:|---:|---:|---:|---:|---:|
{pair_rows}

## Adjacent Layer Drift

| adjacent layers | mean cos | median cos | mean norm ratio l/(l+1) |
|---|---:|---:|---:|
{adj_rows}

## Plots

```text
{plots}
```

## Interpretation

Inspect the layer table and plots. Case A is supported if agreement with `A_readout` is high near the output and systematically deteriorates toward lower layers. Case B is supported if even the highest layers have weak signed agreement with `A_readout`, which points next toward local block linearization or normalization assumptions rather than accumulated residual-flow drift.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Layer-wise residual force drift diagnostic for Section 3 CH2.")
    parser.add_argument("--source_run_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--allow_download", action="store_true")
    args = parser.parse_args()

    configure_precision()
    source = json.loads(args.source_run_config.read_text(encoding="utf-8"))
    config = load_config(source.get("config", "interaction/configs/train_basic.yaml") if "config" in source else "interaction/configs/train_basic.yaml")
    model_name = source["model_name_or_path"]
    max_length = config["model"]["max_length"]
    local_files_only = not args.allow_download
    device = source.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    pair_seed = source.get("pair_seed", source.get("seed", 42))
    random.seed(source.get("seed", 42))
    torch.manual_seed(source.get("seed", 42))

    mmlu_subjects = source.get("mmlu_subjects") or DEFAULT_MMLU_SUBJECTS
    update_samples, _ = load_adapter_samples(
        adapter_name=source["update_dataset"],
        split=source["update_split"],
        model_name=model_name,
        max_length=max_length,
        config=config,
        local_files_only=local_files_only,
        mmlu_subjects=mmlu_subjects,
    )
    observe_samples_all, _ = load_adapter_samples(
        adapter_name=source["observe_dataset"],
        split=source["observe_split"],
        model_name=model_name,
        max_length=max_length,
        config=config,
        local_files_only=local_files_only,
        mmlu_subjects=mmlu_subjects,
    )
    observe_samples = select_samples_by_domain(
        samples=observe_samples_all,
        max_per_domain=source["max_observe_samples_per_domain"],
        seed=source.get("seed", 42),
        max_total=source.get("max_observe_samples"),
    )
    update_tokens = build_update_tokens(update_samples, source["max_update_samples"], source["max_update_tokens_per_sample"])
    observe_tokens = build_observe_tokens(observe_samples, source["max_observe_tokens_per_sample"])
    pairs = build_pairs(update_tokens, observe_tokens, source.get("max_pairs"), pair_seed)

    model = load_model(
        model_name=model_name,
        local_files_only=local_files_only,
        device=device,
        device_map=source.get("device_map"),
        torch_dtype=source.get("torch_dtype", "float32"),
        attn_implementation=source.get("attn_implementation", "eager"),
    )
    records = build_force_records(model, update_tokens, observe_tokens, pairs)
    pair_df, pair_by_layer = compute_pairwise(records, pairs)
    single = summarize_single_token(records)
    adjacent_single, adjacent_pair = compute_adjacent(records, pairs)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(out_dir / "residual_force_pairwise_by_layer.csv", index=False)
    plots = make_plots(out_dir, pair_by_layer, adjacent_single)
    summary = {
        "model_name_or_path": model_name,
        "dtype_report": model_dtype_report(model),
        "source_run_config": str(args.source_run_config),
        "num_pairs": len(pairs),
        "num_update_tokens": len(records["update"]),
        "num_observe_tokens": len(records["observe"]),
        "num_layers": len(pair_by_layer),
        "single_token_by_layer": single,
        "pairwise_by_layer": pair_by_layer,
        "adjacent_single_token": adjacent_single,
        "adjacent_pairwise": adjacent_pair,
        "plots": plots,
    }
    (out_dir / "residual_force_drift_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, summary)
    print(json.dumps({"output_dir": str(out_dir), "num_pairs": len(pairs), "num_layers": len(pair_by_layer)}, indent=2))


if __name__ == "__main__":
    main()
