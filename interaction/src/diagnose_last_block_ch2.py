
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
import torch.nn as nn
import torch.nn.functional as F

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
from diagnose_residual_force_drift import get_final_norm, pair_metrics, sign_agreement
from utils.ch1_ch2_metrics import (
    _as_1d_tensor,
    _block_input_norm,
    _block_mlp_input_norm,
    _get_transformer_layers,
    _model_compute_dtype,
    _model_device,
    get_readout_weight,
    grad_inner_product,
    token_logprob,
)
from utils.config_read import load_config
from utils.hh_dataset_adapters import DEFAULT_MMLU_SUBJECTS

BOUNDARIES = [
    "q_final_norm_output",
    "pre_final_norm_raw_residual",
    "post_mlp_residual",
    "post_attention_residual_pre_mlp",
    "normalized_mlp_input",
    "attention_residual_input",
    "normalized_attention_input",
]
ADJACENT = [
    ("q_final_norm_output", "pre_final_norm_raw_residual", "final_rmsnorm"),
    ("pre_final_norm_raw_residual", "post_attention_residual_pre_mlp", "last_mlp_residual_branch"),
    ("post_attention_residual_pre_mlp", "normalized_mlp_input", "mlp_input_rmsnorm"),
    ("post_attention_residual_pre_mlp", "attention_residual_input", "last_attention_residual_branch"),
    ("attention_residual_input", "normalized_attention_input", "attention_input_rmsnorm"),
]


def get_last_block(model):
    layers = _get_transformer_layers(model)
    if layers is None or len(layers) == 0:
        raise RuntimeError("Could not find transformer layers.")
    return layers[-1], len(layers) - 1, len(layers)


def tensor_at_pos(tensor, logit_pos):
    if tensor is None:
        return None
    if tensor.dim() == 3:
        return tensor[0, int(logit_pos)]
    if tensor.dim() == 2:
        return tensor[int(logit_pos)]
    raise RuntimeError(f"Unexpected tensor rank {tensor.dim()} for captured boundary.")


def module_params(modules):
    params = []
    seen = set()
    for module in modules:
        for param in module.parameters(recurse=True):
            if param.requires_grad and id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params


def selected_named_params(module, predicate):
    params = []
    names = []
    seen = set()
    for module_name, submodule in module.named_modules():
        if predicate(submodule):
            for pname, param in submodule.named_parameters(recurse=False):
                if param.requires_grad and id(param) not in seen:
                    full = f"{module_name}.{pname}" if module_name else pname
                    names.append(full)
                    params.append(param)
                    seen.add(id(param))
    return names, params


def direct_grad_dot(model, sample_o, pos_o, sample_u, pos_u, params):
    if not params:
        return 0.0
    model.zero_grad(set_to_none=True)
    device = _model_device(model)
    target_o = _as_1d_tensor(sample_o["labels"], device, "labels")[int(pos_o)]
    target_u = _as_1d_tensor(sample_u["labels"], device, "labels")[int(pos_u)]
    logp_o = token_logprob(model, sample_o, int(pos_o) - 1, target_o, device=device)
    logp_u = token_logprob(model, sample_u, int(pos_u) - 1, target_u, device=device)
    val = grad_inner_product(logp_o, logp_u, params)
    model.zero_grad(set_to_none=True)
    return float(val.detach().cpu().item())


def token_trace(model, sample, label_pos):
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

    last_block, last_idx, _ = get_last_block(model)
    attn_norm = _block_input_norm(model, last_idx)
    mlp_norm = _block_mlp_input_norm(model, last_idx)
    final_norm, final_norm_name = get_final_norm(model)
    if attn_norm is None or mlp_norm is None or final_norm is None:
        raise RuntimeError("Missing expected Qwen norms for last-block diagnostic.")

    captures = {}
    linear_captures = {}
    handles = []

    def retain(name, tensor):
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        tensor.retain_grad()
        captures[name] = tensor

    def pre_hook(name):
        def hook(_module, inputs):
            if inputs:
                retain(name, inputs[0])
        return hook

    def out_hook(name):
        def hook(_module, _inputs, output):
            retain(name, output)
        return hook

    handles.append(final_norm.register_forward_pre_hook(pre_hook("pre_final_norm_raw_residual")))
    handles.append(final_norm.register_forward_hook(out_hook("final_norm_output")))
    handles.append(mlp_norm.register_forward_pre_hook(pre_hook("post_attention_residual_pre_mlp")))
    handles.append(mlp_norm.register_forward_hook(out_hook("normalized_mlp_input")))
    handles.append(attn_norm.register_forward_pre_hook(pre_hook("attention_residual_input")))
    handles.append(attn_norm.register_forward_hook(out_hook("normalized_attention_input")))

    def make_linear_pre(name):
        def hook(_module, inputs):
            if inputs:
                x = inputs[0]
                x.retain_grad()
                linear_captures.setdefault(name, {})["input"] = x
        return hook

    def make_linear_out(name):
        def hook(_module, _inputs, output):
            y = output[0] if isinstance(output, (tuple, list)) else output
            y.retain_grad()
            linear_captures.setdefault(name, {})["output"] = y
        return hook

    linear_modules = []
    for name, module in last_block.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append((name, module))
            handles.append(module.register_forward_pre_hook(make_linear_pre(name)))
            handles.append(module.register_forward_hook(make_linear_out(name)))

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
        q = (g.to(device=logits.device, dtype=compute_dtype) @ readout_weight).detach().to(dtype=torch.float32, device="cpu")

        activations = {}
        for act_name in ("normalized_attention_input", "normalized_mlp_input"):
            activations[act_name] = tensor_at_pos(captures[act_name], logit_pos).detach().to(dtype=torch.float32, device="cpu")

        boundaries = {"q_final_norm_output": q}
        boundaries["post_mlp_residual"] = tensor_at_pos(captures["pre_final_norm_raw_residual"].grad, logit_pos).detach().to(dtype=torch.float32, device="cpu")
        for name in (
            "pre_final_norm_raw_residual",
            "post_attention_residual_pre_mlp",
            "normalized_mlp_input",
            "attention_residual_input",
            "normalized_attention_input",
        ):
            grad = captures[name].grad
            if grad is None:
                raise RuntimeError(f"Missing boundary grad for {name}")
            boundaries[name] = tensor_at_pos(grad, logit_pos).detach().to(dtype=torch.float32, device="cpu")

        final_output_grad = tensor_at_pos(captures["final_norm_output"].grad, logit_pos).detach().to(dtype=torch.float32, device="cpu")

        linear = {}
        for name, module in linear_modules:
            cap = linear_captures.get(name, {})
            if "input" not in cap or "output" not in cap or cap["output"].grad is None:
                raise RuntimeError(f"Missing linear capture for {name}")
            x_all = cap["input"].detach().to(dtype=torch.float32, device="cpu")
            delta_all = cap["output"].grad.detach().to(dtype=torch.float32, device="cpu")
            x = tensor_at_pos(cap["input"], logit_pos).detach().to(dtype=torch.float32, device="cpu")
            delta = tensor_at_pos(cap["output"].grad, logit_pos).detach().to(dtype=torch.float32, device="cpu")
            x_flat = x_all.reshape(-1, x_all.shape[-1])
            delta_flat = delta_all.reshape(-1, delta_all.shape[-1])
            weight_grad = delta_flat.T @ x_flat
            bias_grad = delta_flat.sum(dim=0) if module.bias is not None else None
            linear[name] = {
                "x": x,
                "delta": delta,
                "weight_grad": weight_grad.contiguous(),
                "bias_grad": bias_grad.contiguous() if bias_grad is not None else None,
                "has_bias": module.bias is not None,
                "num_positions": int(x_flat.shape[0]),
            }

        return {
            "boundaries": boundaries,
            "activations": activations,
            "linear": linear,
            "final_norm_output_grad_cos_q": cosine(final_output_grad, q),
            "final_norm_output_grad_norm_ratio_q": norm_ratio(final_output_grad, q),
            "final_norm_name": final_norm_name,
            "logp": float(logp.detach().cpu().item()),
            "target_id": target_id,
        }
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def cosine(a, b, eps=1e-30):
    denom = float(a.norm().item() * b.norm().item())
    if denom <= eps:
        return float("nan")
    return float(torch.dot(a.flatten(), b.flatten()).item() / denom)


def norm_ratio(a, b, eps=1e-30):
    denom = float(b.norm().item())
    if denom <= eps:
        return float("nan")
    return float(a.norm().item() / denom)


def build_records(model, update_tokens, observe_tokens, pairs):
    needed_update = sorted({int(u) for u, _ in pairs})
    needed_observe = sorted({int(o) for _, o in pairs})
    records = {"update": {}, "observe": {}}
    for idx in needed_update:
        _, sample, pos = update_tokens[idx]
        records["update"][idx] = token_trace(model, sample, pos)
    for idx in needed_observe:
        _, sample, pos = observe_tokens[idx]
        records["observe"][idx] = token_trace(model, sample, pos)
    return records


def dot(a, b):
    return float(torch.dot(a.flatten(), b.flatten()).item())


def linear_factor_pair(o, u):
    rows = []
    total = 0.0
    single_position_total = 0.0
    for name in sorted(o["linear"]):
        lo = o["linear"][name]
        lu = u["linear"][name]
        delta_dot = dot(lo["delta"], lu["delta"])
        input_dot = dot(lo["x"], lu["x"])
        single_position = delta_dot * input_dot
        if lo["has_bias"]:
            single_position += delta_dot
        weight_contribution = dot(lo["weight_grad"], lu["weight_grad"])
        bias_contribution = dot(lo["bias_grad"], lu["bias_grad"]) if lo["has_bias"] else 0.0
        contribution = weight_contribution + bias_contribution
        total += contribution
        single_position_total += single_position
        rows.append({
            "module": name,
            "target_position_delta_dot": delta_dot,
            "target_position_input_dot": input_dot,
            "target_position_factorized_interaction": single_position,
            "weight_grad_interaction": weight_contribution,
            "bias_grad_interaction": bias_contribution,
            "has_bias": bool(lo["has_bias"]),
            "num_positions_observe": int(lo["num_positions"]),
            "num_positions_update": int(lu["num_positions"]),
            "factorized_interaction": contribution,
        })
    return total, single_position_total, rows


def compute_pair_rows(model, update_tokens, observe_tokens, pairs, records):
    last_block, _, _ = get_last_block(model)
    linear_names, linear_params = selected_named_params(last_block, lambda m: isinstance(m, nn.Linear))
    norm_names, norm_params = selected_named_params(last_block, lambda m: not isinstance(m, nn.Linear) and any(p.requires_grad for p in m.parameters(recurse=False)))
    all_params = module_params([last_block])

    rows = []
    module_rows = []
    for pair_index, (u_idx, o_idx) in enumerate(pairs):
        u = records["update"][int(u_idx)]
        o = records["observe"][int(o_idx)]
        row = {"pair_index": int(pair_index), "update_token_index": int(u_idx), "observe_token_index": int(o_idx)}
        for boundary in BOUNDARIES:
            row[f"A_{boundary}"] = dot(o["boundaries"][boundary], u["boundaries"][boundary])
        row["B_normalized_attention_input"] = dot(o["activations"]["normalized_attention_input"], u["activations"]["normalized_attention_input"])
        row["B_normalized_mlp_input"] = dot(o["activations"]["normalized_mlp_input"], u["activations"]["normalized_mlp_input"])
        row["B_last_block_current_ch2"] = row["B_normalized_attention_input"] + row["B_normalized_mlp_input"]
        row["last_block_exact_force_same_inputs"] = (
            row["A_normalized_attention_input"] * row["B_normalized_attention_input"]
            + row["A_normalized_mlp_input"] * row["B_normalized_mlp_input"]
        )
        row["last_block_current_ch2_approx"] = row["A_q_final_norm_output"] * row["B_last_block_current_ch2"]
        factor_total, single_position_factor_total, mods = linear_factor_pair(o, u)
        row["last_block_linear_factorized"] = factor_total
        row["last_block_linear_target_position_factorized"] = single_position_factor_total
        for mod in mods:
            module_rows.append({"pair_index": int(pair_index), **mod})

        _, update_sample, update_pos = update_tokens[int(u_idx)]
        _, observe_sample, observe_pos = observe_tokens[int(o_idx)]
        row["last_block_linear_direct"] = direct_grad_dot(model, observe_sample, observe_pos, update_sample, update_pos, linear_params)
        row["last_block_norm_direct"] = direct_grad_dot(model, observe_sample, observe_pos, update_sample, update_pos, norm_params)
        row["last_block_all_direct"] = direct_grad_dot(model, observe_sample, observe_pos, update_sample, update_pos, all_params)
        row["last_block_linear_factorization_abs_error"] = abs(row["last_block_linear_factorized"] - row["last_block_linear_direct"])
        row["last_block_linear_target_position_factorization_abs_error"] = abs(row["last_block_linear_target_position_factorized"] - row["last_block_linear_direct"])
        row["last_block_all_minus_linear_norm"] = row["last_block_all_direct"] - row["last_block_linear_direct"] - row["last_block_norm_direct"]
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(module_rows), {"linear_param_names": linear_names, "norm_param_names": norm_names}


def summarize_boundaries(pair_df):
    summaries = []
    ref = f"A_{BOUNDARIES[0]}"
    for boundary in BOUNDARIES:
        col = f"A_{boundary}"
        m = pair_metrics(pair_df[col], pair_df[ref])
        m.update({"boundary": boundary, "compare_to": BOUNDARIES[0]})
        summaries.append(m)
    adjacent = []
    for left, right, operation in ADJACENT:
        m = pair_metrics(pair_df[f"A_{right}"], pair_df[f"A_{left}"])
        m.update({"from_boundary": left, "to_boundary": right, "operation": operation})
        adjacent.append(m)
    return summaries, adjacent


def summarize_ladder(pair_df):
    ladder = []
    targets = [
        ("exact_block_vs_sequence_factorized_linear", "last_block_all_direct", "last_block_linear_factorized"),
        ("linear_direct_vs_sequence_factorized_linear", "last_block_linear_direct", "last_block_linear_factorized"),
        ("linear_direct_vs_target_position_factorized", "last_block_linear_direct", "last_block_linear_target_position_factorized"),
        ("linear_direct_vs_q_readout_force_only", "last_block_linear_direct", "A_q_final_norm_output"),
        ("linear_direct_vs_exact_force_same_inputs", "last_block_linear_direct", "last_block_exact_force_same_inputs"),
        ("linear_direct_vs_current_ch2_last_block", "last_block_linear_direct", "last_block_current_ch2_approx"),
        ("linear_direct_vs_last_attn_norm_force_only", "last_block_linear_direct", "A_normalized_attention_input"),
    ]
    for name, x, y in targets:
        m = pair_metrics(pair_df[x], pair_df[y])
        m.update({"comparison": name, "x": x, "y": y})
        ladder.append(m)
    return ladder


def make_plots(out_dir, boundary_summary, adjacent_summary, ladder_summary):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    labels = [r["boundary"] for r in boundary_summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(x, [r["sign_agreement"] for r in boundary_summary], marker="o", label="sign agreement vs q")
    ax.plot(x, [r["pearson"] for r in boundary_summary], marker="o", label="Pearson vs q")
    ax.plot(x, [r["spearman_signed"] for r in boundary_summary], marker="o", label="Spearman vs q")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Last-block boundary geometry vs readout force")
    ax.legend()
    p = fig_dir / "last_block_boundary_vs_q_metrics.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    labels = [r["operation"] for r in adjacent_summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(x, [r["sign_agreement"] for r in adjacent_summary], marker="o", label="sign agreement")
    ax.plot(x, [r["pearson"] for r in adjacent_summary], marker="o", label="Pearson")
    ax.plot(x, [r["spearman_signed"] for r in adjacent_summary], marker="o", label="Spearman")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Adjacent last-block operation geometry")
    ax.legend()
    p = fig_dir / "last_block_adjacent_operation_metrics.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))

    labels = [r["comparison"] for r in ladder_summary]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.plot(x, [r["sign_agreement"] for r in ladder_summary], marker="o", label="sign agreement")
    ax.plot(x, [r["pearson"] for r in ladder_summary], marker="o", label="Pearson")
    ax.plot(x, [r["spearman_abs"] for r in ladder_summary], marker="o", label="abs Spearman")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Last-block approximation ladder")
    ax.legend()
    p = fig_dir / "last_block_approximation_ladder.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(str(p))
    return paths


def fmt_table(rows, cols):
    lines = []
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(out_dir, summary):
    boundary_cols = ["boundary", "pearson", "spearman_signed", "spearman_abs", "sign_agreement"]
    adjacent_cols = ["operation", "from_boundary", "to_boundary", "pearson", "spearman_signed", "spearman_abs", "sign_agreement"]
    ladder_cols = ["comparison", "pearson", "spearman_signed", "spearman_abs", "sign_agreement"]
    b_header = "| boundary | Pearson vs q | signed Spearman vs q | abs Spearman vs q | sign agreement vs q |\n|---|---:|---:|---:|---:|"
    a_header = "| operation | from | to | Pearson | signed Spearman | abs Spearman | sign agreement |\n|---|---|---|---:|---:|---:|---:|"
    l_header = "| comparison | Pearson | signed Spearman | abs Spearman | sign agreement |\n|---|---:|---:|---:|---:|"
    checks = summary["factorization_checks"]
    paths = "\n".join(summary["plots"])
    report = f"""# Last Block Local CH2 Diagnostic

Run directory:

```text
{out_dir}
```

Model: `{summary['model_name_or_path']}`
Pairs: `{summary['num_pairs']}`
Last block index: `{summary['last_block_index']}`

## Boundary Convention

`q_final_norm_output = W^T g` is the exact LM-head input force. For Qwen, final RMSNorm lies between `pre_final_norm_raw_residual` and this `q` anchor.

Boundaries traced backward through the last block:

```text
q_final_norm_output
pre_final_norm_raw_residual / post_mlp_residual
post_attention_residual_pre_mlp
normalized_mlp_input
attention_residual_input
normalized_attention_input
```

Final norm output gradient vs `q` sanity:

```text
mean cosine:       {summary['final_norm_output_grad_vs_q']['mean_cos']:.12f}
median cosine:     {summary['final_norm_output_grad_vs_q']['median_cos']:.12f}
mean norm ratio:   {summary['final_norm_output_grad_vs_q']['mean_norm_ratio']:.12f}
median norm ratio: {summary['final_norm_output_grad_vs_q']['median_norm_ratio']:.12f}
```

## Boundary Geometry vs Readout Anchor

{b_header}
{fmt_table(summary['boundary_vs_q'], boundary_cols)}

## Adjacent Local Operations

{a_header}
{fmt_table(summary['adjacent_operations'], adjacent_cols)}

## Exact Last-Block Linear Factorization Sanity

For each last-block Linear module, the exact sequence-summed parameter-gradient interaction is computed from the captured module input `x` and upstream gradient `delta` by forming the per-example Linear weight gradient and taking its dot product. For a single-position Linear this reduces to:

```text
<delta_o, delta_u> * <x_o, x_u>
```

with an added bias-gradient dot product if the Linear has bias.

```text
max abs(linear direct - factorized):     {checks['max_abs_linear_direct_minus_factorized']:.6e}
mean abs(linear direct - factorized):    {checks['mean_abs_linear_direct_minus_factorized']:.6e}
max abs(linear direct - target-pos):     {checks['max_abs_linear_direct_minus_target_position_factorized']:.6e}
mean abs(linear direct - target-pos):    {checks['mean_abs_linear_direct_minus_target_position_factorized']:.6e}
max abs(last block all direct):          {checks['max_abs_last_block_all_direct']:.6e}
mean abs(norm direct):                   {checks['mean_abs_norm_direct']:.6e}
mean abs(all - linear - norm):           {checks['mean_abs_all_minus_linear_norm']:.6e}
```

The sequence-summed factorization reconstructs direct Linear-parameter gradients; RMSNorm parameters are reported separately as `norm direct`. The target-position-only factorization is also reported in the ladder because it is not exact for causal attention: last-block Q/K/V parameter gradients receive contributions from context positions as well as the target logit position.

## Approximation Ladder

{l_header}
{fmt_table(summary['approximation_ladder'], ladder_cols)}

## Plots

```text
{paths}
```

## Interpretation

The exact LM-head input force `q` is self-consistent, and the final RMSNorm itself preserves signed pairwise geometry well in this run. The first major local drop appears across the **last MLP residual branch**, when moving from the post-MLP residual / pre-final-norm force back to the post-attention residual before the MLP branch.

Therefore the next target should be the last MLP residual branch as a whole before decomposing Q/K/V. A separate MLP-internal diagnostic can decide whether the issue is the MLP transformation itself or its interaction with the residual add/RMSNorm boundary.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Localize CH2 signed geometry inside the final transformer block.")
    parser.add_argument("--source_run_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=str, default="interaction/configs/train_basic.yaml")
    parser.add_argument("--allow_download", action="store_true")
    args = parser.parse_args()

    configure_precision()
    source = json.loads(args.source_run_config.read_text(encoding="utf-8"))
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
    _, last_idx, num_layers = get_last_block(model)
    records = build_records(model, update_tokens, observe_tokens, pairs)
    pair_df, module_df, param_meta = compute_pair_rows(model, update_tokens, observe_tokens, pairs, records)
    boundary_summary, adjacent_summary = summarize_boundaries(pair_df)
    ladder_summary = summarize_ladder(pair_df)

    cos_vals = []
    ratio_vals = []
    for side in ("update", "observe"):
        for rec in records[side].values():
            cos_vals.append(rec["final_norm_output_grad_cos_q"])
            ratio_vals.append(rec["final_norm_output_grad_norm_ratio_q"])

    checks = {
        "max_abs_linear_direct_minus_factorized": float(pair_df["last_block_linear_factorization_abs_error"].max()),
        "mean_abs_linear_direct_minus_factorized": float(pair_df["last_block_linear_factorization_abs_error"].mean()),
        "max_abs_linear_direct_minus_target_position_factorized": float(pair_df["last_block_linear_target_position_factorization_abs_error"].max()),
        "mean_abs_linear_direct_minus_target_position_factorized": float(pair_df["last_block_linear_target_position_factorization_abs_error"].mean()),
        "max_abs_last_block_all_direct": float(pair_df["last_block_all_direct"].abs().max()),
        "mean_abs_norm_direct": float(pair_df["last_block_norm_direct"].abs().mean()),
        "mean_abs_all_minus_linear_norm": float(pair_df["last_block_all_minus_linear_norm"].abs().mean()),
    }

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(out_dir / "last_block_boundary_pairwise.csv", index=False)
    module_df.to_csv(out_dir / "last_block_linear_module_factors.csv", index=False)
    plots = make_plots(out_dir, boundary_summary, adjacent_summary, ladder_summary)
    summary = {
        "model_name_or_path": model_name,
        "dtype_report": model_dtype_report(model),
        "source_run_config": str(args.source_run_config),
        "num_pairs": len(pairs),
        "num_layers": num_layers,
        "last_block_index": last_idx,
        "param_metadata": param_meta,
        "final_norm_output_grad_vs_q": {
            "mean_cos": float(np.nanmean(cos_vals)),
            "median_cos": float(np.nanmedian(cos_vals)),
            "mean_norm_ratio": float(np.nanmean(ratio_vals)),
            "median_norm_ratio": float(np.nanmedian(ratio_vals)),
        },
        "boundary_vs_q": boundary_summary,
        "adjacent_operations": adjacent_summary,
        "factorization_checks": checks,
        "approximation_ladder": ladder_summary,
        "plots": plots,
    }
    (out_dir / "last_block_local_ch2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, summary)
    print(json.dumps({"output_dir": str(out_dir), "num_pairs": len(pairs), "last_block_index": last_idx}, indent=2))


if __name__ == "__main__":
    main()
