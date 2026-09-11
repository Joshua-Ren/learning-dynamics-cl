import argparse
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from run_section3_validation import (
    build_observe_tokens,
    build_pairs,
    build_update_tokens,
    configure_precision,
    load_adapter_samples,
    model_dtype_report,
    resolve_param_scope,
    resolve_torch_dtype,
    run_section3,
    select_samples_by_domain,
    write_csv_rows,
    write_json,
    write_jsonl,
    write_yaml,
)
from utils.ch1_ch2_metrics import _as_1d_tensor, _model_device, grad_inner_product, token_logprob
from utils.config_read import load_config
from utils.hh_dataset_adapters import DEFAULT_MMLU_SUBJECTS, parse_csv_arg


COMPARISONS = {
    "A_actual_vs_exact": ("delta_logp", "first_order_exact"),
    "B_exact_vs_approx": ("first_order_exact", "approx"),
    "C_actual_vs_approx": ("delta_logp", "approx"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Light SFT adaptation followed by fixed Section 3 validation.")
    parser.add_argument("--config", type=str, default="interaction/configs/train_basic.yaml")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--adapt_split", type=str, default="train[10:110]")
    parser.add_argument("--adapt_lr", type=float, default=1e-5)
    parser.add_argument("--adapt_lr_start", type=float, default=None)
    parser.add_argument("--adapt_lr_end", type=float, default=None)
    parser.add_argument("--adapt_lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adapt_max_steps", type=int, default=100)
    parser.add_argument("--adapt_batch_size", type=int, default=1)
    parser.add_argument("--eval_every_steps", type=int, default=0)
    parser.add_argument("--eval_at_epochs", type=str, default="")

    parser.add_argument("--update_split", type=str, default="train[:10]")
    parser.add_argument("--observe_split", type=str, default="test")
    parser.add_argument("--mmlu_subjects", type=str, default="abstract_algebra")
    parser.add_argument("--max_update_samples", type=int, default=10)
    parser.add_argument("--max_update_tokens_per_sample", type=int, default=8)
    parser.add_argument("--max_observe_samples_per_domain", type=int, default=10)
    parser.add_argument("--max_observe_samples", type=int, default=10)
    parser.add_argument("--max_observe_tokens_per_sample", type=int, default=8)
    parser.add_argument("--max_pairs", type=int, default=512)

    parser.add_argument("--validation_lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_seed", type=int, default=42)
    parser.add_argument("--ch_layer", type=str, default="-1")
    parser.add_argument("--param_scope", type=str, default="all", choices=["all", "modeled", "readout_only"])
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["auto", "float32", "float64", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--debug_alignment_rows", type=int, default=0)
    return parser.parse_args()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def load_model_cast(model_name, local_files_only, device, device_map, torch_dtype, attn_implementation):
    requested = resolve_torch_dtype(torch_dtype)
    load_dtype = "auto" if requested in (torch.float32, torch.float64) else requested
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "torch_dtype": load_dtype,
    }
    if device_map:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not device_map:
        model = model.to(device)
    if requested != "auto":
        model = model.to(dtype=requested)
    model.eval()
    return model


def batch_samples(samples, batch_size):
    for start in range(0, len(samples), batch_size):
        yield samples[start:start + batch_size]


def make_batch(samples, device):
    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples], dim=0).to(device),
        "attention_mask": torch.stack([s["attention_mask"] for s in samples], dim=0).to(device),
        "labels": torch.stack([s["labels"] for s in samples], dim=0).to(device),
    }


def run_aux_sft(model, samples, out_dir, args, evaluate):
    device = _model_device(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.adapt_lr)
    losses = []
    checkpoint_meta = {}
    eval_at_epochs = {int(value) for value in args.eval_at_epochs.split(",") if value.strip()}
    steps_per_epoch = math.ceil(len(samples) / args.adapt_batch_size)

    def evaluate_now(name, step, samples_seen, loss):
        metadata = {
            "step": step,
            "samples_seen": samples_seen,
            "epoch": (samples_seen - 1) // len(samples) + 1 if samples_seen else 0,
            "loss": loss,
        }
        checkpoint_meta[name] = metadata
        model.eval()
        evaluate(name, metadata)
        model.train()

    evaluate_now("base", 0, 0, None)
    step = 0
    samples_seen = 0
    model.train()
    while step < args.adapt_max_steps:
        for batch in batch_samples(samples, args.adapt_batch_size):
            step += 1
            if args.adapt_lr_start is not None or args.adapt_lr_end is not None:
                lr_start = args.adapt_lr if args.adapt_lr_start is None else args.adapt_lr_start
                lr_end = args.adapt_lr if args.adapt_lr_end is None else args.adapt_lr_end
                if args.adapt_lr_warmup_steps > 0 and step <= args.adapt_lr_warmup_steps:
                    denom = max(1, args.adapt_lr_warmup_steps - 1)
                    step_lr = lr_start + (lr_end - lr_start) * ((step - 1) / denom)
                else:
                    step_lr = lr_end
                for group in optimizer.param_groups:
                    group["lr"] = step_lr
            else:
                step_lr = args.adapt_lr
            optimizer.zero_grad(set_to_none=True)
            inputs = make_batch(batch, device)
            loss = model(**inputs, output_hidden_states=False, output_attentions=False).loss
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            samples_seen += len(batch)
            epoch = (samples_seen - 1) // len(samples) + 1
            losses.append({
                "step": step,
                "samples_seen": samples_seen,
                "epoch": epoch,
                "loss": loss_value,
                "lr": float(step_lr),
            })
            # Persist the curve continuously: the run has many expensive evaluations.
            pd.DataFrame(losses).to_csv(out_dir / "sft_loss_curve.csv", index=False)
            is_first_epoch_interval = (
                args.eval_every_steps > 0
                and step <= steps_per_epoch
                and step % args.eval_every_steps == 0
            )
            is_selected_epoch_end = (
                samples_seen % len(samples) == 0
                and epoch in eval_at_epochs
                and epoch != 1
            )
            if is_first_epoch_interval:
                evaluate_now(f"epoch01_update{step:04d}", step, samples_seen, loss_value)
            elif is_selected_epoch_end:
                evaluate_now(f"epoch{epoch:02d}_end", step, samples_seen, loss_value)
            if step >= args.adapt_max_steps:
                break
    pd.DataFrame(losses).to_csv(out_dir / "sft_loss_curve.csv", index=False)
    model.eval()
    return checkpoint_meta, losses


def metric_values(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return {
            "n": int(len(x)),
            "pearson": None,
            "spearman_signed": None,
            "spearman_abs": None,
            "sign_agreement": None,
            "slope_y_on_x_through_origin": None,
        }
    denom = float(np.dot(x, x))
    return {
        "n": int(len(x)),
        "pearson": float(x.corr(y, method="pearson")),
        "spearman_signed": float(x.rank(method="average").corr(y.rank(method="average"), method="pearson")),
        "spearman_abs": float(x.abs().rank(method="average").corr(y.abs().rank(method="average"), method="pearson")),
        "sign_agreement": float((np.sign(x) == np.sign(y)).mean()),
        "slope_y_on_x_through_origin": float(np.dot(x, y) / denom) if denom != 0.0 else None,
    }


def compute_update_grad_norms(model, update_tokens, params):
    device = _model_device(model)
    rows = []
    model.eval()
    for token_index, (_, sample, pos) in enumerate(update_tokens):
        model.zero_grad(set_to_none=True)
        target = _as_1d_tensor(sample["labels"], device, "labels")[int(pos)]
        logp = token_logprob(model, sample, int(pos) - 1, target, device=device)
        grads = torch.autograd.grad(logp, params, retain_graph=False, create_graph=False, allow_unused=True)
        sq = 0.0
        for grad in grads:
            if grad is not None:
                sq += float(torch.sum(grad.detach().float() ** 2).cpu().item())
        rows.append({"update_token_index": token_index, "update_logp": float(logp.detach().cpu().item()), "update_grad_norm": math.sqrt(sq)})
    model.zero_grad(set_to_none=True)
    return pd.DataFrame(rows)


def make_validation_args(args, output_dir):
    return SimpleNamespace(
        lr=args.validation_lr,
        ch_layer=args.ch_layer,
        param_scope=args.param_scope,
        readout_sanity="none",
        debug_alignment=args.debug_alignment_rows > 0,
        debug_alignment_rows=args.debug_alignment_rows,
    )


def plot_metrics(out_dir, metrics_df):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    metric_names = ["pearson", "spearman_signed", "spearman_abs", "sign_agreement"]
    checkpoint_order = metrics_df.drop_duplicates("checkpoint").sort_values("adapt_step")
    x_labels = checkpoint_order["checkpoint"].tolist()
    x = checkpoint_order["adapt_step"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)
    for ax, metric in zip(axes, metric_names):
        for comparison in COMPARISONS:
            sub = metrics_df[metrics_df["comparison"] == comparison].set_index("checkpoint").reindex(x_labels)
            ax.plot(x, sub[metric], marker="o", label=comparison.replace("_", " "))
        ax.axhline(0, color="gray", linewidth=0.8)
        if metric == "sign_agreement":
            ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=7)
        ax.set_xlabel("adaptation step")
        ax.set_title(metric)
        ax.set_ylim(-1.05, 1.05)
    axes[0].legend(fontsize=7)
    path = fig_dir / "checkpoint_metric_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def plot_scatters(out_dir, rows_df):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    checkpoints = rows_df.drop_duplicates("checkpoint").sort_values("adapt_step")["checkpoint"].tolist()
    for checkpoint in (checkpoints[0], checkpoints[-1]):
        sub = rows_df[rows_df["checkpoint"] == checkpoint]
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
        for ax, (comparison, (x_col, y_col)) in zip(axes, COMPARISONS.items()):
            x = sub[x_col].to_numpy(dtype=float)
            y = sub[y_col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], s=12, alpha=0.35)
            if mask.any():
                lim = float(np.nanmax(np.abs(np.concatenate([x[mask], y[mask]])))) * 1.05
                if lim == 0.0:
                    lim = 1.0
                ax.plot([-lim, lim], [-lim, lim], color="black", linestyle="--", linewidth=0.8)
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.axvline(0, color="gray", linewidth=0.8)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(comparison)
        path = fig_dir / f"signed_scatters_{checkpoint}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def write_report(out_dir, run_config, metrics_df, support_df, plots):
    metrics_text = metrics_df.to_string(index=False)
    support_text = support_df.to_string(index=False)
    plots_text = "\n".join(plots)
    report = f"""# Step1+ SFT Adaptation Section 3 Report

Run directory:

```text
{out_dir}
```

Model: `{run_config['model_name_or_path']}`
Auxiliary adaptation split: `{run_config['adapt_split']}`
Held-out validation update split: `{run_config['update_split']}`
Validation combo: `gsm8k -> mmlu`
Adaptation LR: `{run_config['adapt_lr']}`
Validation LR: `{run_config['validation_lr']}`

## Metrics

```text
{metrics_text}
```

## Supporting Diagnostics

```text
{support_text}
```

## Figures

```text
{plots_text}
```

## Notes

The auxiliary GSM8K adaptation split starts after the first 10 training examples, while Section 3 validation uses `train[:10]`, so the GSM8K adaptation and validation update examples are disjoint by construction.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    configure_precision()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    local_files_only = not args.allow_download
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    max_length = config["model"]["max_length"]
    mmlu_subjects = parse_csv_arg(args.mmlu_subjects) or DEFAULT_MMLU_SUBJECTS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(args.output_dir / "resolved_config.yaml", config)

    adapt_samples, adapter = load_adapter_samples("gsm8k", args.adapt_split, args.model_name, max_length, config, local_files_only, mmlu_subjects)
    update_samples, _ = load_adapter_samples("gsm8k", args.update_split, args.model_name, max_length, config, local_files_only, mmlu_subjects)
    observe_samples_all, _ = load_adapter_samples("mmlu", args.observe_split, args.model_name, max_length, config, local_files_only, mmlu_subjects)
    observe_samples = select_samples_by_domain(observe_samples_all, args.max_observe_samples_per_domain, args.seed, args.max_observe_samples)
    update_tokens = build_update_tokens(update_samples, args.max_update_samples, args.max_update_tokens_per_sample)
    observe_tokens = build_observe_tokens(observe_samples, args.max_observe_tokens_per_sample)
    pairs = build_pairs(update_tokens, observe_tokens, args.max_pairs, args.pair_seed)

    model = load_model_cast(args.model_name, local_files_only, device, args.device_map, args.torch_dtype, args.attn_implementation)
    param_scope_params, param_scope_metadata = resolve_param_scope(model, args.param_scope)
    run_config = {
        "model_name_or_path": args.model_name,
        "adapt_split": args.adapt_split,
        "adapt_lr": args.adapt_lr,
        "adapt_lr_start": args.adapt_lr_start,
        "adapt_lr_end": args.adapt_lr_end,
        "adapt_lr_warmup_steps": args.adapt_lr_warmup_steps,
        "adapt_max_steps": args.adapt_max_steps,
        "adapt_batch_size": args.adapt_batch_size,
        "eval_every_steps": args.eval_every_steps,
        "eval_at_epochs": args.eval_at_epochs,
        "update_split": args.update_split,
        "observe_split": args.observe_split,
        "mmlu_subjects": mmlu_subjects,
        "max_update_samples": args.max_update_samples,
        "max_update_tokens_per_sample": args.max_update_tokens_per_sample,
        "max_observe_samples_per_domain": args.max_observe_samples_per_domain,
        "max_observe_samples": args.max_observe_samples,
        "max_observe_tokens_per_sample": args.max_observe_tokens_per_sample,
        "max_pairs": args.max_pairs,
        "validation_lr": args.validation_lr,
        "seed": args.seed,
        "pair_seed": args.pair_seed,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "local_files_only": local_files_only,
        "num_adapt_samples": len(adapt_samples),
        "num_update_samples": len(update_samples),
        "num_observe_samples": len(observe_samples),
        "num_update_tokens": len(update_tokens),
        "num_observe_tokens": len(observe_tokens),
        "num_pairs": len(pairs),
        "param_scope_metadata": param_scope_metadata,
        "dtype_report_initial": model_dtype_report(model),
    }
    save_json(args.output_dir / "run_config.json", run_config)

    all_rows = []
    metric_rows = []
    support_rows = []

    def evaluate_checkpoint(checkpoint, metadata):
        print(f"Running Section 3 validation from evaluation point: {checkpoint}")
        checkpoint_dir = args.output_dir / checkpoint / "section3_validation"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        val_args = make_validation_args(args, checkpoint_dir)
        rows = run_section3(model, adapter.tokenizer, update_tokens, observe_tokens, pairs, val_args, param_scope_params)
        for row in rows:
            row["checkpoint"] = checkpoint
            row["adapt_step"] = metadata["step"]
            row["samples_seen"] = metadata["samples_seen"]
            row["epoch"] = metadata["epoch"]
        write_jsonl(checkpoint_dir / "section3_pairs.jsonl", rows)
        write_csv_rows(checkpoint_dir / "section3_pairs.csv", rows)
        all_rows.extend(rows)

        ckpt_df = pd.DataFrame(rows)
        for comparison, (x_col, y_col) in COMPARISONS.items():
            metric_rows.append({
                "checkpoint": checkpoint,
                "adapt_step": metadata["step"],
                "samples_seen": metadata["samples_seen"],
                "epoch": metadata["epoch"],
                "comparison": comparison,
                **metric_values(ckpt_df[x_col], ckpt_df[y_col]),
            })
        diag_df = compute_update_grad_norms(model, update_tokens, param_scope_params)
        support_rows.append({
            "checkpoint": checkpoint,
            "adapt_step": metadata["step"],
            "samples_seen": metadata["samples_seen"],
            "epoch": metadata["epoch"],
            "mean_update_token_logp": float(diag_df["update_logp"].mean()),
            "mean_update_grad_norm": float(diag_df["update_grad_norm"].mean()),
            "mean_abs_actual_delta_logp": float(ckpt_df["delta_logp"].abs().mean()),
            "mean_abs_first_order_exact": float(ckpt_df["first_order_exact"].abs().mean()),
        })
        diag_df.to_csv(checkpoint_dir / "update_token_diagnostics.csv", index=False)

    checkpoint_meta, losses = run_aux_sft(model, adapt_samples, args.output_dir, args, evaluate_checkpoint)

    rows_df = pd.DataFrame(all_rows)
    metrics_df = pd.DataFrame(metric_rows)
    support_df = pd.DataFrame(support_rows)
    rows_df.to_csv(args.output_dir / "all_checkpoint_section3_pairs.csv", index=False)
    metrics_df.to_csv(args.output_dir / "checkpoint_comparison_metrics.csv", index=False)
    support_df.to_csv(args.output_dir / "checkpoint_supporting_diagnostics.csv", index=False)
    checkpoint_summary = {"checkpoint_meta": checkpoint_meta, "loss_final": losses[-1]["loss"] if losses else None}
    save_json(args.output_dir / "checkpoint_summary.json", checkpoint_summary)
    plots = []
    plots.extend(plot_metrics(args.output_dir, metrics_df))
    plots.extend(plot_scatters(args.output_dir, rows_df))
    run_config["dtype_report_final"] = model_dtype_report(model)
    run_config["checkpoint_meta"] = checkpoint_meta
    run_config["plots"] = plots
    save_json(args.output_dir / "run_config.json", run_config)
    write_report(args.output_dir, run_config, metrics_df, support_df, plots)
    print(json.dumps({"output_dir": str(args.output_dir), "num_pairs": len(pairs), "plots": plots}, indent=2))


if __name__ == "__main__":
    main()
