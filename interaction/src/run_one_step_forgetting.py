import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config_read import load_config
from utils.hh_dataset_adapters import (
    DEFAULT_MMLU_SUBJECTS,
    make_hh_dataset_adapter,
    parse_csv_arg,
)
from utils.ch1_ch2_metrics import (
    compute_ch1_ch2_for_update_token,
    compute_ch1_from_factors,
    compute_ch2_backbone_pair,
    extract_ch_factors,
    token_logprob,
)
from utils.one_step_forgetting import (
    UpdateResultWriter,
    apply_single_token_update,
    clone_model_state,
    compute_forgetting_scores,
    eval_probe_samples,
    group_samples_by_domain,
    iter_supervised_positions,
    restore_model_state,
)
from utils.tracking_metrics import compute_off_policy_scores


SUPPORTED_UPDATE_DATASETS = ["gsm8k", "mmlu", "dolly"]
SUPPORTED_PROBE_DATASETS = ["gsm8k", "mmlu", "dolly"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run generalized one-step update forgetting analysis."
    )
    parser.add_argument("--config", type=str, default="interaction/configs/train_basic.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--update_dataset", type=str, default="gsm8k")
    parser.add_argument("--update_split", type=str, default="train")
    parser.add_argument("--probe_dataset", type=str, default="mmlu")
    parser.add_argument("--probe_split", type=str, default="test")
    parser.add_argument("--mmlu_subjects", type=str, default=None)

    parser.add_argument("--max_update_samples", type=int, default=1)
    parser.add_argument("--max_update_tokens_per_sample", type=int, default=1)
    parser.add_argument("--max_probe_samples_per_domain", type=int, default=1)
    parser.add_argument(
        "--max_probe_samples",
        type=int,
        default=None,
        help="Optional global cap on selected probe samples after per-domain sampling.",
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Optional Transformers device_map, e.g. 'auto' to shard a model across visible GPUs.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model load dtype. Defaults to Transformers auto dtype.",
    )

    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow Hugging Face model/dataset downloads. Defaults to local cache only.",
    )
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--debug_update_tokens", action="store_true")
    parser.add_argument("--debug_restore_check", action="store_true")
    parser.add_argument(
        "--compute_ch1_ch2",
        action="store_true",
        help="Compute slow ch1/ch2 diagnostics before each one-token update.",
    )
    parser.add_argument(
        "--ch_layer",
        type=str,
        default="-1",
        help="Hidden-state layer for ch1. Defaults to -1.",
    )
    parser.add_argument(
        "--max_ch_probe_tokens_per_sample",
        type=int,
        default=1,
        help="Maximum supervised probe tokens per probe sample for ch1/ch2 diagnostics.",
    )
    parser.add_argument(
        "--exclude_lm_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude lm_head/readout parameters from ch2 gradients. Defaults to true.",
    )
    parser.add_argument(
        "--manual_ch_debug",
        action="store_true",
        help="Run manual text ch1/ch2 debug mode and exit before dataset loading.",
    )
    parser.add_argument(
        "--manual_update_text",
        type=str,
        default=None,
        help="Hand-written update text for --manual_ch_debug.",
    )
    parser.add_argument(
        "--manual_observe_text",
        type=str,
        default=None,
        help="Hand-written observation text for --manual_ch_debug.",
    )
    parser.add_argument(
        "--manual_update_prompt",
        type=str,
        default="",
        help="Optional prompt prefix to ignore in update labels for --manual_ch_debug.",
    )
    parser.add_argument(
        "--manual_observe_prompt",
        type=str,
        default="",
        help="Optional prompt prefix to ignore in observation labels for --manual_ch_debug.",
    )
    parser.add_argument(
        "--max_manual_update_tokens",
        type=int,
        default=8,
        help="Maximum supervised update tokens scored in --manual_ch_debug.",
    )
    parser.add_argument(
        "--max_manual_observe_tokens",
        type=int,
        default=32,
        help="Maximum supervised observation tokens scored in --manual_ch_debug.",
    )
    return parser.parse_args()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def resolve_model_name(config, args):
    if args.model_name_or_path and args.model_name:
        raise ValueError("Use only one of --model_name or --model_name_or_path.")
    return args.model_name_or_path or args.model_name or config["model"]["name"]


def check_supported_args(args):
    if args.update_dataset not in SUPPORTED_UPDATE_DATASETS:
        raise NotImplementedError(
            "Supported update datasets are "
            f"{SUPPORTED_UPDATE_DATASETS}. "
            f"Got update_dataset={args.update_dataset!r}."
        )
    if args.probe_dataset not in SUPPORTED_PROBE_DATASETS:
        raise NotImplementedError(
            "Supported probe datasets are "
            f"{SUPPORTED_PROBE_DATASETS}. "
            f"Got probe_dataset={args.probe_dataset!r}."
        )
    for name in [
        "max_update_samples",
        "max_update_tokens_per_sample",
        "max_probe_samples_per_domain",
        "max_ch_probe_tokens_per_sample",
        "max_manual_update_tokens",
        "max_manual_observe_tokens",
    ]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative.")
    if args.max_probe_samples is not None and args.max_probe_samples < 0:
        raise ValueError("--max_probe_samples must be non-negative.")
    if args.manual_ch_debug:
        if not args.manual_update_text:
            raise ValueError("--manual_update_text is required with --manual_ch_debug.")
        if not args.manual_observe_text:
            raise ValueError("--manual_observe_text is required with --manual_ch_debug.")


def load_adapter_samples(
    adapter_name,
    split,
    model_name,
    max_length,
    config,
    local_files_only,
    mmlu_subjects=None,
):
    try:
        adapter = make_hh_dataset_adapter(
            adapter_name=adapter_name,
            model_name_or_path=model_name,
            max_length=max_length,
            config=config,
            local_files_only=local_files_only,
            mmlu_subjects=mmlu_subjects,
        )
        dataset = adapter.load_dataset(split=split, local_files_only=local_files_only)
        return adapter.process_dataset(dataset), adapter
    except OSError as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        print(
            "\nFailed to load dataset/tokenizer from Hugging Face "
            f"({mode}).\n"
            "By default this runner avoids network downloads. If this model "
            "and dataset are not cached locally, rerun with --allow_download "
            "only on a machine where downloads are approved.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def select_probe_samples(samples, max_per_domain, seed, max_total=None):
    grouped = group_samples_by_domain(samples)
    rng = random.Random(seed)
    selected = []

    for domain in sorted(grouped):
        domain_samples = list(grouped[domain])
        if max_per_domain is None or max_per_domain >= len(domain_samples):
            chosen = domain_samples
        else:
            indices = sorted(rng.sample(range(len(domain_samples)), k=max_per_domain))
            chosen = [domain_samples[idx] for idx in indices]
        selected.extend(chosen)

    if max_total is not None and max_total < len(selected):
        indices = sorted(rng.sample(range(len(selected)), k=max_total))
        selected = [selected[idx] for idx in indices]

    selected_by_domain = group_samples_by_domain(selected)

    return selected, selected_by_domain


def safe_mean(values_by_domain):
    values = [value for value in values_by_domain.values() if value is not None]
    return sum(values) / len(values) if values else None


def sample_identifier(sample, fallback):
    for key in ("example_id", "sample_id", "raw_index"):
        if key in sample:
            return sample[key]
    return fallback


def token_text(tokenizer, token_id):
    if tokenizer is None:
        return None
    return tokenizer.decode([int(token_id)])


def load_tokenizer(model_name, local_files_only):
    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
            padding_side="right",
            truncation_side="right",
        )
    except OSError as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        print(
            "\nFailed to load tokenizer from Hugging Face "
            f"({mode}).\n"
            "By default this runner avoids network downloads. If this tokenizer "
            "is not cached locally, rerun with --allow_download only on a machine "
            "where downloads are approved.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def build_manual_sample(tokenizer, full_text, prompt_text="", max_length=None):
    tokenized = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=max_length is not None,
        max_length=max_length,
    )
    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask[0]

    labels = input_ids.clone()
    labels[0] = -100
    if prompt_text:
        prompt_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=max_length is not None,
            max_length=max_length,
        )["input_ids"][0]
        prompt_len = min(int(prompt_ids.numel()), int(labels.numel()))
        labels[:prompt_len] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def manual_token_row(prefix, sample, label_pos, tokenizer):
    context_pos = int(label_pos) - 1
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    attention_mask = sample.get("attention_mask")
    context_token_id = int(input_ids[context_pos].item())
    target_token_id = int(labels[int(label_pos)].item())
    row = {
        f"{prefix}_label_pos": int(label_pos),
        f"{prefix}_logit_pos": context_pos,
        f"{prefix}_context_token_id": context_token_id,
        f"{prefix}_context_token": token_text(tokenizer, context_token_id),
        f"{prefix}_target_id": target_token_id,
        f"{prefix}_target_token": token_text(tokenizer, target_token_id),
    }
    if attention_mask is not None:
        row[f"{prefix}_context_attention_mask"] = int(attention_mask[context_pos].item())
        row[f"{prefix}_target_attention_mask"] = int(attention_mask[int(label_pos)].item())
    return row


def run_manual_ch_debug(model, tokenizer, args, output_dir, config, model_name, local_files_only):
    max_length = config["model"].get("max_length")
    update_sample = build_manual_sample(
        tokenizer,
        full_text=args.manual_update_text,
        prompt_text=args.manual_update_prompt,
        max_length=max_length,
    )
    observe_sample = build_manual_sample(
        tokenizer,
        full_text=args.manual_observe_text,
        prompt_text=args.manual_observe_prompt,
        max_length=max_length,
    )

    update_positions = list(
        iter_supervised_positions(
            update_sample,
            max_positions=args.max_manual_update_tokens,
        )
    )
    observe_positions = list(
        iter_supervised_positions(
            observe_sample,
            max_positions=args.max_manual_observe_tokens,
        )
    )
    if not update_positions:
        raise ValueError("Manual update text produced no supervised token positions.")
    if not observe_positions:
        raise ValueError("Manual observation text produced no supervised token positions.")

    update_factors = extract_ch_factors(
        model,
        update_sample,
        layer=args.ch_layer,
        label_positions=update_positions,
    )
    observe_factors = extract_ch_factors(
        model,
        observe_sample,
        layer=args.ch_layer,
        label_positions=observe_positions,
    )
    ch1_values, g_dot_values, h_dot_values = compute_ch1_from_factors(
        observe_factors,
        update_factors,
        return_parts=True,
    )
    obs_norm = observe_factors["hidden"].float().norm(dim=-1, keepdim=True)
    upd_norm = update_factors["hidden"].float().norm(dim=-1, keepdim=True)
    h_cos_values = h_dot_values / (obs_norm.clamp_min(1e-12) * upd_norm.T.clamp_min(1e-12))
    h_cos_values = h_cos_values.clamp(min=-1.0, max=1.0)

    rows = []
    for obs_idx, obs_label_pos in enumerate(observe_positions):
        for upd_idx, upd_label_pos in enumerate(update_positions):
            ch1 = float(ch1_values[obs_idx, upd_idx].item())
            ch2_tensor = compute_ch2_backbone_pair(
                model=model,
                obs_sample=observe_sample,
                obs_label_pos=obs_label_pos,
                update_sample=update_sample,
                update_label_pos=upd_label_pos,
                exclude_lm_head=args.exclude_lm_head,
            )
            ch2 = float(ch2_tensor.item())
            row = {
                "pair_index": len(rows),
                "model_name_or_path": model_name,
                "observe_text": args.manual_observe_text,
                "update_text": args.manual_update_text,
                "observe_prompt": args.manual_observe_prompt,
                "update_prompt": args.manual_update_prompt,
                "layer": int(observe_factors["layer"]),
                "g_dot": float(g_dot_values[obs_idx, upd_idx].item()),
                "h_dot": float(h_dot_values[obs_idx, upd_idx].item()),
                "h_cos": float(h_cos_values[obs_idx, upd_idx].item()),
                "ch1": ch1,
                "ch2": ch2,
                "ch1_plus_ch2": ch1 + ch2,
                "exclude_lm_head": bool(args.exclude_lm_head),
                "position_policy": "causal_lm_next_token: logit_pos = label_pos - 1",
            }
            row.update(manual_token_row("observe", observe_sample, obs_label_pos, tokenizer))
            row.update(manual_token_row("update", update_sample, upd_label_pos, tokenizer))
            rows.append(row)

    run_config = {
        "model_name_or_path": model_name,
        "manual_ch_debug": True,
        "manual_update_text": args.manual_update_text,
        "manual_observe_text": args.manual_observe_text,
        "manual_update_prompt": args.manual_update_prompt,
        "manual_observe_prompt": args.manual_observe_prompt,
        "max_manual_update_tokens": args.max_manual_update_tokens,
        "max_manual_observe_tokens": args.max_manual_observe_tokens,
        "ch_layer": args.ch_layer,
        "exclude_lm_head": args.exclude_lm_head,
        "device": str(next(model.parameters()).device),
        "local_files_only": local_files_only,
        "num_update_positions": len(update_positions),
        "num_observe_positions": len(observe_positions),
        "num_pairs": len(rows),
        "update_positions": update_positions,
        "observe_positions": observe_positions,
        "output_files": [
            "resolved_config.yaml",
            "manual_ch_debug_config.json",
            "manual_ch_debug.jsonl",
            "manual_ch_debug.csv",
            "summary.json",
        ],
    }
    write_yaml(output_dir / "resolved_config.yaml", config)
    write_json(output_dir / "manual_ch_debug_config.json", run_config)
    write_jsonl(output_dir / "manual_ch_debug.jsonl", rows)
    write_csv_rows(output_dir / "manual_ch_debug.csv", rows)
    write_json(output_dir / "summary.json", run_config)

    print(f"Saved manual ch debug outputs to {output_dir}")
    print(f"num_update_positions: {len(update_positions)}")
    print(f"num_observe_positions: {len(observe_positions)}")
    print(f"num_pairs: {len(rows)}")


def make_update_debug_row(sample, sample_index, pos, tokenizer):
    label_id = int(sample["labels"][pos].item())
    return {
        "sample_index": sample_index,
        "sample_id": sample_identifier(sample, sample_index),
        "dataset": sample.get("dataset"),
        "domain": sample.get("domain"),
        "example_id": sample.get("example_id"),
        "raw_index": sample.get("raw_index"),
        "token_pos": pos,
        "target_token_id": label_id,
        "target_token_str": token_text(tokenizer, label_id),
        "position_policy": "causal_lm_next_token: labels[pos] scored by logits[pos - 1]",
    }


def compute_tracking_scores(model, sample, pos, gamma, warn_state):
    if warn_state["disabled"]:
        return None, None, None, None
    try:
        return compute_off_policy_scores(model, sample, pos, gamma=gamma)
    except Exception as exc:
        warn_state["disabled"] = True
        print(
            "Warning: compute_off_policy_scores failed once; writing None for "
            f"SOP/AOP/entropy for this run. Original error: {exc}",
            file=sys.stderr,
        )
        return None, None, None, None


def resolve_torch_dtype(dtype_name):
    if dtype_name in (None, "auto"):
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def load_model(model_name, local_files_only, device, device_map=None, torch_dtype="auto"):
    try:
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "torch_dtype": resolve_torch_dtype(torch_dtype),
        }
        if device_map:
            load_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not device_map:
            model = model.to(device)
        model.eval()
        return model
    except OSError as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        print(
            "\nFailed to load model from Hugging Face "
            f"({mode}).\n"
            "By default this runner avoids network downloads. If the model is "
            "not cached locally, rerun with --allow_download only on a machine "
            "where downloads are approved.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def run_updates(
    model,
    optimizer,
    tokenizer,
    update_samples,
    probe_samples,
    probe_by_domain,
    before_by_id,
    writer,
    args,
):
    device = next(model.parameters()).device
    base_state = clone_model_state(model)
    debug_rows = []
    ch1_ch2_rows = []
    warn_state = {"disabled": False}
    restore_check_done = False
    n_update_samples = min(args.max_update_samples, len(update_samples))

    for sample_index in range(n_update_samples):
        sample = update_samples[sample_index]
        positions = list(
            iter_supervised_positions(
                sample,
                max_positions=args.max_update_tokens_per_sample,
            )
        )

        for pos in positions:
            restore_model_state(model, base_state)
            current_ch_rows = []
            sop, aop_l2, aop_kl, entropy = compute_tracking_scores(
                model=model,
                sample=sample,
                pos=pos,
                gamma=args.gamma,
                warn_state=warn_state,
            )

            update_id = f"{sample_index}_{pos}"
            if args.compute_ch1_ch2:
                rows = compute_ch1_ch2_for_update_token(
                    model=model,
                    update_sample=sample,
                    update_label_pos=pos,
                    probe_samples=probe_samples,
                    layer=args.ch_layer,
                    max_probe_tokens_per_sample=args.max_ch_probe_tokens_per_sample,
                    exclude_lm_head=args.exclude_lm_head,
                )
                for row in rows:
                    row.update(
                        {
                            "update_id": update_id,
                            "update_sample_index": sample_index,
                            "update_sample_id": sample_identifier(sample, sample_index),
                            "update_dataset": sample.get("dataset", args.update_dataset),
                            "update_domain": sample.get("domain"),
                            "update_example_id": sample.get("example_id"),
                            "update_raw_index": sample.get("raw_index"),
                        }
                    )
                current_ch_rows = rows

            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)
            loss_val = apply_single_token_update(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pos=pos,
            )

            _, after_by_id = eval_probe_samples(model, probe_samples)

            if args.debug_restore_check and not restore_check_done:
                restore_model_state(model, base_state)
                _, restored_by_id = eval_probe_samples(model, probe_samples)
                max_abs_diff = 0.0
                for probe_id, before_row in before_by_id.items():
                    restored_logp = restored_by_id[probe_id]["total_logp"]
                    diff = abs(before_row["total_logp"] - restored_logp)
                    max_abs_diff = max(max_abs_diff, diff)
                print(
                    "debug_restore_check: "
                    f"update_id={sample_index}_{pos} "
                    f"num_probes={len(before_by_id)} "
                    f"max_abs_total_logp_diff={max_abs_diff:.12g}"
                )
                restore_model_state(model, base_state)
                restore_check_done = True

            f_avg_by_domain, f_ratio_by_domain = compute_forgetting_scores(
                before_by_id=before_by_id,
                after_by_id=after_by_id,
                probe_by_domain=probe_by_domain,
                tau=args.tau,
            )

            label_id = int(sample["labels"][pos].item())
            update_extra = {
                "loss": loss_val,
                "update_dataset": sample.get("dataset", args.update_dataset),
                "update_domain": sample.get("domain"),
                "update_example_id": sample.get("example_id"),
                "update_raw_index": sample.get("raw_index"),
                "target_token_id": label_id,
                "target_token_str": token_text(tokenizer, label_id),
            }
            pair_extra = {
                "algo": "sft",
                "sample_id": sample_identifier(sample, sample_index),
                "token_pos": pos,
                "sop": sop,
                "entropy": entropy,
                "aop_l2": aop_l2,
                "aop_kl": aop_kl,
                "f_avg_mean": safe_mean(f_avg_by_domain),
                "f_ratio_mean": safe_mean(f_ratio_by_domain),
                **update_extra,
            }
            for domain in writer.domains:
                pair_extra[f"f_avg_{domain}"] = f_avg_by_domain.get(domain)
                pair_extra[f"f_ratio_{domain}"] = f_ratio_by_domain.get(domain)
            for row in current_ch_rows:
                probe_logp_after = token_logprob(
                    model=model,
                    sample=probe_samples[int(row["probe_index"])],
                    logit_pos=int(row["probe_logit_pos"]),
                    target_id=int(row["probe_target_id"]),
                    device=device,
                )
                probe_logp_after = float(probe_logp_after.detach().float().cpu().item())
                probe_logp_before = row.get("probe_logp_before")
                delta_logp_actual = probe_logp_after - probe_logp_before
                row.update(
                    {
                        "probe_logp_after": probe_logp_after,
                        "delta_logp_actual": delta_logp_actual,
                        "f_actual": -delta_logp_actual,
                    }
                )
                row.update(pair_extra)
            ch1_ch2_rows.extend(current_ch_rows)

            writer.add_update(
                update_id=update_id,
                algo="sft",
                sample_id=sample_identifier(sample, sample_index),
                token_pos=pos,
                sop=sop,
                entropy=entropy,
                aop_l2=aop_l2,
                aop_kl=aop_kl,
                f_avg_by_domain=f_avg_by_domain,
                f_ratio_by_domain=f_ratio_by_domain,
                extra=update_extra,
            )

            if args.debug_update_tokens:
                debug_rows.append(make_update_debug_row(sample, sample_index, pos, tokenizer))

    restore_model_state(model, base_state)
    return debug_rows, ch1_ch2_rows

def main():
    args = parse_args()
    check_supported_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    model_name = resolve_model_name(config, args)
    max_length = config["model"]["max_length"]
    local_files_only = not args.allow_download
    mmlu_subjects = parse_csv_arg(args.mmlu_subjects) or DEFAULT_MMLU_SUBJECTS
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manual_ch_debug:
        tokenizer = load_tokenizer(model_name, local_files_only=local_files_only)
        model = load_model(
            model_name=model_name,
            local_files_only=local_files_only,
            device=device,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
        )
        run_manual_ch_debug(
            model=model,
            tokenizer=tokenizer,
            args=args,
            output_dir=output_dir,
            config=config,
            model_name=model_name,
            local_files_only=local_files_only,
        )
        return

    run_config = {
        "model_name_or_path": model_name,
        "update_dataset": args.update_dataset,
        "update_split": args.update_split,
        "probe_dataset": args.probe_dataset,
        "probe_split": args.probe_split,
        "mmlu_subjects": mmlu_subjects,
        "max_update_samples": args.max_update_samples,
        "max_update_tokens_per_sample": args.max_update_tokens_per_sample,
        "max_probe_samples_per_domain": args.max_probe_samples_per_domain,
        "max_probe_samples": args.max_probe_samples,
        "lr": args.lr,
        "gamma": args.gamma,
        "tau": args.tau,
        "seed": args.seed,
        "device": device,
        "device_map": args.device_map,
        "torch_dtype": args.torch_dtype,
        "local_files_only": local_files_only,
        "save_csv": args.save_csv,
        "debug_update_tokens": args.debug_update_tokens,
        "debug_restore_check": args.debug_restore_check,
        "compute_ch1_ch2": args.compute_ch1_ch2,
        "ch_layer": args.ch_layer,
        "max_ch_probe_tokens_per_sample": args.max_ch_probe_tokens_per_sample,
        "exclude_lm_head": args.exclude_lm_head,
    }
    write_yaml(output_dir / "resolved_config.yaml", config)
    write_json(output_dir / "run_config.json", run_config)

    update_samples, update_adapter = load_adapter_samples(
        adapter_name=args.update_dataset,
        split=args.update_split,
        model_name=model_name,
        max_length=max_length,
        config=config,
        local_files_only=local_files_only,
    )
    probe_samples_all, _ = load_adapter_samples(
        adapter_name=args.probe_dataset,
        split=args.probe_split,
        model_name=model_name,
        max_length=max_length,
        config=config,
        local_files_only=local_files_only,
        mmlu_subjects=mmlu_subjects,
    )
    probe_samples, probe_by_domain = select_probe_samples(
        samples=probe_samples_all,
        max_per_domain=args.max_probe_samples_per_domain,
        seed=args.seed,
        max_total=args.max_probe_samples,
    )
    if not update_samples:
        raise ValueError("No update samples were produced.")
    if not probe_samples:
        raise ValueError("No probe samples were selected.")

    model = load_model(
        model_name=model_name,
        local_files_only=local_files_only,
        device=device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    before_rows, before_by_id = eval_probe_samples(model, probe_samples)
    write_jsonl(output_dir / "probe_before.jsonl", before_rows)

    writer = UpdateResultWriter(domains=sorted(probe_by_domain.keys()))
    debug_rows, ch1_ch2_rows = run_updates(
        model=model,
        optimizer=optimizer,
        tokenizer=update_adapter.tokenizer,
        update_samples=update_samples,
        probe_samples=probe_samples,
        probe_by_domain=probe_by_domain,
        before_by_id=before_by_id,
        writer=writer,
        args=args,
    )
    writer.save(output_dir, config=run_config, save_csv=args.save_csv)

    if args.debug_update_tokens:
        write_jsonl(output_dir / "debug_update_tokens.jsonl", debug_rows)
    if args.compute_ch1_ch2:
        write_jsonl(output_dir / "ch1_ch2_debug.jsonl", ch1_ch2_rows)
        write_csv_rows(output_dir / "ch1_ch2_pairs.csv", ch1_ch2_rows)

    summary = {
        "num_update_samples_loaded": len(update_samples),
        "num_probe_samples_loaded": len(probe_samples_all),
        "num_probe_samples_selected": len(probe_samples),
        "probe_counts_by_domain": {
            domain: len(items) for domain, items in sorted(probe_by_domain.items())
        },
        "num_updates_written": len(writer.summary_rows),
        "output_files": [
            "resolved_config.yaml",
            "run_config.json",
            "probe_before.jsonl",
            "updates_summary.parquet",
            "updates_by_domain.parquet",
        ],
    }
    if args.save_csv:
        summary["output_files"].extend(["updates_summary.csv", "updates_by_domain.csv"])
    if args.debug_update_tokens:
        summary["output_files"].append("debug_update_tokens.jsonl")
    if args.compute_ch1_ch2:
        summary["output_files"].append("ch1_ch2_debug.jsonl")
        summary["output_files"].append("ch1_ch2_pairs.csv")
        summary["num_ch1_ch2_rows_written"] = len(ch1_ch2_rows)
    write_json(output_dir / "summary.json", summary)

    print(f"Saved one-step forgetting outputs to {output_dir}")
    print(f"num_updates_written: {len(writer.summary_rows)}")
    print(f"probe_domains: {sorted(probe_by_domain.keys())}")


if __name__ == "__main__":
    main()
