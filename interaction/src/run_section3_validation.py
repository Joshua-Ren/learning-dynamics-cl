import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM

from utils.ch1_ch2_metrics import (
    compute_ch1_from_factors,
    compute_ch12_real_pair,
    compute_ch2_approx_variants_from_factors,
    compute_ch2_backbone_pair,
    get_readout_weight,
    grad_inner_product,
    extract_block_inputs_wo_rms,
    extract_ch_factors,
    extract_normalized_attn_inputs,
    extract_normalized_block_inputs,
    token_logprob,
)
from utils.config_read import load_config
from utils.hh_dataset_adapters import (
    DEFAULT_MMLU_SUBJECTS,
    make_hh_dataset_adapter,
    parse_csv_arg,
)
from utils.one_step_forgetting import (
    apply_single_token_update,
    clone_model_state,
    group_samples_by_domain,
    iter_supervised_positions,
    restore_model_state,
)


SUPPORTED_DATASETS = ["gsm8k", "mmlu", "dolly"]
PARAM_SCOPES = ["all", "modeled", "readout_only"]
READOUT_SANITY_MODES = ["none", "functional_untied", "tied_decomp"]

READOUT_NAME_MARKERS = ("lm_head", "embed_out", "output_projection")
EMBED_NAME_MARKERS = ("embed_tokens", "wte", "word_embeddings")
NORM_NAME_MARKERS = ("norm", "ln_", "layernorm", "layer_norm")
MODELED_PROJECTION_MARKERS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "c_attn",
    "c_proj",
    "c_fc",
    "dense",
    "fc",
    "wq",
    "wk",
    "wv",
    "wo",
    "w1",
    "w2",
    "w3",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Section 3 approximation validation on update-observe token pairs."
    )
    parser.add_argument("--config", type=str, default="interaction/configs/train_basic.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--update_dataset", type=str, default="gsm8k", choices=SUPPORTED_DATASETS)
    parser.add_argument("--update_split", type=str, default="train")
    parser.add_argument("--observe_dataset", type=str, default="mmlu", choices=SUPPORTED_DATASETS)
    parser.add_argument("--observe_split", type=str, default="test")
    parser.add_argument("--mmlu_subjects", type=str, default=None)

    parser.add_argument("--max_update_samples", type=int, default=1)
    parser.add_argument("--max_update_tokens_per_sample", type=int, default=1)
    parser.add_argument("--max_observe_samples_per_domain", type=int, default=1)
    parser.add_argument("--max_observe_samples", type=int, default=None)
    parser.add_argument("--max_observe_tokens_per_sample", type=int, default=1)
    parser.add_argument("--max_pairs", type=int, default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_seed", type=int, default=None)
    parser.add_argument("--ch_layer", type=str, default="-1")
    parser.add_argument(
        "--param_scope",
        type=str,
        default="all",
        choices=PARAM_SCOPES,
        help="Parameter scope for exact first-order interaction and one-step update.",
    )
    parser.add_argument(
        "--readout_sanity",
        type=str,
        default="none",
        choices=READOUT_SANITY_MODES,
        help="Optional algebraic readout sanity diagnostic independent of tied input embeddings.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float32",
        choices=["auto", "float32", "float64", "float16", "bfloat16"],
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        help="Attention implementation passed to from_pretrained; default eager for precision runs.",
    )
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow Hugging Face model/dataset downloads. Defaults to local cache only.",
    )
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--debug_alignment", action="store_true")
    parser.add_argument("--debug_alignment_rows", type=int, default=10)
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
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def resolve_model_name(config, args):
    if args.model_name and args.model_name_or_path:
        raise ValueError("Use only one of --model_name or --model_name_or_path.")
    return args.model_name_or_path or args.model_name or config["model"]["name"]


def resolve_torch_dtype(dtype_name):
    if dtype_name in (None, "auto"):
        return "auto"
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def configure_precision():
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def model_dtype_report(model):
    param_counts = {}
    buffer_counts = {}
    for param in model.parameters():
        param_counts[str(param.dtype)] = param_counts.get(str(param.dtype), 0) + param.numel()
    for buffer in model.buffers():
        if torch.is_floating_point(buffer):
            buffer_counts[str(buffer.dtype)] = buffer_counts.get(str(buffer.dtype), 0) + buffer.numel()
    return {
        "parameter_dtypes": param_counts,
        "floating_buffer_dtypes": buffer_counts,
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None,
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def assert_requested_precision(model, torch_dtype):
    requested = resolve_torch_dtype(torch_dtype)
    if requested == "auto":
        return
    bad = []
    for name, param in model.named_parameters():
        if param.is_floating_point() and param.dtype != requested:
            bad.append((name, str(param.dtype)))
            if len(bad) >= 5:
                break
    if bad:
        details = ", ".join(f"{name}:{dtype}" for name, dtype in bad)
        raise RuntimeError(f"Model precision check failed for requested {requested}: {details}")


def _name_has_any(name, markers):
    lower = name.lower()
    return any(marker in lower for marker in markers)


def _is_readout_param(name, param, readout_weight):
    if _name_has_any(name, READOUT_NAME_MARKERS):
        return True
    return readout_weight is not None and param is readout_weight


def _is_modeled_projection_param(name, param):
    if param.dim() < 2:
        return False
    if _name_has_any(name, EMBED_NAME_MARKERS):
        return False
    if _name_has_any(name, NORM_NAME_MARKERS):
        return False
    return _name_has_any(name, MODELED_PROJECTION_MARKERS)


def detect_tied_readout(model):
    try:
        readout_weight = get_readout_weight(model)
    except AttributeError:
        return {"has_readout": False, "is_tied": None, "readout_shape": None}

    input_embeddings = None
    if hasattr(model, "get_input_embeddings"):
        emb_module = model.get_input_embeddings()
        input_embeddings = getattr(emb_module, "weight", None)

    is_tied = None
    if input_embeddings is not None:
        is_tied = readout_weight.data_ptr() == input_embeddings.data_ptr()

    return {
        "has_readout": True,
        "is_tied": bool(is_tied) if is_tied is not None else None,
        "readout_shape": list(readout_weight.shape),
        "input_embedding_shape": list(input_embeddings.shape) if input_embeddings is not None else None,
    }


def resolve_param_scope(model, scope):
    try:
        readout_weight = get_readout_weight(model)
    except AttributeError:
        readout_weight = None

    selected = []
    selected_names = []
    categories = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        include = False
        category = None
        if scope == "all":
            include = True
            category = "all_trainable"
        elif scope == "readout_only":
            include = _is_readout_param(name, param, readout_weight)
            category = "readout" if include else None
        elif scope == "modeled":
            if _is_readout_param(name, param, readout_weight):
                include = True
                category = "readout_ch1"
            elif _is_modeled_projection_param(name, param):
                include = True
                category = "transformer_projection_ch2"
        else:
            raise ValueError(f"Unknown param_scope: {scope}")

        if include:
            selected.append(param)
            selected_names.append(name)
            categories[category] = categories.get(category, 0) + param.numel()

    if not selected:
        raise ValueError(f"param_scope={scope!r} selected no trainable parameters.")

    all_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    selected_total = sum(param.numel() for param in selected)
    tied_readout = detect_tied_readout(model)
    metadata = {
        "scope": scope,
        "num_tensors": len(selected),
        "num_parameters": selected_total,
        "total_trainable_parameters": all_trainable,
        "fraction_trainable_parameters": selected_total / all_trainable if all_trainable else None,
        "categories_num_parameters": categories,
        "parameter_names": selected_names,
        "tied_readout": tied_readout,
        "readout_sanity_valid": not (scope == "readout_only" and tied_readout.get("is_tied") is True),
        "notes": {
            "all": "All trainable parameters used by the model.",
            "readout_only": "Parameters whose name/object corresponds to lm_head/embed_out/output_projection. If readout is tied, the shared tensor is reported in tied_readout.",
            "modeled": "Readout parameters represented by CH1 plus non-embedding, non-norm, rank>=2 transformer projection weights represented by the current CH2 approximation.",
        }[scope],
    }
    return selected, metadata


def compute_scoped_grad_interaction(model, obs_sample, obs_label_pos, update_sample, update_label_pos, params):
    model.eval()
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    obs_target = obs_sample["labels"][int(obs_label_pos)].to(device)
    update_target = update_sample["labels"][int(update_label_pos)].to(device)
    logp_o = token_logprob(
        model,
        obs_sample,
        logit_pos=int(obs_label_pos) - 1,
        target_id=obs_target,
        device=device,
    )
    logp_u = token_logprob(
        model,
        update_sample,
        logit_pos=int(update_label_pos) - 1,
        target_id=update_target,
        device=device,
    )
    score = grad_inner_product(logp_o, logp_u, params)
    model.zero_grad(set_to_none=True)
    return score.detach().cpu()


def _functional_readout_logp(weight, hidden, target_id):
    logits = hidden @ weight.T
    logp = F.log_softmax(logits, dim=-1)[target_id]
    return logits, logp


def compute_functional_untied_readout_sanity(model, observe_factors, update_factors):
    device = next(model.parameters()).device
    compute_dtype = get_readout_weight(model).dtype
    readout_weight = get_readout_weight(model).detach().to(device=device, dtype=compute_dtype)
    weight = readout_weight.clone().requires_grad_(True)

    h_o = observe_factors["hidden"][0].to(device=device, dtype=compute_dtype).detach()
    h_u = update_factors["hidden"][0].to(device=device, dtype=compute_dtype).detach()
    target_o = int(observe_factors["target_ids"][0].item())
    target_u = int(update_factors["target_ids"][0].item())

    logits_o, logp_o = _functional_readout_logp(weight, h_o, target_o)
    logits_u, logp_u = _functional_readout_logp(weight, h_u, target_u)
    model_logits_o = observe_factors["logits"][0].to(device=device, dtype=compute_dtype)
    model_logits_u = update_factors["logits"][0].to(device=device, dtype=compute_dtype)
    model_logp_o = observe_factors["log_probs"][0].to(device=device, dtype=compute_dtype)
    model_logp_u = update_factors["log_probs"][0].to(device=device, dtype=compute_dtype)

    grad_o = torch.autograd.grad(logp_o, weight, retain_graph=True, create_graph=False)[0]
    grad_u = torch.autograd.grad(logp_u, weight, retain_graph=False, create_graph=False)[0]
    raw = torch.sum(grad_o * grad_u).detach().cpu()

    return {
        "first_order_exact_readout_functional_raw": float(raw.item()),
        "functional_readout_logit_max_abs_error_observe": float((logits_o.detach() - model_logits_o).abs().max().cpu().item()),
        "functional_readout_logit_max_abs_error_update": float((logits_u.detach() - model_logits_u).abs().max().cpu().item()),
        "functional_readout_logp_abs_error_observe": float((logp_o.detach() - model_logp_o).abs().cpu().item()),
        "functional_readout_logp_abs_error_update": float((logp_u.detach() - model_logp_u).abs().cpu().item()),
    }


def _functional_readout_grad(model, factors):
    device = next(model.parameters()).device
    compute_dtype = get_readout_weight(model).dtype
    readout_weight = get_readout_weight(model).detach().to(device=device, dtype=compute_dtype)
    weight = readout_weight.clone().requires_grad_(True)
    hidden = factors["hidden"][0].to(device=device, dtype=compute_dtype).detach()
    target = int(factors["target_ids"][0].item())
    logits, logp = _functional_readout_logp(weight, hidden, target)
    grad = torch.autograd.grad(logp, weight, retain_graph=False, create_graph=False)[0].detach()
    return grad, logits.detach(), logp.detach()


def _shared_embedding_grad(model, sample, label_pos):
    model.eval()
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    shared_weight = get_readout_weight(model)
    target = sample["labels"][int(label_pos)].to(device)
    logp = token_logprob(
        model,
        sample,
        logit_pos=int(label_pos) - 1,
        target_id=target,
        device=device,
    )
    grad = torch.autograd.grad(logp, shared_weight, retain_graph=False, create_graph=False)[0].detach()
    model.zero_grad(set_to_none=True)
    return grad


def compute_tied_readout_decomposition(
    model,
    observe_sample,
    observe_pos,
    update_sample,
    update_pos,
    observe_factors,
    update_factors,
):
    tied = detect_tied_readout(model)
    if tied.get("is_tied") is not True:
        return {
            "tied_decomp_valid": False,
            "tied_decomp_reason": "readout is not tied to input embeddings",
        }

    out_o, logits_o, logp_o = _functional_readout_grad(model, observe_factors)
    out_u, logits_u, logp_u = _functional_readout_grad(model, update_factors)
    shared_o = _shared_embedding_grad(model, observe_sample, observe_pos)
    shared_u = _shared_embedding_grad(model, update_sample, update_pos)

    in_o = shared_o - out_o
    in_u = shared_u - out_u

    out_out = torch.sum(out_o * out_u)
    in_in = torch.sum(in_o * in_u)
    out_in = torch.sum(out_o * in_u)
    in_out = torch.sum(in_o * out_u)
    extra = in_in + out_in + in_out
    total = out_out + extra
    shared_total = torch.sum(shared_o * shared_u)

    model_logits_o = observe_factors["logits"][0].to(device=logits_o.device, dtype=logits_o.dtype)
    model_logits_u = update_factors["logits"][0].to(device=logits_u.device, dtype=logits_u.dtype)
    model_logp_o = observe_factors["log_probs"][0].to(device=logp_o.device, dtype=logp_o.dtype)
    model_logp_u = update_factors["log_probs"][0].to(device=logp_u.device, dtype=logp_u.dtype)

    recon_o = out_o + in_o
    recon_u = out_u + in_u
    return {
        "tied_decomp_valid": True,
        "tied_out_out": float(out_out.detach().cpu().item()),
        "tied_in_in": float(in_in.detach().cpu().item()),
        "tied_out_in": float(out_in.detach().cpu().item()),
        "tied_in_out": float(in_out.detach().cpu().item()),
        "tied_extra": float(extra.detach().cpu().item()),
        "tied_total": float(total.detach().cpu().item()),
        "tied_shared_direct": float(shared_total.detach().cpu().item()),
        "tied_shared_reconstruction_max_abs_error_observe": float((shared_o - recon_o).abs().max().cpu().item()),
        "tied_shared_reconstruction_max_abs_error_update": float((shared_u - recon_u).abs().max().cpu().item()),
        "tied_total_direct_abs_error": float((shared_total - total).abs().cpu().item()),
        "tied_functional_logit_max_abs_error_observe": float((logits_o - model_logits_o).abs().max().cpu().item()),
        "tied_functional_logit_max_abs_error_update": float((logits_u - model_logits_u).abs().max().cpu().item()),
        "tied_functional_logp_abs_error_observe": float((logp_o - model_logp_o).abs().cpu().item()),
        "tied_functional_logp_abs_error_update": float((logp_u - model_logp_u).abs().cpu().item()),
    }


def load_model(
    model_name,
    local_files_only,
    device,
    device_map=None,
    torch_dtype="auto",
    attn_implementation="eager",
):
    try:
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": local_files_only,
            "torch_dtype": resolve_torch_dtype(torch_dtype),
        }
        if device_map:
            load_kwargs["device_map"] = device_map
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not device_map:
            model = model.to(device)
        assert_requested_precision(model, torch_dtype)
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
            "By default this runner avoids network downloads. If this dataset "
            "or tokenizer is not cached locally, rerun with --allow_download "
            "only where downloads are approved.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def select_samples_by_domain(samples, max_per_domain, seed, max_total=None):
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
    return selected


def sample_id(sample, fallback):
    for key in ("example_id", "sample_id", "raw_index", "probe_id"):
        if key in sample:
            return sample[key]
    return fallback


def decode_token(tokenizer, token_id):
    return tokenizer.decode([int(token_id)]) if tokenizer is not None else None


def context_excerpt(sample, logit_pos, tokenizer, window=8):
    input_ids = sample["input_ids"]
    attention_mask = sample.get("attention_mask")
    left = max(0, int(logit_pos) - window + 1)
    right = int(logit_pos) + 1
    if attention_mask is not None:
        while left < right and int(attention_mask[left].item()) == 0:
            left += 1
    return tokenizer.decode(input_ids[left:right], skip_special_tokens=False)


def token_metadata(prefix, sample, sample_index, label_pos, tokenizer):
    logit_pos = int(label_pos) - 1
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    target_id = int(labels[int(label_pos)].item())
    context_id = int(input_ids[logit_pos].item())
    row = {
        f"{prefix}_sample_index": sample_index,
        f"{prefix}_sample_id": sample_id(sample, sample_index),
        f"{prefix}_dataset": sample.get("dataset"),
        f"{prefix}_domain": sample.get("domain"),
        f"{prefix}_example_id": sample.get("example_id"),
        f"{prefix}_raw_index": sample.get("raw_index"),
        f"{prefix}_label_pos": int(label_pos),
        f"{prefix}_logit_pos": logit_pos,
        f"{prefix}_context_token_id": context_id,
        f"{prefix}_context_token": decode_token(tokenizer, context_id),
        f"{prefix}_target_id": target_id,
        f"{prefix}_target_token": decode_token(tokenizer, target_id),
        f"{prefix}_context_excerpt": context_excerpt(sample, logit_pos, tokenizer),
    }
    attention_mask = sample.get("attention_mask")
    if attention_mask is not None:
        row[f"{prefix}_context_attention_mask"] = int(attention_mask[logit_pos].item())
        row[f"{prefix}_target_attention_mask"] = int(attention_mask[int(label_pos)].item())
    return row


def build_update_tokens(samples, max_samples, max_tokens_per_sample):
    tokens = []
    for sample_index, sample in enumerate(samples[:max_samples]):
        for label_pos in iter_supervised_positions(sample, max_positions=max_tokens_per_sample):
            tokens.append((sample_index, sample, int(label_pos)))
    return tokens


def build_observe_tokens(samples, max_tokens_per_sample):
    tokens = []
    for sample_index, sample in enumerate(samples):
        for label_pos in iter_supervised_positions(sample, max_positions=max_tokens_per_sample):
            tokens.append((sample_index, sample, int(label_pos)))
    return tokens


def build_pairs(update_tokens, observe_tokens, max_pairs, seed):
    pairs = [(u_idx, o_idx) for u_idx in range(len(update_tokens)) for o_idx in range(len(observe_tokens))]
    if max_pairs is None or max_pairs >= len(pairs):
        return pairs
    if max_pairs < 0:
        raise ValueError("--max_pairs must be non-negative or omitted.")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(pairs)), k=max_pairs))
    return [pairs[idx] for idx in indices]


def hoyer_sparsity(values):
    vals = torch.as_tensor(values, dtype=torch.float64).abs().flatten()
    n = int(vals.numel())
    if n <= 1:
        return 0.0
    l2 = vals.norm(p=2)
    if float(l2.item()) == 0.0:
        return 0.0
    l1_over_l2 = vals.sum() / l2
    return float(((n**0.5 - l1_over_l2) / (n**0.5 - 1.0)).item())


def density(values):
    return 1.0 - hoyer_sparsity(values)


def cumulative_abs_mass(values):
    vals = torch.as_tensor(values, dtype=torch.float64).abs().flatten()
    if vals.numel() == 0:
        return []
    sorted_vals = torch.sort(vals, descending=True).values
    total = sorted_vals.sum()
    if float(total.item()) == 0.0:
        return [0.0 for _ in sorted_vals.tolist()]
    return (torch.cumsum(sorted_vals, dim=0) / total).tolist()


def compute_pair_components(
    model,
    update_sample,
    update_pos,
    observe_sample,
    observe_pos,
    layer,
    param_scope_params,
    readout_sanity="none",
):
    update_factors = extract_ch_factors(
        model,
        update_sample,
        layer=layer,
        label_positions=[update_pos],
    )
    observe_factors = extract_ch_factors(
        model,
        observe_sample,
        layer=layer,
        label_positions=[observe_pos],
    )
    ch1_matrix, g_dot_matrix, h_dot_matrix = compute_ch1_from_factors(
        observe_factors,
        update_factors,
        return_parts=True,
    )

    update_block_inputs = extract_normalized_block_inputs(
        model,
        update_sample,
        label_positions=[update_pos],
        layer=layer,
    )
    observe_block_inputs = extract_normalized_block_inputs(
        model,
        observe_sample,
        label_positions=[observe_pos],
        layer=layer,
    )
    update_attn_inputs = extract_normalized_attn_inputs(
        model,
        update_sample,
        label_positions=[update_pos],
        layer=layer,
    )
    observe_attn_inputs = extract_normalized_attn_inputs(
        model,
        observe_sample,
        label_positions=[observe_pos],
        layer=layer,
    )
    update_block_inputs_wo_rms = extract_block_inputs_wo_rms(
        model,
        update_sample,
        label_positions=[update_pos],
        layer=layer,
    )
    observe_block_inputs_wo_rms = extract_block_inputs_wo_rms(
        model,
        observe_sample,
        label_positions=[observe_pos],
        layer=layer,
    )
    ch2_variants = compute_ch2_approx_variants_from_factors(
        model,
        observe_factors,
        update_factors,
        obs_block_inputs=observe_block_inputs,
        update_block_inputs=update_block_inputs,
        obs_attn_inputs=observe_attn_inputs,
        update_attn_inputs=update_attn_inputs,
        obs_block_inputs_wo_rms=observe_block_inputs_wo_rms,
        update_block_inputs_wo_rms=update_block_inputs_wo_rms,
    )
    ch2_exact_backbone = compute_ch2_backbone_pair(
        model=model,
        obs_sample=observe_sample,
        obs_label_pos=observe_pos,
        update_sample=update_sample,
        update_label_pos=update_pos,
        exclude_lm_head=True,
    )
    first_order_raw = compute_scoped_grad_interaction(
        model=model,
        obs_sample=observe_sample,
        obs_label_pos=observe_pos,
        update_sample=update_sample,
        update_label_pos=update_pos,
        params=param_scope_params,
    )

    functional_readout = {}
    if readout_sanity == "functional_untied":
        functional_readout = compute_functional_untied_readout_sanity(
            model,
            observe_factors=observe_factors,
            update_factors=update_factors,
        )
    elif readout_sanity == "tied_decomp":
        functional_readout = compute_tied_readout_decomposition(
            model,
            observe_sample=observe_sample,
            observe_pos=observe_pos,
            update_sample=update_sample,
            update_pos=update_pos,
            observe_factors=observe_factors,
            update_factors=update_factors,
        )

    return {
        "layer": int(update_factors["layer"]),
        "observe_logp_before": float(observe_factors["log_probs"][0].item()),
        "update_logp_before": float(update_factors["log_probs"][0].item()),
        "g_dot": float(g_dot_matrix[0, 0].item()),
        "h_dot": float(h_dot_matrix[0, 0].item()),
        "ch1": float(ch1_matrix[0, 0].item()),
        "ch2": float(ch2_variants["ch2_approx"][0, 0].item()),
        "ch2_exact_backbone": float(ch2_exact_backbone.item()),
        "first_order_exact_raw": float(first_order_raw.item()),
        "ch2_approx_gwwg": float(ch2_variants["ch2_approx_gwwg"][0, 0].item()),
        "ch2_approx_singleh": float(ch2_variants["ch2_approx_singleh"][0, 0].item()),
        "ch2_approx_wo_rms": float(ch2_variants["ch2_approx_wo_rms"][0, 0].item()),
        "ch2_approx_aggh": float(ch2_variants["ch2_approx_aggh"][0, 0].item()),
        **functional_readout,
    }


def compute_after_logp(model, observe_sample, observe_pos):
    device = next(model.parameters()).device
    labels = observe_sample["labels"]
    target_id = int(labels[int(observe_pos)].item())
    logp = token_logprob(
        model=model,
        sample=observe_sample,
        logit_pos=int(observe_pos) - 1,
        target_id=target_id,
        device=device,
    )
    return float(logp.detach().cpu().item())


def run_section3(model, tokenizer, update_tokens, observe_tokens, pairs, args, param_scope_params):
    device = next(model.parameters()).device
    base_state = clone_model_state(model)
    optimizer = torch.optim.SGD(param_scope_params, lr=args.lr)
    rows = []
    printed_debug = 0

    pairs_by_update = {}
    for pair_index, (update_token_index, observe_token_index) in enumerate(pairs):
        pairs_by_update.setdefault(update_token_index, []).append((pair_index, observe_token_index))

    try:
        for update_token_index in sorted(pairs_by_update):
            update_sample_index, update_sample, update_pos = update_tokens[update_token_index]
            restore_model_state(model, base_state)

            pending_rows = []
            for pair_index, observe_token_index in pairs_by_update[update_token_index]:
                observe_sample_index, observe_sample, observe_pos = observe_tokens[observe_token_index]
                components = compute_pair_components(
                    model=model,
                    update_sample=update_sample,
                    update_pos=update_pos,
                    observe_sample=observe_sample,
                    observe_pos=observe_pos,
                    layer=args.ch_layer,
                    param_scope_params=param_scope_params,
                    readout_sanity=args.readout_sanity,
                )
                ch1 = components["ch1"]
                ch2 = components["ch2"]
                row = {
                    "pair_index": pair_index,
                    "position_policy": "causal_lm_next_token: label_pos is scored by logits[label_pos - 1]",
                    "sign_scale_policy": (
                        "delta_logp = logp_after - logp_before; SGD uses loss=-logp_update, "
                        "so first_order_exact = lr * <grad logp_observe, grad logp_update>; "
                        "approx = lr * (ch1 + ch2)"
                    ),
                    "lr": args.lr,
                    "param_scope": args.param_scope,
                    **token_metadata(
                        "update",
                        update_sample,
                        update_sample_index,
                        update_pos,
                        tokenizer,
                    ),
                    **token_metadata(
                        "observe",
                        observe_sample,
                        observe_sample_index,
                        observe_pos,
                        tokenizer,
                    ),
                    **components,
                    "first_order_exact": args.lr * components["first_order_exact_raw"],
                    "approx_raw": ch1 + ch2,
                    "approx": args.lr * (ch1 + ch2),
                    "ch1_scaled": args.lr * ch1,
                    "ch2_scaled": args.lr * ch2,
                    f"first_order_exact_{args.param_scope}": args.lr * components["first_order_exact_raw"],
                    "readout_sanity_target": args.lr * ch1 if args.param_scope == "readout_only" else None,
                    "functional_readout_sanity_target": args.lr * ch1 if args.readout_sanity == "functional_untied" else None,
                    "first_order_exact_readout_functional": args.lr * components["first_order_exact_readout_functional_raw"] if args.readout_sanity == "functional_untied" else None,
                    "tied_out_out_scaled": args.lr * components["tied_out_out"] if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "tied_in_in_scaled": args.lr * components["tied_in_in"] if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "tied_out_in_scaled": args.lr * components["tied_out_in"] if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "tied_in_out_scaled": args.lr * components["tied_in_out"] if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "tied_extra_scaled": args.lr * components["tied_extra"] if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "tied_total_scaled": args.lr * components["tied_total"] if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "tied_sign_flip": (math.copysign(1.0, components["tied_total"]) != math.copysign(1.0, components["tied_out_out"])) if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") and components["tied_total"] != 0.0 and components["tied_out_out"] != 0.0 else None,
                    "tied_correction_raw": components["tied_extra"] if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "tied_correction_scaled": args.lr * components["tied_extra"] if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "approx_tied_raw": ch1 + ch2 + components["tied_extra"] if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "approx_tied": args.lr * (ch1 + ch2 + components["tied_extra"]) if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "approx_tied_readout_only_raw": components["tied_total"] if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "approx_tied_readout_only": args.lr * components["tied_total"] if args.readout_sanity == "tied_decomp" and components.get("tied_decomp_valid") else None,
                    "modeled_sanity_target": args.lr * (ch1 + ch2) if args.param_scope == "modeled" else None,
                }
                pending_rows.append((row, observe_sample, observe_pos))

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

            for row, observe_sample, observe_pos in pending_rows:
                observe_logp_after = compute_after_logp(model, observe_sample, observe_pos)
                observe_p_before = math.exp(row["observe_logp_before"])
                observe_p_after = math.exp(observe_logp_after)
                row["update_loss"] = update_loss
                row["observe_logp_after"] = observe_logp_after
                row["delta_logp"] = observe_logp_after - row["observe_logp_before"]
                row["observe_p_before"] = observe_p_before
                row["observe_p_after"] = observe_p_after
                row["delta_p"] = observe_p_after - observe_p_before
                rows.append(row)

                if args.debug_alignment and printed_debug < args.debug_alignment_rows:
                    print(
                        "alignment_debug "
                        f"pair_index={row['pair_index']} "
                        f"update_logit_pos={row['update_logit_pos']} "
                        f"update_context_token={row['update_context_token']!r} "
                        f"update_label_pos={row['update_label_pos']} "
                        f"update_target_token={row['update_target_token']!r} "
                        f"observe_logit_pos={row['observe_logit_pos']} "
                        f"observe_context_token={row['observe_context_token']!r} "
                        f"observe_label_pos={row['observe_label_pos']} "
                        f"observe_target_token={row['observe_target_token']!r}"
                    )
                    printed_debug += 1
    finally:
        restore_model_state(model, base_state)

    return sorted(rows, key=lambda row: row["pair_index"])


def summarize_rows(rows):
    ch1_values = [row["ch1"] for row in rows]
    ch2_values = [row["ch2"] for row in rows]
    return {
        "num_pairs": len(rows),
        "hoyer_sparsity_ch1": hoyer_sparsity(ch1_values),
        "density_ch1": density(ch1_values),
        "hoyer_sparsity_ch2": hoyer_sparsity(ch2_values),
        "density_ch2": density(ch2_values),
        "cumulative_abs_mass_ch1": cumulative_abs_mass(ch1_values),
        "cumulative_abs_mass_ch2": cumulative_abs_mass(ch2_values),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    configure_precision()
    config = load_config(args.config)
    model_name = resolve_model_name(config, args)
    max_length = config["model"]["max_length"]
    local_files_only = not args.allow_download
    mmlu_subjects = parse_csv_arg(args.mmlu_subjects) or DEFAULT_MMLU_SUBJECTS
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pair_seed = args.seed if args.pair_seed is None else args.pair_seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model_name_or_path": model_name,
        "update_dataset": args.update_dataset,
        "update_split": args.update_split,
        "observe_dataset": args.observe_dataset,
        "observe_split": args.observe_split,
        "mmlu_subjects": mmlu_subjects,
        "max_update_samples": args.max_update_samples,
        "max_update_tokens_per_sample": args.max_update_tokens_per_sample,
        "max_observe_samples_per_domain": args.max_observe_samples_per_domain,
        "max_observe_samples": args.max_observe_samples,
        "max_observe_tokens_per_sample": args.max_observe_tokens_per_sample,
        "max_pairs": args.max_pairs,
        "lr": args.lr,
        "seed": args.seed,
        "pair_seed": pair_seed,
        "ch_layer": args.ch_layer,
        "param_scope": args.param_scope,
        "readout_sanity": args.readout_sanity,
        "torch_dtype": args.torch_dtype,
        "device": device,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "local_files_only": local_files_only,
        "save_csv": args.save_csv,
        "debug_alignment": args.debug_alignment,
        "quantity_conventions": {
            "delta_logp": "observe_logp_after - observe_logp_before",
            "delta_p": "exp(observe_logp_after) - exp(observe_logp_before)",
            "first_order_exact": "lr * <grad logp_observe, grad logp_update>",
            "ch1": "<e_y_o - pi_o, e_y_u - pi_u> * <h_o, h_u>",
            "ch2": "forward-computable ch2_approx from Eq. 8-style readout/layer overlap",
            "approx": "lr * (ch1 + ch2)",
            "approx_tied": "lr * (ch1 + ch2 + tied_extra), where tied_extra is tied_in_in + tied_out_in + tied_in_out from tied_decomp",
        },
    }
    write_yaml(output_dir / "resolved_config.yaml", config)

    update_samples, update_adapter = load_adapter_samples(
        adapter_name=args.update_dataset,
        split=args.update_split,
        model_name=model_name,
        max_length=max_length,
        config=config,
        local_files_only=local_files_only,
        mmlu_subjects=mmlu_subjects,
    )
    observe_samples_all, _ = load_adapter_samples(
        adapter_name=args.observe_dataset,
        split=args.observe_split,
        model_name=model_name,
        max_length=max_length,
        config=config,
        local_files_only=local_files_only,
        mmlu_subjects=mmlu_subjects,
    )
    observe_samples = select_samples_by_domain(
        samples=observe_samples_all,
        max_per_domain=args.max_observe_samples_per_domain,
        seed=args.seed,
        max_total=args.max_observe_samples,
    )
    update_tokens = build_update_tokens(
        update_samples,
        max_samples=args.max_update_samples,
        max_tokens_per_sample=args.max_update_tokens_per_sample,
    )
    observe_tokens = build_observe_tokens(
        observe_samples,
        max_tokens_per_sample=args.max_observe_tokens_per_sample,
    )
    pairs = build_pairs(
        update_tokens=update_tokens,
        observe_tokens=observe_tokens,
        max_pairs=args.max_pairs,
        seed=pair_seed,
    )
    if not update_tokens:
        raise ValueError("No update tokens were selected.")
    if not observe_tokens:
        raise ValueError("No observation tokens were selected.")
    if not pairs:
        raise ValueError("No update-observation pairs were selected.")

    model = load_model(
        model_name=model_name,
        local_files_only=local_files_only,
        device=device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    dtype_report = model_dtype_report(model)
    param_scope_params, param_scope_metadata = resolve_param_scope(model, args.param_scope)
    run_config["dtype_report"] = dtype_report
    if args.param_scope == "readout_only" and param_scope_metadata.get("tied_readout", {}).get("is_tied") is True:
        print(
            "WARNING: param_scope=readout_only selected a tied input/output embedding. "
            "The actual optimizer update is not a pure CH1/readout update; "
            "treat readout_sanity_valid=false for this run.",
            file=sys.stderr,
        )
    run_config["param_scope_metadata"] = param_scope_metadata
    write_json(output_dir / "run_config.json", run_config)
    print(f"Precision report: {dtype_report}")

    rows = run_section3(
        model=model,
        tokenizer=update_adapter.tokenizer,
        update_tokens=update_tokens,
        observe_tokens=observe_tokens,
        pairs=pairs,
        args=args,
        param_scope_params=param_scope_params,
    )
    summary = summarize_rows(rows)
    summary.update(
        {
            "num_update_samples_loaded": len(update_samples),
            "num_observe_samples_loaded": len(observe_samples_all),
            "num_observe_samples_selected": len(observe_samples),
            "num_update_tokens": len(update_tokens),
            "num_observe_tokens": len(observe_tokens),
            "param_scope_metadata": param_scope_metadata,
            "output_files": [
                "resolved_config.yaml",
                "run_config.json",
                "section3_pairs.jsonl",
                "summary.json",
            ],
        }
    )
    if args.save_csv:
        summary["output_files"].append("section3_pairs.csv")

    write_jsonl(output_dir / "section3_pairs.jsonl", rows)
    if args.save_csv:
        write_csv_rows(output_dir / "section3_pairs.csv", rows)
    write_json(output_dir / "summary.json", summary)

    print(f"Saved Section 3 validation outputs to {output_dir}")
    print(f"num_pairs: {len(rows)}")
    print("main columns: delta_logp, delta_p, first_order_exact, ch1, ch2, approx")


if __name__ == "__main__":
    main()
