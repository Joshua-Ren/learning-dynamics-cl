import argparse
import ast
import contextlib
import gc
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import GRPO_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Memory-efficient Forward Valuation with RH/GH readout scores and "
            "batch-local vocabularies built from top-k model predictions at each position."
        )
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str, default="math_without_reason")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--prediction_topk", type=int, default=32)
    parser.add_argument(
        "--vocab_mode",
        type=str,
        default="topk_unique",
        choices=["topk_unique", "total_unique"],
        help=(
            "Vocabulary construction mode. `topk_unique` uses the union of top-k "
            "predictions inside each batch. `total_unique` uses the global unique "
            "tokens collected from train/test texts."
        ),
    )
    parser.add_argument(
        "--lowest_likelihood_ratio",
        type=float,
        default=1.0,
        help=(
            "Only keep the lowest-likelihood K%% input token positions for value "
            "computation. The likelihood is computed on the observed next token "
            "selected by input ids. Use 1.0 to keep all positions."
        ),
    )
    parser.add_argument(
        "--train_score_chunk",
        type=int,
        default=16,
        help="How many train batches to score against at a time.",
    )
    parser.add_argument("--n_class", type=int, default=10)
    parser.add_argument("--n_sample_per_class", type=int, default=90)
    parser.add_argument(
        "--embed_device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--score_device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--result_path", type=str, default="result_dict_streaming_gh_rh.json")
    parser.add_argument("--time_path", type=str, default="time_dict_streaming_gh_rh.json")
    parser.add_argument(
        "--readout_channel",
        type=str,
        default=None,
        choices=["rh", "gh", "both"],
        help="Single-channel alias. Use `both` to score RH plus GH.",
    )
    parser.add_argument(
        "--readout_channels",
        nargs="+",
        default=None,
        help=(
            "Readout score channel(s): `rh`, `gh`, or `both`. Also accepts "
            "comma-separated values such as `rh,gh`. If both RH and GH are enabled, "
            "the final similarity score is their sum."
        ),
    )
    parser.add_argument(
        "--gh_embedding_layers",
        nargs="+",
        default=["all"],
        help=(
            "GH hidden layers to concatenate before scoring. Default `all` with "
            "--gh_layer_index_mode bottom resolves to layers 1..L-1, excluding "
            "the input embedding layer 0 and final RH layer L."
        ),
    )
    parser.add_argument(
        "--gh_layer_index_mode",
        choices=["bottom", "top"],
        default="bottom",
        help=(
            "How to interpret --gh_embedding_layers. `bottom` counts hidden states "
            "from the input side: 1=first decoder layer, ..., L=final layer. "
            "`top` matches forgetting.py offsets from the final layer."
        ),
    )
    parser.add_argument(
        "--gh_use_input_layernorm",
        "--gh_input_layernorm",
        dest="gh_use_input_layernorm",
        action="store_true",
        help=(
            "GH-only: apply each selected decoder block's input_layernorm to "
            "the hidden state before concatenating."
        ),
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--approximate_proposed",
        action="store_true",
        help=(
            "Use token_sum similarity multiplied by avg_pb similarity as the "
            "final proposed score, and skip full proposed representation scoring."
        ),
    )
    return parser.parse_args()


def normalize_readout_channels(
    readout_channels: Optional[List[str]],
    readout_channel: Optional[str],
) -> Tuple[str, ...]:
    requested: List[str] = []
    if readout_channel:
        requested.append(readout_channel)
    if readout_channels:
        for value in readout_channels:
            requested.extend(part.strip() for part in value.split(",") if part.strip())
    if not requested:
        requested = ["rh"]
    if "both" in requested:
        requested.extend(["rh", "gh"])

    invalid = sorted({value for value in requested if value not in {"rh", "gh", "both"}})
    if invalid:
        raise ValueError(
            "readout_channels only supports `rh`, `gh`, or `both`; got "
            f"{', '.join(invalid)}."
        )
    return tuple(channel for channel in ("rh", "gh") if channel in requested)


GH_ALL_LAYERS = "all"


def parse_gh_embedding_layers(raw_layers: Iterable[str]):
    raw_items = [str(item).strip() for item in raw_layers]
    if not raw_items:
        raise ValueError("gh_embedding_layers must contain at least one layer index or `all`.")

    if len(raw_items) == 1 and raw_items[0].lower() in {GH_ALL_LAYERS, f"[{GH_ALL_LAYERS}]"}:
        return GH_ALL_LAYERS

    if len(raw_items) == 1 and raw_items[0].startswith("["):
        parsed = ast.literal_eval(raw_items[0])
        if isinstance(parsed, str):
            if parsed.lower() == GH_ALL_LAYERS:
                return GH_ALL_LAYERS
            raise ValueError(f"Unsupported gh_embedding_layers value: {parsed!r}")
        if isinstance(parsed, int):
            layer_indices = [parsed]
        else:
            layer_indices = list(parsed)
            if any(isinstance(item, str) and item.lower() == GH_ALL_LAYERS for item in layer_indices):
                if len(layer_indices) == 1:
                    return GH_ALL_LAYERS
                raise ValueError("Use either `all` or explicit gh_embedding_layers, not both.")
    else:
        layer_indices = []
        saw_all = False
        for raw_item in raw_items:
            cleaned = raw_item.strip().removeprefix("[").removesuffix("]")
            for piece in cleaned.split(","):
                piece = piece.strip()
                if not piece:
                    continue
                if piece.lower() == GH_ALL_LAYERS:
                    saw_all = True
                    continue
                layer_indices.append(int(piece))
        if saw_all:
            if layer_indices:
                raise ValueError("Use either `all` or explicit gh_embedding_layers, not both.")
            return GH_ALL_LAYERS

    if not layer_indices:
        raise ValueError("gh_embedding_layers must contain at least one layer index or `all`.")
    return [int(layer_index) for layer_index in layer_indices]


def get_num_decoder_layers(model: AutoModelForCausalLM) -> int:
    config = model.config
    for attr_name in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, attr_name):
            return int(getattr(config, attr_name))
    if hasattr(config, "text_config"):
        text_config = config.text_config
        for attr_name in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(text_config, attr_name):
                return int(getattr(text_config, attr_name))
    raise ValueError("Could not infer the number of decoder layers from model config.")


def get_decoder_layers(model: AutoModelForCausalLM):
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()

    def resolve_attr_path(root, attr_path: Tuple[str, ...]):
        module = root
        for attr_name in attr_path:
            module = getattr(module, attr_name, None)
            if module is None:
                return None
        return module

    decoder_attr_paths = (
        ("model",),
        ("base_model", "model"),
        ("transformer",),
        ("gpt_neox",),
        ("decoder",),
        (),
    )
    for attr_path in decoder_attr_paths:
        module = resolve_attr_path(model, attr_path)
        if module is None:
            continue
        for layers_attr in ("layers", "h", "block"):
            layers = getattr(module, layers_attr, None)
            if layers is not None:
                return layers

    raise ValueError("Could not find decoder layers for GH input-layernorm normalization.")


def get_input_layernorm(layer, layer_index: int):
    for attr_name in (
        "input_layernorm",
        "input_layer_norm",
        "ln_1",
        "layernorm_before",
        "self_attn_layer_norm",
    ):
        layernorm = getattr(layer, attr_name, None)
        if layernorm is not None:
            return layernorm
    raise ValueError(f"Could not find input layer norm on decoder layer {layer_index}.")


def resolve_gh_embedding_layers(
    gh_embedding_layers,
    model: AutoModelForCausalLM,
    gh_layer_index_mode: str,
) -> List[int]:
    if gh_embedding_layers != GH_ALL_LAYERS:
        return gh_embedding_layers

    num_layers = get_num_decoder_layers(model)
    if num_layers <= 1:
        raise ValueError(
            f"Could not resolve `all` gh_embedding_layers without the final layer "
            f"for num_layers={num_layers}."
        )
    if gh_layer_index_mode == "bottom":
        return list(range(1, num_layers))
    if gh_layer_index_mode == "top":
        return list(range(2, num_layers + 1))
    raise ValueError(f"Unsupported gh_layer_index_mode: {gh_layer_index_mode}")


def get_hidden_feature_width(hidden_size: int, gh_embedding_layers: List[int]) -> int:
    if not gh_embedding_layers:
        raise ValueError("gh_embedding_layers must contain at least one layer.")
    return hidden_size * len(gh_embedding_layers)


def evaluate_similarity_matrix(
    similarity_matrix: torch.Tensor,
    n_train: int,
    n_val: int,
    n_sample_per_class: int,
    n_class: int,
) -> Dict[str, float]:
    auc_list = []
    recall_list = []
    for i in range(n_val):
        gt_array = np.zeros(n_train)
        gt_array[
            (i // n_class) * n_sample_per_class : ((i // n_class) + 1) * n_sample_per_class
        ] = 1
        scores = similarity_matrix[i].cpu().float().numpy()
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e30, neginf=-1e30)
        auc_list.append(float(roc_auc_score(gt_array, scores, multi_class="ovr")))

        sorted_labels = np.argsort(scores)[::-1] // n_sample_per_class
        recall = (
            np.count_nonzero(
                (sorted_labels[:n_sample_per_class] == (i // n_class)).astype(int)
            )
            / float(n_sample_per_class)
        )
        recall_list.append(float(recall))

    return {
        "auc_mean": float(np.mean(auc_list)),
        "auc_std": float(np.std(auc_list)),
        "recall_mean": float(np.mean(recall_list)),
        "recall_std": float(np.std(recall_list)),
    }


def collect_unique_tokens(data_train, data_test, tokenizer, max_length: int) -> torch.Tensor:
    unique_token_ids = set()
    for dataset in [data_train, data_test]:
        for item in dataset:
            tokens = tokenizer.encode(
                item["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )
            unique_token_ids.update(tokens)
    return torch.tensor(sorted(unique_token_ids), dtype=torch.int64)


def forward_logits_and_hidden(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gh_embedding_layers: Optional[List[int]] = None,
    gh_layer_index_mode: str = "bottom",
    gh_use_input_layernorm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    need_gh_hidden = gh_embedding_layers is not None
    input_layernorms = (
        get_decoder_layers(model)
        if need_gh_hidden and gh_use_input_layernorm
        else None
    )

    def select_gh_hidden(outputs) -> Optional[torch.Tensor]:
        if gh_embedding_layers is None:
            return None
        if outputs.hidden_states is None:
            raise ValueError("Model output did not include hidden_states for GH scoring.")
        max_layer_index = len(outputs.hidden_states) 
        if gh_layer_index_mode not in {"bottom", "top"}:
            raise ValueError(f"Unsupported gh_layer_index_mode: {gh_layer_index_mode}")
        selected_hiddens = []
        for layer_index in gh_embedding_layers:
            if layer_index <= 0:
                raise ValueError(
                    "GH hidden layers must use positive indices and exclude input embedding 0."
                )
            if gh_layer_index_mode == "bottom":
                hidden_state_index = layer_index
            else:
                hidden_state_index = len(outputs.hidden_states) - layer_index
            if hidden_state_index <= 0 or hidden_state_index >= max_layer_index:
                raise ValueError(
                    "GH hidden layers must resolve to hidden states 1..L-1. "
                    f"Got gh_embedding_layers={gh_embedding_layers}, "
                    f"gh_layer_index_mode={gh_layer_index_mode!r}, resolved index "
                    f"{hidden_state_index}, valid range 1..{max_layer_index - 1}."
                )
            normalized_hidden = outputs.hidden_states[hidden_state_index]
            if input_layernorms is not None:
                if hidden_state_index >= len(input_layernorms):
                    raise ValueError(
                        "gh_use_input_layernorm cannot be used with layer "
                        f"{layer_index} in gh_layer_index_mode={gh_layer_index_mode!r}: "
                        "the selected hidden state is the final post-norm hidden state, "
                        "not an input to a decoder layer."
                    )
                input_layernorm = get_input_layernorm(
                    input_layernorms[hidden_state_index],
                    hidden_state_index,
                )
                normalized_hidden = input_layernorm(normalized_hidden)
            selected_hiddens.append(normalized_hidden)

        if len(selected_hiddens) == 1:
            return selected_hiddens[0]
        return torch.cat(selected_hiddens, dim=-1)

    if hasattr(model, "model") and hasattr(model, "lm_head"):
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=need_gh_hidden,
            return_dict=True,
        )
        rh_hidden = outputs.last_hidden_state
        logits = model.lm_head(rh_hidden)
        return logits, rh_hidden, select_gh_hidden(outputs)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    rh_hidden = outputs.hidden_states[-1]
    return outputs.logits, rh_hidden, select_gh_hidden(outputs)


def build_batch_vocabulary(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    prediction_topk: int,
) -> torch.Tensor:
    kept_logits = logits[valid_mask]
    if kept_logits.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=logits.device)
    topk_width = min(prediction_topk, logits.shape[-1])
    topk_ids = torch.topk(kept_logits, k=topk_width, dim=-1).indices
    return torch.unique(topk_ids.reshape(-1), sorted=True)


def build_low_likelihood_position_mask(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    lowest_likelihood_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_denom = torch.logsumexp(logits, dim=-1, keepdim=True)
    if lowest_likelihood_ratio >= 1.0:
        return valid_mask, log_denom

    label_logits = logits.gather(-1, labels.unsqueeze(-1))
    label_probs = torch.exp(label_logits - log_denom).squeeze(-1)
    label_probs = torch.where(valid_mask, label_probs, torch.full_like(label_probs, float("inf")))
    valid_counts = valid_mask.sum(dim=1)
    keep_counts = torch.ceil(valid_counts.to(torch.float32) * lowest_likelihood_ratio).to(torch.long)
    keep_counts = torch.where(valid_counts > 0, keep_counts.clamp(min=1), keep_counts)
    order = torch.argsort(label_probs, dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1,
        order,
        torch.arange(label_probs.shape[1], device=label_probs.device).unsqueeze(0).expand_as(order),
    )
    keep_mask = (ranks < keep_counts.unsqueeze(1)) & valid_mask
    return keep_mask, log_denom


def get_lm_head_weight(model: AutoModelForCausalLM) -> torch.Tensor:
    output_embeddings = (
        model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    )
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        return output_embeddings.weight
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight
    raise ValueError("Could not find an LM-head weight matrix for GH readout_channel.")


@torch.no_grad()
def compute_batch_representations(
    model: AutoModelForCausalLM,
    batch,
    embed_device: str,
    prediction_topk: int,
    vocab_mode: str,
    lowest_likelihood_ratio: float,
    global_vocab_ids_cpu: Optional[torch.Tensor] = None,
    global_vocab_ids_device: Optional[torch.Tensor] = None,
    readout_channels: Tuple[str, ...] = ("rh",),
    lm_head_weight: Optional[torch.Tensor] = None,
    gh_embedding_layers: Optional[List[int]] = None,
    gh_layer_index_mode: str = "bottom",
    gh_use_input_layernorm: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    compute_proposed: bool = True,
) -> Tuple[
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    input_ids = batch["input_ids"].to(embed_device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(embed_device, non_blocking=True)
    compute_rh = "rh" in readout_channels
    compute_gh = "gh" in readout_channels

    use_amp = embed_device.startswith("cuda")
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
    )
    with amp_ctx:
        logits, rh_hidden, gh_hidden = forward_logits_and_hidden(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gh_embedding_layers=gh_embedding_layers if compute_gh else None,
            gh_layer_index_mode=gh_layer_index_mode,
            gh_use_input_layernorm=gh_use_input_layernorm,
        )

    logits = logits[:, :-1, :].float()
    rh_hidden = rh_hidden[:, :-1, :].float()
    gh_hidden = gh_hidden[:, :-1, :].float() if gh_hidden is not None else None
    labels = input_ids[:, 1:]
    valid_mask = attention_mask[:, :-1].bool()
    keep_position_mask, log_denom = build_low_likelihood_position_mask(
        logits=logits,
        labels=labels,
        valid_mask=valid_mask,
        lowest_likelihood_ratio=lowest_likelihood_ratio,
    )

    if vocab_mode == "total_unique":
        if global_vocab_ids_cpu is None or global_vocab_ids_device is None:
            raise ValueError("global_vocab_ids are required when vocab_mode='total_unique'.")
        batch_vocab_ids = global_vocab_ids_device
        batch_vocab_ids_cpu = global_vocab_ids_cpu
    else:
        batch_vocab_ids = build_batch_vocabulary(logits, keep_position_mask, prediction_topk)
        batch_vocab_ids_cpu = batch_vocab_ids.detach().cpu()

    batch_size = logits.shape[0]
    hidden_dim = rh_hidden.shape[-1]
    gh_feature_width = (
        get_hidden_feature_width(hidden_dim, gh_embedding_layers)
        if compute_gh and gh_embedding_layers is not None
        else hidden_dim
    )
    if compute_gh:
        if gh_hidden is None:
            raise ValueError("GH scoring requested but no GH hidden states were returned.")
        if gh_hidden.shape[-1] != gh_feature_width:
            raise ValueError(
                "GH concatenated hidden width mismatch: expected "
                f"{gh_feature_width}, got {gh_hidden.shape[-1]}."
            )
    vocab_size = batch_vocab_ids.numel()
    keep_counts = keep_position_mask.sum(dim=1)
    max_keep = int(keep_counts.max().item()) if keep_counts.numel() > 0 else 0

    token_sum = rh_hidden.new_zeros((batch_size, hidden_dim))
    avg_pb = rh_hidden.new_zeros((batch_size, vocab_size))
    rh_proposed = (
        rh_hidden.new_zeros((batch_size, vocab_size, hidden_dim))
        if compute_proposed and compute_rh
        else None
    )
    gh_vocab_ids_cpu = torch.arange(hidden_dim, dtype=torch.int64)
    gh_proposed = (
        rh_hidden.new_zeros((batch_size, hidden_dim, gh_feature_width))
        if compute_proposed and compute_gh
        else None
    )

    if max_keep > 0:
        keep_order = torch.argsort((~keep_position_mask).to(torch.int64), dim=1)[:, :max_keep]
        kept_token_mask = (
            torch.arange(max_keep, device=logits.device).unsqueeze(0) < keep_counts.unsqueeze(1)
        )
        kept_rh_hidden = rh_hidden.gather(
            1,
            keep_order.unsqueeze(-1).expand(-1, -1, hidden_dim),
        )
        kept_rh_hidden = torch.nan_to_num(kept_rh_hidden, nan=0.0, posinf=0.0, neginf=0.0)
        kept_rh_hidden = kept_rh_hidden * kept_token_mask.unsqueeze(-1).to(kept_rh_hidden.dtype)
        token_sum = torch.nan_to_num(kept_rh_hidden.sum(dim=1), nan=0.0, posinf=0.0, neginf=0.0)
        kept_gh_hidden = None
        if compute_gh and gh_hidden is not None:
            kept_gh_hidden = gh_hidden.gather(
                1,
                keep_order.unsqueeze(-1).expand(-1, -1, gh_feature_width),
            )
            kept_gh_hidden = torch.nan_to_num(
                kept_gh_hidden,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            kept_gh_hidden = kept_gh_hidden * kept_token_mask.unsqueeze(-1).to(kept_gh_hidden.dtype)

        if vocab_size > 0:
            selected_logits = logits.index_select(dim=-1, index=batch_vocab_ids)
            kept_selected_logits = selected_logits.gather(
                1,
                keep_order.unsqueeze(-1).expand(-1, -1, vocab_size),
            )
            kept_log_denom = log_denom.gather(
                1,
                keep_order.unsqueeze(-1),
            )
            kept_labels = labels.gather(1, keep_order)
            prob_u = torch.exp(kept_selected_logits - kept_log_denom)
            one_hot_u = (kept_labels.unsqueeze(-1) == batch_vocab_ids.view(1, 1, -1)).to(prob_u.dtype)
            pb_diff = torch.nan_to_num(one_hot_u - prob_u, nan=0.0, posinf=0.0, neginf=0.0)
            pb_diff = pb_diff * kept_token_mask.unsqueeze(-1).to(pb_diff.dtype)
            avg_pb = torch.nan_to_num(pb_diff.sum(dim=1), nan=0.0, posinf=0.0, neginf=0.0)
            if compute_proposed and compute_rh:
                rh_proposed = torch.nan_to_num(
                    torch.matmul(pb_diff.transpose(1, 2), kept_rh_hidden),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            if compute_proposed and compute_gh:
                if kept_gh_hidden is None:
                    raise ValueError("GH scoring requested but kept GH hidden states are missing.")
                if lm_head_weight is None:
                    raise ValueError("lm_head_weight is required when scoring the GH channel.")
                selected_lm_head = lm_head_weight.index_select(0, batch_vocab_ids).to(
                    device=pb_diff.device,
                    dtype=pb_diff.dtype,
                )
                if selected_lm_head.shape[-1] != hidden_dim:
                    raise ValueError(
                        "LM-head hidden width mismatch: expected "
                        f"{hidden_dim}, got {selected_lm_head.shape[-1]}."
                    )
                gh_readout_error = torch.matmul(pb_diff, selected_lm_head)
                gh_proposed = torch.nan_to_num(
                    torch.matmul(gh_readout_error.transpose(1, 2), kept_gh_hidden),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

    return (
        rh_proposed.detach().cpu().to(torch.bfloat16) if rh_proposed is not None else None,
        batch_vocab_ids_cpu,
        gh_proposed.detach().cpu().to(torch.bfloat16) if gh_proposed is not None else None,
        gh_vocab_ids_cpu,
        token_sum.detach().cpu().to(torch.bfloat16),
        avg_pb.detach().cpu().to(torch.bfloat16),
    )


def compute_train_representations(
    dataloader_train: DataLoader,
    model: AutoModelForCausalLM,
    embed_device: str,
    prediction_topk: int,
    vocab_mode: str,
    lowest_likelihood_ratio: float,
    global_vocab_ids_cpu: Optional[torch.Tensor] = None,
    readout_channels: Tuple[str, ...] = ("rh",),
    lm_head_weight: Optional[torch.Tensor] = None,
    gh_embedding_layers: Optional[List[int]] = None,
    gh_layer_index_mode: str = "bottom",
    gh_use_input_layernorm: bool = False,
    compute_proposed: bool = True,
) -> List[Dict[str, Optional[torch.Tensor]]]:
    train_representations: List[Dict[str, Optional[torch.Tensor]]] = []
    global_vocab_ids_device = None
    if vocab_mode == "total_unique":
        if global_vocab_ids_cpu is None:
            raise ValueError("global_vocab_ids_cpu is required when vocab_mode='total_unique'.")
        global_vocab_ids_device = global_vocab_ids_cpu.to(embed_device, non_blocking=True)

    for step, batch in enumerate(dataloader_train):
        rh_proposed, rh_vocab_ids, gh_proposed, gh_vocab_ids, token_sum, avg_pb = (
            compute_batch_representations(
                model=model,
                batch=batch,
                embed_device=embed_device,
                prediction_topk=prediction_topk,
                vocab_mode=vocab_mode,
                lowest_likelihood_ratio=lowest_likelihood_ratio,
                global_vocab_ids_cpu=global_vocab_ids_cpu,
                global_vocab_ids_device=global_vocab_ids_device,
                readout_channels=readout_channels,
                lm_head_weight=lm_head_weight,
                gh_embedding_layers=gh_embedding_layers,
                gh_layer_index_mode=gh_layer_index_mode,
                gh_use_input_layernorm=gh_use_input_layernorm,
                compute_proposed=compute_proposed,
            )
        )
        train_representations.append(
            {
                "rh_proposed": rh_proposed,
                "rh_vocab_ids": rh_vocab_ids,
                "gh_proposed": gh_proposed,
                "gh_vocab_ids": gh_vocab_ids,
                "vocab_ids": rh_vocab_ids,
                "token_sum": token_sum,
                "avg_pb": avg_pb,
            }
        )

        if (step + 1) % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return train_representations


def compute_channel_similarity(
    channel: str,
    test_repr: Dict[str, Optional[torch.Tensor]],
    train_repr: Dict[str, Optional[torch.Tensor]],
    score_device: str,
    approximate_proposed: bool,
) -> torch.Tensor:
    if channel == "rh" and approximate_proposed:
        token_sim = dense_pairwise_dot(
            test_repr=test_repr["token_sum"],
            train_repr=train_repr["token_sum"],
            score_device=score_device,
        )
        avg_pb_sim = aligned_scalar_pairwise_dot(
            test_values=test_repr["avg_pb"],
            test_vocab_ids=test_repr["vocab_ids"],
            train_values=train_repr["avg_pb"],
            train_vocab_ids=train_repr["vocab_ids"],
            score_device=score_device,
        )
        return token_sim * avg_pb_sim

    repr_key = f"{channel}_proposed"
    vocab_key = f"{channel}_vocab_ids"
    if test_repr[repr_key] is None or train_repr[repr_key] is None:
        raise ValueError(f"{channel.upper()} representations are missing.")
    return aligned_vector_pairwise_dot(
        test_values=test_repr[repr_key],
        test_vocab_ids=test_repr[vocab_key],
        train_values=train_repr[repr_key],
        train_vocab_ids=train_repr[vocab_key],
        score_device=score_device,
    )


def sum_channel_similarities(
    readout_channels: Tuple[str, ...],
    test_repr: Dict[str, Optional[torch.Tensor]],
    train_repr: Dict[str, Optional[torch.Tensor]],
    score_device: str,
    approximate_proposed: bool,
) -> torch.Tensor:
    channel_scores = [
        compute_channel_similarity(
            channel=channel,
            test_repr=test_repr,
            train_repr=train_repr,
            score_device=score_device,
            approximate_proposed=approximate_proposed,
        )
        for channel in readout_channels
    ]
    total_score = channel_scores[0]
    for score in channel_scores[1:]:
        total_score = total_score + score
    return total_score


def get_vocab_intersection_indices(
    test_vocab_ids: torch.Tensor,
    train_vocab_ids: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if test_vocab_ids.numel() == 0 or train_vocab_ids.numel() == 0:
        return None, None

    positions = torch.searchsorted(train_vocab_ids, test_vocab_ids)
    valid = positions < train_vocab_ids.numel()
    if not valid.any():
        return None, None

    test_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    train_idx = positions[valid]
    matched = train_vocab_ids[train_idx] == test_vocab_ids[valid]
    if not matched.any():
        return None, None

    return test_idx[matched], train_idx[matched]


def dense_pairwise_dot(
    test_repr: torch.Tensor,
    train_repr: torch.Tensor,
    score_device: str,
) -> torch.Tensor:
    test_flat = test_repr.to(score_device, dtype=torch.float32, non_blocking=True)
    train_flat = train_repr.to(score_device, dtype=torch.float32, non_blocking=True)
    sim = torch.matmul(test_flat, train_flat.transpose(0, 1))
    sim = torch.nan_to_num(sim, nan=0.0, posinf=1e30, neginf=-1e30)
    return sim.cpu()


def aligned_scalar_pairwise_dot(
    test_values: torch.Tensor,
    test_vocab_ids: torch.Tensor,
    train_values: torch.Tensor,
    train_vocab_ids: torch.Tensor,
    score_device: str,
) -> torch.Tensor:
    test_idx, train_idx = get_vocab_intersection_indices(test_vocab_ids, train_vocab_ids)
    if test_idx is None or train_idx is None or test_idx.numel() == 0:
        return torch.zeros((test_values.shape[0], train_values.shape[0]), dtype=torch.float32)

    test_common = test_values[:, test_idx].to(score_device, dtype=torch.float32, non_blocking=True)
    train_common = train_values[:, train_idx].to(score_device, dtype=torch.float32, non_blocking=True)
    sim = torch.matmul(test_common, train_common.transpose(0, 1))
    sim = torch.nan_to_num(sim, nan=0.0, posinf=1e30, neginf=-1e30)
    return sim.cpu()


def aligned_vector_pairwise_dot(
    test_values: torch.Tensor,
    test_vocab_ids: torch.Tensor,
    train_values: torch.Tensor,
    train_vocab_ids: torch.Tensor,
    score_device: str,
) -> torch.Tensor:
    test_idx, train_idx = get_vocab_intersection_indices(test_vocab_ids, train_vocab_ids)
    if test_idx is None or train_idx is None or test_idx.numel() == 0:
        return torch.zeros((test_values.shape[0], train_values.shape[0]), dtype=torch.float32)

    test_common = test_values[:, test_idx, :].to(score_device, dtype=torch.float32, non_blocking=True)
    train_common = train_values[:, train_idx, :].to(score_device, dtype=torch.float32, non_blocking=True)
    test_flat = test_common.reshape(test_common.shape[0], -1)
    train_flat = train_common.reshape(train_common.shape[0], -1)
    sim = torch.matmul(test_flat, train_flat.transpose(0, 1))
    sim = torch.nan_to_num(sim, nan=0.0, posinf=1e30, neginf=-1e30)
    return sim.cpu()


def score_test_streaming(
    dataloader_test: DataLoader,
    model: AutoModelForCausalLM,
    train_representations: List[Dict[str, Optional[torch.Tensor]]],
    embed_device: str,
    score_device: str,
    prediction_topk: int,
    train_score_chunk: int,
    vocab_mode: str,
    lowest_likelihood_ratio: float,
    global_vocab_ids_cpu: Optional[torch.Tensor] = None,
    readout_channels: Tuple[str, ...] = ("rh",),
    lm_head_weight: Optional[torch.Tensor] = None,
    gh_embedding_layers: Optional[List[int]] = None,
    gh_layer_index_mode: str = "bottom",
    gh_use_input_layernorm: bool = False,
    approximate_proposed: bool = False,
) -> torch.Tensor:
    proposed_rows = []
    global_vocab_ids_device = None
    if vocab_mode == "total_unique":
        if global_vocab_ids_cpu is None:
            raise ValueError("global_vocab_ids_cpu is required when vocab_mode='total_unique'.")
        global_vocab_ids_device = global_vocab_ids_cpu.to(embed_device, non_blocking=True)

    for step, batch in enumerate(dataloader_test):
        rh_proposed, rh_vocab_ids, gh_proposed, gh_vocab_ids, test_token_sum, test_avg_pb = (
            compute_batch_representations(
                model=model,
                batch=batch,
                embed_device=embed_device,
                prediction_topk=prediction_topk,
                vocab_mode=vocab_mode,
                lowest_likelihood_ratio=lowest_likelihood_ratio,
                global_vocab_ids_cpu=global_vocab_ids_cpu,
                global_vocab_ids_device=global_vocab_ids_device,
                readout_channels=readout_channels,
                lm_head_weight=lm_head_weight,
                gh_embedding_layers=gh_embedding_layers,
                gh_layer_index_mode=gh_layer_index_mode,
                gh_use_input_layernorm=gh_use_input_layernorm,
                compute_proposed=not approximate_proposed,
            )
        )
        test_repr = {
            "rh_proposed": rh_proposed,
            "rh_vocab_ids": rh_vocab_ids,
            "gh_proposed": gh_proposed,
            "gh_vocab_ids": gh_vocab_ids,
            "vocab_ids": rh_vocab_ids,
            "token_sum": test_token_sum,
            "avg_pb": test_avg_pb,
        }

        batch_scores = []
        for start in range(0, len(train_representations), train_score_chunk):
            score_parts = []
            for train_repr in train_representations[start : start + train_score_chunk]:
                score_parts.append(
                    sum_channel_similarities(
                        readout_channels=readout_channels,
                        test_repr=test_repr,
                        train_repr=train_repr,
                        score_device=score_device,
                        approximate_proposed=approximate_proposed,
                    )
                )

            batch_scores.append(torch.cat(score_parts, dim=1))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        proposed_rows.append(torch.cat(batch_scores, dim=1))

        if (step + 1) % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(proposed_rows, dim=0)


def main() -> None:
    args = parse_args()
    start_time = time.time()
    readout_channels = normalize_readout_channels(args.readout_channels, args.readout_channel)
    gh_embedding_layers = parse_gh_embedding_layers(args.gh_embedding_layers)

    if args.prediction_topk <= 0:
        raise ValueError(f"prediction_topk must be positive, got {args.prediction_topk}.")
    if not (0.0 < args.lowest_likelihood_ratio <= 1.0):
        raise ValueError(
            f"lowest_likelihood_ratio must be in (0, 1], got {args.lowest_likelihood_ratio}."
        )
    if args.approximate_proposed and "gh" in readout_channels:
        raise ValueError(
            "approximate_proposed only supports the RH channel. Remove "
            "--approximate_proposed or use --readout_channels rh."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    train_path = Path(args.dataset_dir) / f"{args.dataset_name}_train.hf"
    test_path = Path(args.dataset_dir) / f"{args.dataset_name}_test.hf"
    data_train = load_from_disk(str(train_path))
    data_test = load_from_disk(str(test_path))

    dataset_train = GRPO_dataset(data_train, tokenizer, max_length=args.max_length)
    dataset_test = GRPO_dataset(data_test, tokenizer, max_length=args.max_length)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    global_vocab_ids = None
    if args.vocab_mode == "total_unique":
        print("Collecting global unique tokens...")
        global_vocab_ids = collect_unique_tokens(data_train, data_test, tokenizer, args.max_length)
        print(f"Global unique tokens: {global_vocab_ids.numel()}")
        print(
            f"Using vocab_mode=total_unique, lowest_likelihood_ratio={args.lowest_likelihood_ratio}, "
            f"train_score_chunk={args.train_score_chunk}"
        )
    else:
        print(
            "Using batch-local vocabularies from top-k predictions. "
            f"vocab_mode=topk_unique, prediction_topk={args.prediction_topk}, "
            f"lowest_likelihood_ratio={args.lowest_likelihood_ratio}, "
            f"train_score_chunk={args.train_score_chunk}"
        )
    print(f"Readout channels: {', '.join(readout_channels)}")
    if len(readout_channels) > 1:
        print("Final similarity score: sum of RH and GH channel scores.")

    dtype = torch.float16 if args.embed_device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.to(args.embed_device)
    model.eval()
    lm_head_weight = get_lm_head_weight(model) if "gh" in readout_channels else None
    gh_resolved_hidden_indices = None
    if "gh" in readout_channels:
        gh_embedding_layers = resolve_gh_embedding_layers(
            gh_embedding_layers,
            model,
            args.gh_layer_index_mode,
        )
        num_layers = get_num_decoder_layers(model)
        resolved_indices = [
            layer_index
            if args.gh_layer_index_mode == "bottom"
            else num_layers + 1 - layer_index
            for layer_index in gh_embedding_layers
        ]
        invalid_indices = [
            hidden_state_index
            for hidden_state_index in resolved_indices
            if hidden_state_index <= 0 or hidden_state_index >= num_layers
        ]
        if invalid_indices:
            raise ValueError(
                "GH must use hidden states 1..L-1 and RH uses L. "
                f"Resolved GH hidden-state indices {resolved_indices}, "
                f"valid range is 1..{num_layers - 1}."
            )
        gh_resolved_hidden_indices = resolved_indices
        print(f"GH embedding layers: {gh_embedding_layers}")
        print(f"GH resolved hidden-state indices: {resolved_indices}")
        print(f"GH layer index mode: {args.gh_layer_index_mode}")
        print(f"GH use input layernorm: {args.gh_use_input_layernorm}")
    else:
        gh_embedding_layers = None

    feat_start = time.time()
    print("Building train representations...")
    train_representations = compute_train_representations(
        dataloader_train=dataloader_train,
        model=model,
        embed_device=args.embed_device,
        prediction_topk=args.prediction_topk,
        vocab_mode=args.vocab_mode,
        lowest_likelihood_ratio=args.lowest_likelihood_ratio,
        global_vocab_ids_cpu=global_vocab_ids,
        readout_channels=readout_channels,
        lm_head_weight=lm_head_weight,
        gh_embedding_layers=gh_embedding_layers,
        gh_layer_index_mode=args.gh_layer_index_mode,
        gh_use_input_layernorm=args.gh_use_input_layernorm,
        compute_proposed=not args.approximate_proposed,
    )
    feat_time = time.time() - feat_start

    train_vocab_sizes = [int(repr_dict["vocab_ids"].numel()) for repr_dict in train_representations]
    avg_vocab = float(np.mean(train_vocab_sizes)) if train_vocab_sizes else 0.0
    max_vocab = int(np.max(train_vocab_sizes)) if train_vocab_sizes else 0
    if args.approximate_proposed:
        print(
            "Train representations ready in approximate mode. "
            f"num_train_batches={len(train_representations)}, avg_batch_vocab={avg_vocab:.1f}, "
            f"max_batch_vocab={max_vocab}"
        )
    else:
        sample_shapes = {}
        if train_representations:
            first_repr = train_representations[0]
            for channel in readout_channels:
                proposed = first_repr[f"{channel}_proposed"]
                if proposed is not None:
                    sample_shapes[channel] = tuple(proposed.shape)
        print(
            "Train representations ready. "
            f"num_train_batches={len(train_representations)}, first_proposed_shapes={sample_shapes}, "
            f"avg_batch_vocab={avg_vocab:.1f}, max_batch_vocab={max_vocab}"
        )

    score_start = time.time()
    print("Scoring test batches (streaming)...")
    train_2_test_values = score_test_streaming(
        dataloader_test=dataloader_test,
        model=model,
        train_representations=train_representations,
        embed_device=args.embed_device,
        score_device=args.score_device,
        prediction_topk=args.prediction_topk,
        train_score_chunk=args.train_score_chunk,
        vocab_mode=args.vocab_mode,
        lowest_likelihood_ratio=args.lowest_likelihood_ratio,
        global_vocab_ids_cpu=global_vocab_ids,
        readout_channels=readout_channels,
        lm_head_weight=lm_head_weight,
        gh_embedding_layers=gh_embedding_layers,
        gh_layer_index_mode=args.gh_layer_index_mode,
        gh_use_input_layernorm=args.gh_use_input_layernorm,
        approximate_proposed=args.approximate_proposed,
    )
    score_time = time.time() - score_start

    n_train = train_2_test_values.shape[1]
    n_val = train_2_test_values.shape[0]
    if args.n_sample_per_class * args.n_class != n_train:
        raise ValueError(
            f"n_sample_per_class({args.n_sample_per_class}) * n_class({args.n_class}) "
            f"!= n_train({n_train})."
        )
    if n_val % args.n_class != 0:
        raise ValueError(f"n_val({n_val}) must be divisible by n_class({args.n_class}).")

    proposed_metrics = evaluate_similarity_matrix(
        train_2_test_values,
        n_train=n_train,
        n_val=n_val,
        n_sample_per_class=args.n_sample_per_class,
        n_class=args.n_class,
    )

    total_time = time.time() - start_time
    score_label = "+".join(readout_channels)
    print(
        f"{score_label} proposed Auc: "
        f"{proposed_metrics['auc_mean']:.3f}/{proposed_metrics['auc_std']:.3f}"
    )
    print(
        f"{score_label} proposed Recall: "
        f"{proposed_metrics['recall_mean']:.3f}/{proposed_metrics['recall_std']:.3f}"
    )
    print(
        f"Timing: total={total_time:.1f}s, train_feature={feat_time:.1f}s, "
        f"test_score={score_time:.1f}s"
    )

    result_dict = {
        "approximate_proposed": bool(args.approximate_proposed),
        "readout_channels": list(readout_channels),
        "gh_embedding_layers": gh_embedding_layers if "gh" in readout_channels else None,
        "gh_resolved_hidden_indices": gh_resolved_hidden_indices,
        "gh_layer_index_mode": args.gh_layer_index_mode if "gh" in readout_channels else None,
        "gh_use_input_layernorm": (
            bool(args.gh_use_input_layernorm) if "gh" in readout_channels else False
        ),
        "vocab_mode": args.vocab_mode,
        "prediction_topk": int(args.prediction_topk),
        "lowest_likelihood_ratio": float(args.lowest_likelihood_ratio),
        "proposed_auc": proposed_metrics["auc_mean"],
        "proposed_auc_std": proposed_metrics["auc_std"],
        "proposed_recall": proposed_metrics["recall_mean"],
        "proposed_recall_std": proposed_metrics["recall_std"],
    }
    time_dict = {
        args.model_name: total_time,
        "train_feature_sec": feat_time,
        "test_score_sec": score_time,
    }

    with open(args.result_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f)
    with open(args.time_path, "w", encoding="utf-8") as f:
        json.dump(time_dict, f)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
