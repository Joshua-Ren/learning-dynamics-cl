"""Layerwise h-similarity diagnostic for the MMLU/GSM8K template experiment.

This reuses the paired samples and RH prediction-error definition from
``analyze_template_similarity.py``.  For each transformer layer, it measures
the layer hidden-state similarity and the RH-style product with the same
final-logit prediction-error similarity.  With the default answer_first scope,
MMLU contributes only the state that predicts its gold option letter.

This is a layerwise diagnostic, not the exact GH score: GH first projects the
prediction error through the LM head before combining it with intermediate h.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from analyze_template_similarity import (
    DEFAULT_ARTIFACTS,
    DEFAULT_MODEL,
    SamplePair,
    aligned_error_dot,
    build_paired_samples,
    choose_samples,
    encode_text,
    load_model_and_tokenizer,
    mmlu_answer_label_index,
    resolve_device,
)
from forvalue_streaming_ghrh import (
    build_batch_vocabulary,
    build_low_likelihood_position_mask,
    forward_logits_and_hidden,
    get_num_decoder_layers,
)


DEFAULT_OUTPUT = (
    DEFAULT_ARTIFACTS / "score_analysis_qwen25_base_mmlu5000_mmlu_answer_first_per_layer_input_rmsnorm_n50_seed42"
)


@dataclass
class LayerFeature:
    example_id: str
    source_row: int | None
    token_count: int
    kept_token_count: int
    truncated: bool
    answer_label_index: int | None
    h_sum_by_layer: torch.Tensor  # [decoder layer 1..L, hidden size]
    error_vocab_ids: torch.Tensor
    error_sum: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-layer MMLU/GSM8K h and RH-style score similarity."
    )
    parser.add_argument("--model_name", default=str(DEFAULT_MODEL))
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data_dir", default=str(DEFAULT_ARTIFACTS / "data"))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--num_mmlu", type=int, default=50)
    parser.add_argument("--num_gsm", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--prediction_topk", type=int, default=32)
    parser.add_argument("--lowest_likelihood_ratio", type=float, default=1.0)
    parser.add_argument(
        "--mmlu_token_scope",
        choices=("full", "answer_first"),
        default="answer_first",
    )
    parser.add_argument(
        "--gh_use_input_layernorm",
        "--gh_input_layernorm",
        dest="gh_use_input_layernorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply each selected intermediate hidden state to the next decoder "
            "block's input RMSNorm, matching forvalue_streaming_ghrh.py."
        ),
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def extract_layer_feature(
    model,
    tokenizer,
    sample: SamplePair,
    text: str,
    args: argparse.Namespace,
    device: str,
    num_layers: int,
) -> LayerFeature:
    input_ids, truncated = encode_text(tokenizer, text, args.max_length)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    intermediate_layers = list(range(1, num_layers))
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        logits, rh_hidden, gh_hidden = forward_logits_and_hidden(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gh_embedding_layers=intermediate_layers,
            gh_use_input_layernorm=args.gh_use_input_layernorm,
        )
    if gh_hidden is None:
        raise ValueError("Intermediate hidden states were not returned.")

    logits = logits[:, :-1, :].float()
    rh_hidden = rh_hidden[:, :-1, :].float()
    labels = input_ids[:, 1:]
    valid_mask = attention_mask[:, :-1].bool()
    keep_mask, log_denom = build_low_likelihood_position_mask(
        logits=logits,
        labels=labels,
        valid_mask=valid_mask,
        lowest_likelihood_ratio=args.lowest_likelihood_ratio,
    )

    answer_label_index = None
    score_mask = keep_mask
    if args.mmlu_token_scope == "answer_first" and sample.qa_answer_char_start is not None:
        answer_char_start = (
            sample.qa_answer_char_start if text == sample.qa_text else sample.pr_answer_char_start
        )
        answer_label_index = mmlu_answer_label_index(
            tokenizer, text, answer_char_start, args.max_length
        )
        if not keep_mask[0, answer_label_index]:
            raise ValueError(
                "The MMLU answer position was removed by --lowest_likelihood_ratio; use 1.0."
            )
        score_mask = torch.zeros_like(keep_mask)
        score_mask[0, answer_label_index] = True
    keep_indices = torch.nonzero(score_mask[0], as_tuple=False).squeeze(-1)
    if keep_indices.numel() == 0:
        raise ValueError(f"No token positions retained for {sample.example_id}")

    vocab_ids = build_batch_vocabulary(logits, score_mask, args.prediction_topk)
    selected_logits = logits[0, keep_indices].index_select(-1, vocab_ids)
    selected_log_denom = log_denom[0, keep_indices]
    if selected_log_denom.ndim == 1:
        selected_log_denom = selected_log_denom.unsqueeze(-1)
    probability = torch.exp(selected_logits - selected_log_denom)
    selected_labels = labels[0, keep_indices]
    observed = (selected_labels.unsqueeze(-1) == vocab_ids.unsqueeze(0)).to(probability.dtype)
    error_sum = torch.nan_to_num(
        (observed - probability).sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0
    )

    hidden_size = rh_hidden.shape[-1]
    sequence_length = rh_hidden.shape[1]
    intermediate_hidden = gh_hidden[:, :-1, :].float().reshape(
        1, sequence_length, num_layers - 1, hidden_size
    ).permute(0, 2, 1, 3)
    all_layer_hidden = torch.cat((intermediate_hidden, rh_hidden.unsqueeze(1)), dim=1)
    h_sum_by_layer = torch.nan_to_num(
        all_layer_hidden[0].index_select(1, keep_indices).sum(dim=1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    feature = LayerFeature(
        example_id=sample.example_id,
        source_row=sample.source_row,
        token_count=int(input_ids.shape[1]),
        kept_token_count=int(keep_indices.numel()),
        truncated=truncated,
        answer_label_index=answer_label_index,
        h_sum_by_layer=h_sum_by_layer.detach().cpu().float(),
        error_vocab_ids=vocab_ids.detach().cpu().to(torch.int64),
        error_sum=error_sum.detach().cpu().float(),
    )
    del logits, rh_hidden, gh_hidden, all_layer_hidden, intermediate_hidden
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return feature


def layerwise_matrices(
    mmlu_features: list[LayerFeature], gsm_features: list[LayerFeature]
) -> dict[str, np.ndarray]:
    mmlu_h = torch.stack([feature.h_sum_by_layer for feature in mmlu_features]).float()
    gsm_h = torch.stack([feature.h_sum_by_layer for feature in gsm_features]).float()
    h_dot = torch.einsum("nld,mld->lnm", mmlu_h, gsm_h).numpy().astype(np.float64)
    mmlu_unit = mmlu_h / mmlu_h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gsm_unit = gsm_h / gsm_h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    h_cosine = torch.einsum("nld,mld->lnm", mmlu_unit, gsm_unit).numpy().astype(np.float64)
    error_dot = np.empty((len(mmlu_features), len(gsm_features)), dtype=np.float64)
    for left_index, left in enumerate(mmlu_features):
        for right_index, right in enumerate(gsm_features):
            error_dot[left_index, right_index] = aligned_error_dot(left, right)
    return {
        "h_dot": h_dot,
        "h_cosine": h_cosine,
        "error_dot": np.broadcast_to(error_dot, h_dot.shape).copy(),
        "approximate_score": h_dot * error_dot[None, :, :],
    }


def bootstrap_layers(
    delta: np.ndarray, seed: int, samples: int
) -> dict[str, list[float]]:
    mean = delta.mean(axis=(1, 2))
    if samples <= 0:
        return {"mean": mean.tolist()}
    rng = np.random.default_rng(seed)
    layers, rows, columns = delta.shape
    estimates = np.empty((samples, layers), dtype=np.float64)
    for index in range(samples):
        row_indices = rng.integers(0, rows, size=rows)
        column_indices = rng.integers(0, columns, size=columns)
        estimates[index] = delta[:, row_indices][:, :, column_indices].mean(axis=(1, 2))
    return {
        "mean": mean.tolist(),
        "ci95_low": np.quantile(estimates, 0.025, axis=0).tolist(),
        "ci95_high": np.quantile(estimates, 0.975, axis=0).tolist(),
    }


def condition_means(matrices: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {metric: values.mean(axis=(1, 2)).tolist() for metric, values in matrices.items()}


def write_csv(path: Path, primary: dict[str, dict[str, list[float]]]) -> None:
    rows = []
    layer_count = len(primary["h_dot"]["mean"])
    for layer in range(layer_count):
        row: dict[str, Any] = {"layer": layer + 1, "readout": "RH(final)" if layer + 1 == layer_count else "GH hidden"}
        for metric, values in primary.items():
            row[f"{metric}_delta"] = values["mean"][layer]
            if "ci95_low" in values:
                row[f"{metric}_ci95_low"] = values["ci95_low"][layer]
                row[f"{metric}_ci95_high"] = values["ci95_high"][layer]
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path, primary: dict[str, dict[str, list[float]]], gh_use_input_layernorm: bool
) -> None:
    normalization_description = (
        "For layers 1–27, the intermediate state is passed through the next decoder "
        "block's input RMSNorm, exactly as `gh_use_input_layernorm` in the scorer. "
        "Layer 28 is final RH after the model's final RMSNorm."
        if gh_use_input_layernorm
        else "Layers 1–27 use raw intermediate hidden states; layer 28 is final RH after final RMSNorm."
    )
    lines = [
        "# Layerwise MMLU answer-position template analysis",
        "",
        "MMLU retains only the hidden state/error that predicts its gold option immediately after `Answer:` or `result:`. GSM8K retains all positions. Layers 1–27 are intermediate GH hidden states; layer 28 is final RH.",
        "",
        normalization_description,
        "",
        "The delta is MMLU Q/A × GSM Q/A minus MMLU Q/A × GSM P/R. Positive means GSM P/R is less similar. `approximate_score` is the RH-style `h_dot × prediction_error_dot`; it is a layerwise diagnostic, not the exact GH score.",
        "",
        "| Layer | h-dot delta | 95% CI | cosine delta | 95% CI | score delta | 95% CI |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    layer_count = len(primary["h_dot"]["mean"])
    for layer in range(layer_count):
        def interval(metric: str) -> str:
            if "ci95_low" not in primary[metric]:
                return "not requested"
            return f"[{primary[metric]['ci95_low'][layer]:.3e}, {primary[metric]['ci95_high'][layer]:.3e}]"
        lines.append(
            f"| {layer + 1} | {primary['h_dot']['mean'][layer]:.3e} | {interval('h_dot')} | "
            f"{primary['h_cosine']['mean'][layer]:.3e} | {interval('h_cosine')} | "
            f"{primary['approximate_score']['mean'][layer]:.3e} | {interval('approximate_score')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.lowest_likelihood_ratio <= 1.0:
        raise ValueError("--lowest_likelihood_ratio must be in (0, 1].")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    mmlu = choose_samples(
        build_paired_samples(data_dir / "mmlu.jsonl", data_dir / "mmlu_problem_result.jsonl", "mmlu"),
        args.num_mmlu, args.seed, "mmlu"
    )
    gsm = choose_samples(
        build_paired_samples(data_dir / "gsm8k_question_answer.jsonl", data_dir / "gsm8k_problem_result.jsonl", "gsm8k"),
        args.num_gsm, args.seed, "gsm8k"
    )
    device = resolve_device(args.device)
    print(f"Loading model on {device}: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args, device)
    num_layers = get_num_decoder_layers(model)
    print(f"Decoder layers: {num_layers}; MMLU token scope: {args.mmlu_token_scope}")
    features: dict[str, list[LayerFeature]] = {}
    for dataset_name, samples in (("mmlu", mmlu), ("gsm", gsm)):
        for variant in ("qa", "pr"):
            key = f"{dataset_name}_{variant}"
            features[key] = []
            print(f"Extracting {key}: {len(samples)} samples")
            for index, sample in enumerate(samples, start=1):
                text = sample.qa_text if variant == "qa" else sample.pr_text
                features[key].append(
                    extract_layer_feature(model, tokenizer, sample, text, args, device, num_layers)
                )
                if index % 10 == 0 or index == len(samples):
                    print(f"  {key}: {index}/{len(samples)}")

    conditions = {
        "MMLU Q/A__GSM Q/A": ("mmlu_qa", "gsm_qa"),
        "MMLU Q/A__GSM P/R": ("mmlu_qa", "gsm_pr"),
        "MMLU P/R__GSM Q/A": ("mmlu_pr", "gsm_qa"),
        "MMLU P/R__GSM P/R": ("mmlu_pr", "gsm_pr"),
    }
    matrices = {name: layerwise_matrices(features[left], features[right]) for name, (left, right) in conditions.items()}
    primary_matrices = {
        metric: matrices["MMLU Q/A__GSM Q/A"][metric] - matrices["MMLU Q/A__GSM P/R"][metric]
        for metric in matrices["MMLU Q/A__GSM Q/A"]
    }
    primary = {
        metric: bootstrap_layers(values, args.seed + index, args.bootstrap_samples)
        for index, (metric, values) in enumerate(primary_matrices.items())
    }
    payload = {f"{condition.replace(' ', '_').replace('/', '').replace('__', '_x_')}__{metric}": values for condition, result in matrices.items() for metric, values in result.items()}
    np.savez_compressed(output_dir / "layerwise_pairwise_matrices.npz", **payload)
    summary = {
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "mmlu_token_scope": args.mmlu_token_scope,
        "gh_use_input_layernorm": args.gh_use_input_layernorm,
        "num_layers": num_layers,
        "layer_indexing": "1..L decoder outputs; L is final RH, 1..L-1 are GH hidden states.",
        "definition": "Per-layer h dot/cosine; approximate_score = h_dot * final-logit prediction-error dot.",
        "condition_means": {name: condition_means(result) for name, result in matrices.items()},
        "primary_contrast": "MMLU Q/A × GSM Q/A minus MMLU Q/A × GSM P/R; positive means GSM P/R is less similar.",
        "primary_delta": primary,
        "mmlu_kept_tokens": [item.kept_token_count for item in features["mmlu_qa"]],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "primary_delta_by_layer.csv", primary)
    write_report(output_dir / "report.md", primary, args.gh_use_input_layernorm)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
