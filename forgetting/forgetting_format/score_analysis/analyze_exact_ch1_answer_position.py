"""Exact original CH1/RH score under paired Q/A and P/R templates.

This uses the non-approximate RH representation from
``forvalue_streaming_ghrh.py``:

    R(v) = sum_t (onehot(y_t=v) - p_t(v)) * h_t
    score(i, j) = sum_{v in V_i intersection V_j} <R_i(v), R_j(v)>

MMLU can be restricted to the one position predicting the gold option token
immediately after ``Answer:`` or ``result:``. GSM8K keeps all token positions.
The vocabulary is the unique top-k predicted tokens for each single-example
feature batch, so this is the original topk_unique CH1 computation with batch
size one and no dependence on arbitrary sample pairing in a batch.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from analyze_template_similarity import (
    DEFAULT_ARTIFACTS,
    DEFAULT_MODEL,
    SamplePair,
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
)


DEFAULT_OUTPUT = (
    DEFAULT_ARTIFACTS / "score_analysis_qwen25_base_mmlu5000_exact_ch1_mmlu_answer_first_n50_seed42"
)


@dataclass
class CH1Feature:
    example_id: str
    source_row: int | None
    token_count: int
    kept_token_count: int
    truncated: bool
    answer_label_index: int | None
    vocab_ids: torch.Tensor
    proposed: torch.Tensor  # [unique top-k vocabulary, hidden size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure exact original CH1/RH score on paired MMLU/GSM8K template samples."
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
        help="answer_first retains only the MMLU gold-option prediction position.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def extract_ch1_feature(
    model,
    tokenizer,
    sample: SamplePair,
    text: str,
    args: argparse.Namespace,
    device: str,
) -> CH1Feature:
    input_ids, truncated = encode_text(tokenizer, text, args.max_length)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        logits, rh_hidden, _ = forward_logits_and_hidden(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gh_embedding_layers=None,
        )

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
            sample.qa_answer_char_start
            if text == sample.qa_text
            else sample.pr_answer_char_start
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
    if vocab_ids.numel() == 0:
        raise ValueError(f"Empty top-k vocabulary for {sample.example_id}")
    kept_hidden = rh_hidden[0, keep_indices]
    kept_logits = logits[0, keep_indices].index_select(-1, vocab_ids)
    kept_log_denom = log_denom[0, keep_indices]
    if kept_log_denom.ndim == 1:
        kept_log_denom = kept_log_denom.unsqueeze(-1)
    if kept_log_denom.ndim != 2 or kept_log_denom.shape[-1] != 1:
        raise ValueError(f"Unexpected log-denominator shape {tuple(kept_log_denom.shape)}")
    kept_labels = labels[0, keep_indices]
    probability = torch.exp(kept_logits - kept_log_denom)
    observed = (kept_labels.unsqueeze(-1) == vocab_ids.unsqueeze(0)).to(probability.dtype)
    prediction_error = torch.nan_to_num(
        observed - probability, nan=0.0, posinf=0.0, neginf=0.0
    )
    proposed = torch.nan_to_num(
        torch.matmul(prediction_error.transpose(0, 1), kept_hidden),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    feature = CH1Feature(
        example_id=sample.example_id,
        source_row=sample.source_row,
        token_count=int(input_ids.shape[1]),
        kept_token_count=int(keep_indices.numel()),
        truncated=truncated,
        answer_label_index=answer_label_index,
        vocab_ids=vocab_ids.detach().cpu().to(torch.int64),
        # Original scorer materializes the proposed representation in bfloat16 on CPU.
        proposed=proposed.detach().cpu().to(torch.bfloat16),
    )
    del logits, rh_hidden, prediction_error, probability, proposed
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return feature


def aligned_ch1_dot(left: CH1Feature, right: CH1Feature) -> float:
    positions = torch.searchsorted(right.vocab_ids, left.vocab_ids)
    valid = positions < right.vocab_ids.numel()
    if not valid.any():
        return 0.0
    left_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    right_indices = positions[valid]
    matched = right.vocab_ids[right_indices] == left.vocab_ids[left_indices]
    if not matched.any():
        return 0.0
    left_values = left.proposed[left_indices[matched]].float()
    right_values = right.proposed[right_indices[matched]].float()
    return float(torch.sum(left_values * right_values).item())


def pairwise_matrix(mmlu: list[CH1Feature], gsm: list[CH1Feature]) -> np.ndarray:
    scores = np.empty((len(mmlu), len(gsm)), dtype=np.float64)
    for row, mmlu_feature in enumerate(mmlu):
        for column, gsm_feature in enumerate(gsm):
            scores[row, column] = aligned_ch1_dot(mmlu_feature, gsm_feature)
    return scores


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def bootstrap_delta(matrix: np.ndarray, seed: int, count: int) -> dict[str, float]:
    result = {"mean": float(matrix.mean())}
    if count <= 0:
        return result
    rng = np.random.default_rng(seed)
    rows, columns = matrix.shape
    draws = np.empty(count, dtype=np.float64)
    for index in range(count):
        row_indices = rng.integers(0, rows, size=rows)
        column_indices = rng.integers(0, columns, size=columns)
        draws[index] = matrix[np.ix_(row_indices, column_indices)].mean()
    result["ci95_low"] = float(np.quantile(draws, 0.025))
    result["ci95_high"] = float(np.quantile(draws, 0.975))
    return result


def feature_metadata(dataset: str, variant: str, features: Iterable[CH1Feature]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "variant": variant,
            "example_id": item.example_id,
            "source_row": item.source_row,
            "token_count": item.token_count,
            "kept_token_count": item.kept_token_count,
            "truncated": item.truncated,
            "answer_label_index": item.answer_label_index,
            "vocab_size": int(item.vocab_ids.numel()),
            "proposed_l2_norm": float(item.proposed.float().norm().item()),
        }
        for item in features
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    stats: dict[str, dict[str, float]],
    contrasts: dict[str, dict[str, float]],
    scope: str,
) -> None:
    lines = [
        "# Exact original CH1/RH template score analysis",
        "",
        "This is the non-approximate original CH1/RH score: align vocabulary tokens, then dot their per-token `R(v)` vectors.",
        "It does not use `h_dot × error_dot`.",
        "",
        (
            "MMLU scope: only the hidden state/error predicting its gold option immediately after `Answer:` or `result:`."
            if scope == "answer_first"
            else "MMLU scope: all prompt, completion, and EOS prediction positions."
        ),
        "GSM8K scope: all positions.",
        "",
        "## Mean pairwise exact CH1 scores",
        "",
        "| MMLU template × GSM template | exact CH1 score |",
        "| --- | ---: |",
    ]
    for name, values in stats.items():
        lines.append(f"| {name.replace('__', ' × ')} | {values['mean']:.5e} |")
    lines.extend(
        [
            "",
            "## Paired template contrasts",
            "",
            "A positive delta means GSM P/R yields a lower exact CH1 score than GSM Q/A.",
            "",
            "| Contrast | delta | 95% cluster-bootstrap CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, values in contrasts.items():
        interval = (
            f"[{values['ci95_low']:.5e}, {values['ci95_high']:.5e}]"
            if "ci95_low" in values
            else "not requested"
        )
        lines.append(f"| {name} | {values['mean']:.5e} | {interval} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.prediction_topk <= 0:
        raise ValueError("--prediction_topk must be positive.")
    if not 0.0 < args.lowest_likelihood_ratio <= 1.0:
        raise ValueError("--lowest_likelihood_ratio must be in (0, 1].")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_mmlu = choose_samples(
        build_paired_samples(data_dir / "mmlu.jsonl", data_dir / "mmlu_problem_result.jsonl", "mmlu"),
        args.num_mmlu,
        args.seed,
        "mmlu",
    )
    sampled_gsm = choose_samples(
        build_paired_samples(
            data_dir / "gsm8k_question_answer.jsonl",
            data_dir / "gsm8k_problem_result.jsonl",
            "gsm8k",
        ),
        args.num_gsm,
        args.seed,
        "gsm8k",
    )
    device = resolve_device(args.device)
    print(f"Loading model on {device}: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args, device)
    features: dict[str, list[CH1Feature]] = {}
    for dataset_name, samples in (("mmlu", sampled_mmlu), ("gsm", sampled_gsm)):
        for variant in ("qa", "pr"):
            key = f"{dataset_name}_{variant}"
            features[key] = []
            print(f"Extracting {key}: {len(samples)} samples")
            for index, sample in enumerate(samples, start=1):
                text = sample.qa_text if variant == "qa" else sample.pr_text
                features[key].append(extract_ch1_feature(model, tokenizer, sample, text, args, device))
                if index % 10 == 0 or index == len(samples):
                    print(f"  {key}: {index}/{len(samples)}")

    conditions = {
        "MMLU Q/A__GSM Q/A": ("mmlu_qa", "gsm_qa"),
        "MMLU Q/A__GSM P/R": ("mmlu_qa", "gsm_pr"),
        "MMLU P/R__GSM Q/A": ("mmlu_pr", "gsm_qa"),
        "MMLU P/R__GSM P/R": ("mmlu_pr", "gsm_pr"),
    }
    matrices = {
        name: pairwise_matrix(features[mmlu_key], features[gsm_key])
        for name, (mmlu_key, gsm_key) in conditions.items()
    }
    contrasts = {
        "MMLU Q/A: GSM Q/A minus GSM P/R": matrices["MMLU Q/A__GSM Q/A"]
        - matrices["MMLU Q/A__GSM P/R"],
        "MMLU P/R: GSM P/R minus GSM Q/A": matrices["MMLU P/R__GSM P/R"]
        - matrices["MMLU P/R__GSM Q/A"],
    }
    stats = {name: summarize(matrix) for name, matrix in matrices.items()}
    contrast_stats = {
        name: bootstrap_delta(matrix, args.seed + 1000 * index, args.bootstrap_samples)
        for index, (name, matrix) in enumerate(contrasts.items(), start=1)
    }
    payload = {
        name.lower().replace(" ", "_").replace("/", "").replace("__", "_x_"): matrix
        for name, matrix in matrices.items()
    }
    payload.update(
        {
            f"{('qa_primary' if index == 0 else 'pr_control')}__delta": matrix
            for index, matrix in enumerate(contrasts.values())
        }
    )
    np.savez_compressed(output_dir / "exact_ch1_pairwise_matrices.npz", **payload)
    metadata = []
    metadata.extend(feature_metadata("mmlu", "question_answer", features["mmlu_qa"]))
    metadata.extend(feature_metadata("mmlu", "problem_result", features["mmlu_pr"]))
    metadata.extend(feature_metadata("gsm8k", "question_answer", features["gsm_qa"]))
    metadata.extend(feature_metadata("gsm8k", "problem_result", features["gsm_pr"]))
    write_jsonl(output_dir / "feature_metadata.jsonl", metadata)
    summary = {
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "seed": args.seed,
        "num_mmlu": args.num_mmlu,
        "num_gsm": args.num_gsm,
        "prediction_topk": args.prediction_topk,
        "lowest_likelihood_ratio": args.lowest_likelihood_ratio,
        "mmlu_token_scope": args.mmlu_token_scope,
        "gsm_token_scope": "full",
        "batch_vocabulary": "topk_unique with feature batch_size=1",
        "definition": "exact CH1/RH score = aligned_vector_pairwise_dot(R_i, R_j), R(v)=sum_t(onehot-p)*h_t",
        "condition_stats": stats,
        "paired_contrasts": contrast_stats,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "report.md", stats, contrast_stats, args.mmlu_token_scope)
    primary = contrast_stats["MMLU Q/A: GSM Q/A minus GSM P/R"]
    print("Primary exact CH1 contrast (positive means GSM P/R is lower):")
    print(
        f"  delta={primary['mean']:.6e}, CI=[{primary.get('ci95_low', float('nan')):.6e}, "
        f"{primary.get('ci95_high', float('nan')):.6e}]"
    )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
