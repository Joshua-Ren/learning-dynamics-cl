"""Decompose template-sensitive RH score components on paired MMLU and GSM8K samples.

This is a small controlled diagnostic for the score used by
forvalue_streaming_ghrh.py.  It reuses that module's final RH hidden state,
top-k batch vocabulary, and prediction-error definition:

    h_sum = sum_t h_t
    error = sum_t (onehot(y_t) - p_t)
    approximate_score = dot(h_sum_a, h_sum_b) * dot(error_a, error_b)

The script treats MMLU as the test side and GSM8K as the train side, samples
the same source examples under Q/A and P/R templates, and writes the complete
2x2 cross-template matrix.  It also reports cosine(h_sum) as a norm-invariant
diagnostic; the original score itself uses unnormalized dot products.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
FORMAT_ROOT = SCRIPT_DIR.parent
FORGETTING_ROOT = FORMAT_ROOT.parent
for import_root in (str(FORMAT_ROOT), str(FORGETTING_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from common import read_jsonl, write_jsonl
from forvalue_streaming_ghrh import (
    build_batch_vocabulary,
    build_low_likelihood_position_mask,
    forward_logits_and_hidden,
)


DEFAULT_ARTIFACTS = FORMAT_ROOT / "artifacts"
DEFAULT_MODEL = (
    DEFAULT_ARTIFACTS
    / "run_full_e1_mmlu_aux5000_question_answer_5000effective_qwen25_1p5b_base_fp32_len1024_lr1e5_bs4ga8"
)
DEFAULT_OUTPUT = DEFAULT_ARTIFACTS / "score_analysis_qwen25_base_mmlu5000_n50_seed42"


@dataclass
class SamplePair:
    example_id: str
    source_row: int | None
    qa_text: str
    pr_text: str
    qa_answer_char_start: int | None
    pr_answer_char_start: int | None


@dataclass
class RHFeature:
    example_id: str
    source_row: int | None
    token_count: int
    kept_token_count: int
    truncated: bool
    answer_label_index: int | None
    h_sum: torch.Tensor
    error_vocab_ids: torch.Tensor
    error_sum: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure RH and prediction-error template similarity on paired MMLU/GSM8K samples."
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
        default="full",
        help=(
            "Which MMLU token positions contribute. answer_first retains only "
            "the first gold option token predicted immediately after Answer:/result:."
        ),
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float32"),
        default="auto",
        help="Model weight dtype. Auto uses bfloat16 on CUDA and float32 on CPU.",
    )
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return value


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records = list(read_jsonl(path))
    if not records:
        raise ValueError(f"No records found in {path}")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        example_id = str(record["example_id"])
        if example_id in indexed:
            raise ValueError(f"Duplicate example_id {example_id} in {path}")
        indexed[example_id] = record
    return indexed


def build_paired_samples(
    qa_path: Path,
    pr_path: Path,
    dataset_name: str,
) -> list[SamplePair]:
    qa_records = records_by_id(qa_path)
    pr_records = records_by_id(pr_path)
    if set(qa_records) != set(pr_records):
        raise ValueError(
            f"{dataset_name}: Q/A and P/R files do not have identical example_id sets."
        )

    pairs: list[SamplePair] = []
    for example_id, qa_record in qa_records.items():
        pr_record = pr_records[example_id]
        qa_prompt = str(qa_record["prompt"])
        pr_prompt = str(pr_record["prompt"])
        if dataset_name == "mmlu":
            qa_completion = f" {str(qa_record['target']).strip()}"
            pr_completion = f" {str(pr_record['target']).strip()}"
            if qa_completion != pr_completion:
                raise ValueError(f"{dataset_name}: target mismatch for {example_id}")
        elif dataset_name == "gsm8k":
            qa_completion = str(qa_record["completion"])
            pr_completion = str(pr_record["completion"])
            if qa_completion != pr_completion:
                raise ValueError(f"{dataset_name}: completion mismatch for {example_id}")
        else:
            raise ValueError(f"Unsupported dataset_name={dataset_name!r}")

        source_row = qa_record.get("source_row")
        if source_row != pr_record.get("source_row"):
            raise ValueError(f"{dataset_name}: source_row mismatch for {example_id}")
        pairs.append(
            SamplePair(
                example_id=example_id,
                source_row=int(source_row) if source_row is not None else None,
                qa_text=qa_prompt + qa_completion,
                pr_text=pr_prompt + pr_completion,
                qa_answer_char_start=len(qa_prompt) if dataset_name == "mmlu" else None,
                pr_answer_char_start=len(pr_prompt) if dataset_name == "mmlu" else None,
            )
        )

    return pairs


def choose_samples(samples: list[SamplePair], count: int, seed: int, label: str) -> list[SamplePair]:
    if count <= 0:
        raise ValueError(f"{label}: sample count must be positive")
    if count > len(samples):
        raise ValueError(f"{label}: requested {count} samples but only {len(samples)} are available")
    selected_indices = random.Random(seed).sample(range(len(samples)), count)
    return [samples[index] for index in selected_indices]


def load_model_and_tokenizer(args: argparse.Namespace, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=resolve_dtype(args.dtype, device),
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def encode_text(tokenizer, text: str, max_length: int) -> tuple[torch.Tensor, bool]:
    token_ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    if tokenizer.eos_token_id is not None and (
        not token_ids or token_ids[-1] != tokenizer.eos_token_id
    ):
        token_ids.append(tokenizer.eos_token_id)
    truncated = len(token_ids) > max_length
    if truncated:
        # Match completion-preserving SFT truncation: retain the end of the sequence.
        token_ids = token_ids[-max_length:]
    if len(token_ids) < 2:
        raise ValueError("A scored sequence needs at least two tokens.")
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0), truncated


def mmlu_answer_label_index(
    tokenizer, text: str, answer_char_start: int, max_length: int
) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoded["input_ids"])
    answer_token_input_index = next(
        (
            index
            for index, (start, end) in enumerate(encoded["offset_mapping"])
            if end > start and end > answer_char_start
        ),
        None,
    )
    if answer_token_input_index is None:
        raise ValueError("Could not locate the MMLU answer token from tokenizer offsets.")
    if tokenizer.eos_token_id is not None and (
        not token_ids or token_ids[-1] != tokenizer.eos_token_id
    ):
        token_ids.append(tokenizer.eos_token_id)
    trim_start = max(0, len(token_ids) - max_length)
    retained_length = min(len(token_ids), max_length)
    answer_label_index = answer_token_input_index - trim_start - 1
    if not 0 <= answer_label_index < retained_length - 1:
        raise ValueError("MMLU answer token was truncated or has no valid prediction position.")
    return answer_label_index


def extract_rh_feature(
    model,
    tokenizer,
    sample: SamplePair,
    text: str,
    args: argparse.Namespace,
    device: str,
) -> RHFeature:
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
                "The MMLU answer position was removed by --lowest_likelihood_ratio; "
                "use --lowest_likelihood_ratio 1.0 for answer-position analysis."
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
        raise ValueError(
            f"Unexpected selected log-denominator shape {tuple(kept_log_denom.shape)}"
        )
    kept_labels = labels[0, keep_indices]

    probability = torch.exp(kept_logits - kept_log_denom)
    observed = (kept_labels.unsqueeze(-1) == vocab_ids.unsqueeze(0)).to(probability.dtype)
    prediction_error = torch.nan_to_num(observed - probability, nan=0.0, posinf=0.0, neginf=0.0)

    h_sum = torch.nan_to_num(kept_hidden.sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
    error_sum = torch.nan_to_num(prediction_error.sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0)

    feature = RHFeature(
        example_id=sample.example_id,
        source_row=sample.source_row,
        token_count=int(input_ids.shape[1]),
        kept_token_count=int(keep_indices.numel()),
        truncated=truncated,
        answer_label_index=answer_label_index,
        h_sum=h_sum.detach().cpu().float(),
        error_vocab_ids=vocab_ids.detach().cpu().to(torch.int64),
        error_sum=error_sum.detach().cpu().float(),
    )
    del logits, rh_hidden, prediction_error, probability
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return feature


def aligned_error_dot(left: RHFeature, right: RHFeature) -> float:
    positions = torch.searchsorted(right.error_vocab_ids, left.error_vocab_ids)
    valid = positions < right.error_vocab_ids.numel()
    if not valid.any():
        return 0.0
    left_indices = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    right_indices = positions[valid]
    matched = right.error_vocab_ids[right_indices] == left.error_vocab_ids[left_indices]
    if not matched.any():
        return 0.0
    left_values = left.error_sum[left_indices[matched]]
    right_values = right.error_sum[right_indices[matched]]
    return float(torch.dot(left_values, right_values).item())


def pairwise_matrices(
    mmlu_features: list[RHFeature],
    gsm_features: list[RHFeature],
) -> dict[str, np.ndarray]:
    mmlu_h = torch.stack([feature.h_sum for feature in mmlu_features]).float()
    gsm_h = torch.stack([feature.h_sum for feature in gsm_features]).float()
    h_dot = torch.matmul(mmlu_h, gsm_h.transpose(0, 1)).numpy()
    mmlu_norm = mmlu_h.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gsm_norm = gsm_h.norm(dim=1, keepdim=True).clamp_min(1e-12)
    h_cosine = torch.matmul(mmlu_h / mmlu_norm, (gsm_h / gsm_norm).transpose(0, 1)).numpy()

    error_dot = np.empty((len(mmlu_features), len(gsm_features)), dtype=np.float64)
    for mmlu_index, mmlu_feature in enumerate(mmlu_features):
        for gsm_index, gsm_feature in enumerate(gsm_features):
            error_dot[mmlu_index, gsm_index] = aligned_error_dot(mmlu_feature, gsm_feature)
    approximate_score = h_dot.astype(np.float64) * error_dot
    return {
        "h_dot": h_dot.astype(np.float64),
        "h_cosine": h_cosine.astype(np.float64),
        "error_dot": error_dot,
        "approximate_score": approximate_score,
    }


def summarize_matrix(matrix: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(matrix.mean()),
        "median": float(np.median(matrix)),
        "std": float(matrix.std(ddof=1)),
        "minimum": float(matrix.min()),
        "maximum": float(matrix.max()),
    }


def bootstrap_delta(
    delta_matrix: np.ndarray,
    seed: int,
    num_samples: int,
) -> dict[str, float]:
    if num_samples <= 0:
        return {"mean": float(delta_matrix.mean())}
    rng = np.random.default_rng(seed)
    rows, columns = delta_matrix.shape
    estimates = np.empty(num_samples, dtype=np.float64)
    for index in range(num_samples):
        row_indices = rng.integers(0, rows, size=rows)
        column_indices = rng.integers(0, columns, size=columns)
        estimates[index] = delta_matrix[np.ix_(row_indices, column_indices)].mean()
    return {
        "mean": float(delta_matrix.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }


def feature_metadata(dataset: str, variant: str, features: Iterable[RHFeature]) -> list[dict[str, Any]]:
    records = []
    for feature in features:
        records.append(
            {
                "dataset": dataset,
                "variant": variant,
                "example_id": feature.example_id,
                "source_row": feature.source_row,
                "token_count": feature.token_count,
                "kept_token_count": feature.kept_token_count,
                "truncated": feature.truncated,
                "answer_label_index": feature.answer_label_index,
                "h_norm": float(feature.h_sum.norm().item()),
                "error_vocab_size": int(feature.error_vocab_ids.numel()),
                "error_l2_norm": float(feature.error_sum.norm().item()),
            }
        )
    return records


def scientific(value: float) -> str:
    return f"{value:.5e}"


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    condition_stats: dict[str, dict[str, dict[str, float]]],
    deltas: dict[str, dict[str, dict[str, float]]],
) -> None:
    scope_description = (
        "MMLU scope: only the first gold option token predicted immediately after Answer:/result:."
        if summary["manifest"]["mmlu_token_scope"] == "answer_first"
        else "MMLU scope: all prompt, completion, and EOS prediction positions."
    )
    lines = [
        "# Qwen2.5-1.5B Base MMLU-bridge template score analysis",
        "",
        "Model: MMLU-5k Q/A full-parameter checkpoint. MMLU test is the test side; GSM8K train is the train side.",
        "",
        scope_description,
        "",
        "RH h dot and prediction-error dot follow forvalue_streaming_ghrh.py. The approximate score is their product. Cosine(h) is an additional normalized diagnostic.",
        "",
        "## Mean pairwise scores",
        "",
        "| MMLU template × GSM template | h dot | cosine(h) | prediction-error dot | approximate score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition, metric_stats in condition_stats.items():
        lines.append(
            "| "
            + condition.replace("__", " × ")
            + " | "
            + scientific(metric_stats["h_dot"]["mean"])
            + " | "
            + f"{metric_stats['h_cosine']['mean']:.6f}"
            + " | "
            + scientific(metric_stats["error_dot"]["mean"])
            + " | "
            + scientific(metric_stats["approximate_score"]["mean"])
            + " |"
        )

    lines.extend(
        [
            "",
            "## Paired template contrasts",
            "",
            "A positive delta means replacing GSM Q/A by GSM P/R lowers the corresponding similarity/score.",
            "",
            "| Contrast | Metric | delta | 95% cluster-bootstrap CI |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for contrast, metric_summaries in deltas.items():
        for metric, values in metric_summaries.items():
            if "ci95_low" in values:
                interval = f"[{scientific(values['ci95_low'])}, {scientific(values['ci95_high'])}]"
            else:
                interval = "not requested"
            lines.append(
                f"| {contrast} | {metric} | {scientific(values['mean'])} | {interval} |"
            )

    primary = deltas["MMLU Q/A: GSM Q/A minus GSM P/R"]
    direction = (
        "supports a decrease"
        if primary["h_dot"]["mean"] > 0 and primary["h_cosine"]["mean"] > 0
        else "does not show a joint h-dot and cosine decrease"
    )
    lines.extend(
        [
            "",
            "## Directional readout",
            "",
            "For the primary contrast, a positive delta is evidence that GSM P/R is less similar to standard MMLU Q/A. "
            + f"The current result {direction}.",
            "",
            "Raw pairwise matrices are stored in pairwise_matrices.npz; per-example norms and token counts are in feature_metadata.jsonl.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.lowest_likelihood_ratio <= 1.0:
        raise ValueError("--lowest_likelihood_ratio must be in (0, 1].")
    if args.prediction_topk <= 0:
        raise ValueError("--prediction_topk must be positive")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    mmlu_pairs = build_paired_samples(
        data_dir / "mmlu.jsonl",
        data_dir / "mmlu_problem_result.jsonl",
        "mmlu",
    )
    gsm_pairs = build_paired_samples(
        data_dir / "gsm8k_question_answer.jsonl",
        data_dir / "gsm8k_problem_result.jsonl",
        "gsm8k",
    )
    sampled_mmlu = choose_samples(mmlu_pairs, args.num_mmlu, args.seed, "mmlu")
    sampled_gsm = choose_samples(gsm_pairs, args.num_gsm, args.seed, "gsm8k")

    manifest = {
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "device": device,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "prediction_topk": args.prediction_topk,
        "lowest_likelihood_ratio": args.lowest_likelihood_ratio,
        "mmlu_token_scope": args.mmlu_token_scope,
        "seed": args.seed,
        "mmlu_role": "test",
        "gsm_role": "train",
        "mmlu_samples": [
            {"example_id": item.example_id, "source_row": item.source_row} for item in sampled_mmlu
        ],
        "gsm_samples": [
            {"example_id": item.example_id, "source_row": item.source_row} for item in sampled_gsm
        ],
    }
    with (output_dir / "sample_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"Loading model on {device}: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args, device)

    all_features: dict[str, list[RHFeature]] = {}
    for dataset_name, samples in (("mmlu", sampled_mmlu), ("gsm", sampled_gsm)):
        for variant in ("qa", "pr"):
            key = f"{dataset_name}_{variant}"
            print(f"Extracting {key}: {len(samples)} samples")
            all_features[key] = []
            for index, sample in enumerate(samples, start=1):
                text = sample.qa_text if variant == "qa" else sample.pr_text
                all_features[key].append(
                    extract_rh_feature(model, tokenizer, sample, text, args, device)
                )
                if index % 10 == 0 or index == len(samples):
                    print(f"  {key}: {index}/{len(samples)}")

    metadata = []
    metadata.extend(feature_metadata("mmlu", "question_answer", all_features["mmlu_qa"]))
    metadata.extend(feature_metadata("mmlu", "problem_result", all_features["mmlu_pr"]))
    metadata.extend(feature_metadata("gsm8k", "question_answer", all_features["gsm_qa"]))
    metadata.extend(feature_metadata("gsm8k", "problem_result", all_features["gsm_pr"]))
    write_jsonl(output_dir / "feature_metadata.jsonl", metadata)

    condition_features = {
        "MMLU Q/A__GSM Q/A": ("mmlu_qa", "gsm_qa"),
        "MMLU Q/A__GSM P/R": ("mmlu_qa", "gsm_pr"),
        "MMLU P/R__GSM Q/A": ("mmlu_pr", "gsm_qa"),
        "MMLU P/R__GSM P/R": ("mmlu_pr", "gsm_pr"),
    }
    matrices: dict[str, dict[str, np.ndarray]] = {}
    condition_stats: dict[str, dict[str, dict[str, float]]] = {}
    npz_payload: dict[str, np.ndarray] = {}
    for condition, (mmlu_key, gsm_key) in condition_features.items():
        condition_matrix = pairwise_matrices(all_features[mmlu_key], all_features[gsm_key])
        matrices[condition] = condition_matrix
        condition_stats[condition] = {
            metric: summarize_matrix(values) for metric, values in condition_matrix.items()
        }
        safe_condition = condition.lower().replace(" ", "_").replace("/", "").replace("__", "_x_")
        for metric, values in condition_matrix.items():
            npz_payload[f"{safe_condition}__{metric}"] = values
    np.savez_compressed(output_dir / "pairwise_matrices.npz", **npz_payload)

    contrasts = {
        "MMLU Q/A: GSM Q/A minus GSM P/R": (
            matrices["MMLU Q/A__GSM Q/A"],
            matrices["MMLU Q/A__GSM P/R"],
        ),
        "MMLU P/R: GSM P/R minus GSM Q/A": (
            matrices["MMLU P/R__GSM P/R"],
            matrices["MMLU P/R__GSM Q/A"],
        ),
    }
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for contrast_index, (contrast, (aligned, comparison)) in enumerate(contrasts.items()):
        deltas[contrast] = {}
        for metric in aligned:
            delta_matrix = aligned[metric] - comparison[metric]
            deltas[contrast][metric] = bootstrap_delta(
                delta_matrix,
                seed=args.seed + 1000 * (contrast_index + 1) + list(aligned).index(metric),
                num_samples=args.bootstrap_samples,
            )
            safe_contrast = "qa_primary" if contrast_index == 0 else "pr_control"
            npz_payload[f"{safe_contrast}__{metric}__delta"] = delta_matrix

    # Include contrasts in the same NPZ after all arrays are known.
    np.savez_compressed(output_dir / "pairwise_matrices.npz", **npz_payload)
    summary = {
        "manifest": manifest,
        "definition": {
            "h_dot": "dot(sum retained final-layer RH hidden states)",
            "h_cosine": "cosine(sum retained final-layer RH hidden states)",
            "prediction_error_dot": (
                "dot(sum(onehot(next_token)-p(next_token)) over the intersection "
                "of the two per-example top-k vocabularies)"
            ),
            "approximate_score": "h_dot * prediction_error_dot",
            "score_implementation": "Matches --readout_channels rh --approximate_proposed.",
        },
        "condition_stats": condition_stats,
        "paired_contrasts": deltas,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_markdown(output_dir / "report.md", summary, condition_stats, deltas)

    primary = deltas["MMLU Q/A: GSM Q/A minus GSM P/R"]
    print("Primary contrast (positive means GSM P/R is less similar to MMLU Q/A):")
    for metric, values in primary.items():
        print(
            f"  {metric}: delta={values['mean']:.6e}, "
            f"CI=[{values.get('ci95_low', float('nan')):.6e}, "
            f"{values.get('ci95_high', float('nan')):.6e}]"
        )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
