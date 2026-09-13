"""Compare the GH LM-head-projected prediction-error component without h.

For every sample this diagnostic computes

    g_sum = sum_{t,v} (onehot(y_t=v) - p_t(v)) W_v,

where W is the LM-head weight and v is the same per-sample top-k vocabulary as
``forvalue_streaming_ghrh.py``.  This is the token-summed version of that
script's ``gh_readout_error``.  It intentionally does not use h, so it is not
an exact separable GH score component; the exact GH score pairs every g_t with
its corresponding h_t.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from analyze_template_similarity import (
    DEFAULT_ARTIFACTS,
    DEFAULT_MODEL,
    SamplePair,
    bootstrap_delta,
    build_paired_samples,
    choose_samples,
    extract_rh_feature,
    load_model_and_tokenizer,
    resolve_device,
)
from forvalue_streaming_ghrh import get_lm_head_weight


DEFAULT_OUTPUT = (
    DEFAULT_ARTIFACTS / "score_analysis_qwen25_base_mmlu5000_gh_error_readout_n50_seed42"
)


@dataclass
class GHErrorFeature:
    example_id: str
    source_row: int | None
    token_count: int
    kept_token_count: int
    g_sum: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GH LM-head-projected prediction-error similarity without h."
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
        "--mmlu_token_scope", choices=("full", "answer_first"), default="answer_first"
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def project_gh_error(feature, lm_head_weight: torch.Tensor, device: str) -> GHErrorFeature:
    vocab_ids = feature.error_vocab_ids.to(device=device, dtype=torch.long)
    selected_weight = lm_head_weight.index_select(0, vocab_ids).float()
    g_sum = torch.matmul(feature.error_sum.to(device=device, dtype=torch.float32), selected_weight)
    return GHErrorFeature(
        example_id=feature.example_id,
        source_row=feature.source_row,
        token_count=feature.token_count,
        kept_token_count=feature.kept_token_count,
        g_sum=torch.nan_to_num(g_sum, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu(),
    )


def pairwise_matrices(
    mmlu_features: list[GHErrorFeature], gsm_features: list[GHErrorFeature]
) -> dict[str, np.ndarray]:
    mmlu = torch.stack([item.g_sum for item in mmlu_features]).float()
    gsm = torch.stack([item.g_sum for item in gsm_features]).float()
    dot = torch.matmul(mmlu, gsm.transpose(0, 1)).numpy().astype(np.float64)
    mmlu_unit = mmlu / mmlu.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gsm_unit = gsm / gsm.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosine = torch.matmul(mmlu_unit, gsm_unit.transpose(0, 1)).numpy().astype(np.float64)
    return {"gh_error_dot": dot, "gh_error_cosine": cosine}


def summarize(matrix: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(matrix.mean()),
        "median": float(np.median(matrix)),
        "std": float(matrix.std(ddof=1)),
    }


def write_report(
    path: Path,
    stats: dict[str, dict[str, dict[str, float]]],
    contrasts: dict[str, dict[str, dict[str, float]]],
) -> None:
    lines = [
        "# GH LM-head-projected prediction-error similarity (no h)",
        "",
        "Each vector is `g_sum = sum_t,v (onehot(y_t=v) - p_t(v)) W_v`. MMLU uses only the gold-option prediction position; GSM8K uses all retained positions. h is not used.",
        "",
        "## Mean pairwise similarity",
        "",
        "| MMLU template × GSM template | g dot | cosine(g) |",
        "| --- | ---: | ---: |",
    ]
    for condition, values in stats.items():
        lines.append(
            f"| {condition.replace('__', ' × ')} | {values['gh_error_dot']['mean']:.5e} | "
            f"{values['gh_error_cosine']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired template contrasts",
            "",
            "Positive delta means GSM P/R is less similar than GSM Q/A.",
            "",
            "| Contrast | metric | delta | 95% cluster-bootstrap CI |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for contrast, metrics in contrasts.items():
        for metric, values in metrics.items():
            interval = (
                f"[{values['ci95_low']:.5e}, {values['ci95_high']:.5e}]"
                if "ci95_low" in values else "not requested"
            )
            lines.append(
                f"| {contrast} | {metric} | {values['mean']:.5e} | "
                f"{interval} |"
            )
    lines.extend(
        [
            "",
            "This is a pure GH readout-error diagnostic, not an exact GH score: the scorer's exact representation additionally preserves tokenwise coupling to h.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.lowest_likelihood_ratio <= 1.0:
        raise ValueError("--lowest_likelihood_ratio must be in (0, 1].")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mmlu = choose_samples(
        build_paired_samples(data_dir / "mmlu.jsonl", data_dir / "mmlu_problem_result.jsonl", "mmlu"),
        args.num_mmlu,
        args.seed,
        "mmlu",
    )
    gsm = choose_samples(
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
    lm_head_weight = get_lm_head_weight(model).detach()
    features: dict[str, list[GHErrorFeature]] = {}
    for dataset_name, samples in (("mmlu", mmlu), ("gsm", gsm)):
        for variant in ("qa", "pr"):
            key = f"{dataset_name}_{variant}"
            print(f"Extracting/projecting {key}: {len(samples)} samples")
            features[key] = []
            for index, sample in enumerate(samples, start=1):
                text = sample.qa_text if variant == "qa" else sample.pr_text
                rh_feature = extract_rh_feature(model, tokenizer, sample, text, args, device)
                features[key].append(project_gh_error(rh_feature, lm_head_weight, device))
                if index % 10 == 0 or index == len(samples):
                    print(f"  {key}: {index}/{len(samples)}")

    conditions = {
        "MMLU Q/A__GSM Q/A": ("mmlu_qa", "gsm_qa"),
        "MMLU Q/A__GSM P/R": ("mmlu_qa", "gsm_pr"),
        "MMLU P/R__GSM Q/A": ("mmlu_pr", "gsm_qa"),
        "MMLU P/R__GSM P/R": ("mmlu_pr", "gsm_pr"),
    }
    matrices = {
        name: pairwise_matrices(features[mmlu_key], features[gsm_key])
        for name, (mmlu_key, gsm_key) in conditions.items()
    }
    stats = {
        condition: {metric: summarize(values) for metric, values in result.items()}
        for condition, result in matrices.items()
    }
    contrast_inputs = {
        "MMLU Q/A: GSM Q/A minus GSM P/R": (
            matrices["MMLU Q/A__GSM Q/A"], matrices["MMLU Q/A__GSM P/R"]
        ),
        "MMLU P/R: GSM P/R minus GSM Q/A": (
            matrices["MMLU P/R__GSM P/R"], matrices["MMLU P/R__GSM Q/A"]
        ),
    }
    contrasts: dict[str, dict[str, dict[str, float]]] = {}
    payload: dict[str, np.ndarray] = {}
    for contrast_index, (name, (aligned, comparison)) in enumerate(contrast_inputs.items()):
        contrasts[name] = {}
        for metric, values in aligned.items():
            delta = values - comparison[metric]
            contrasts[name][metric] = bootstrap_delta(
                delta, args.seed + 100 * contrast_index + list(aligned).index(metric), args.bootstrap_samples
            )
            payload[f"contrast_{contrast_index}_{metric}"] = delta
    for condition, result in matrices.items():
        prefix = condition.lower().replace(" ", "_").replace("/", "").replace("__", "_x_")
        for metric, values in result.items():
            payload[f"{prefix}__{metric}"] = values
    np.savez_compressed(output_dir / "pairwise_matrices.npz", **payload)
    summary = {
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "mmlu_token_scope": args.mmlu_token_scope,
        "definition": "g_sum = sum_t,v (onehot(y_t=v)-p_t(v)) W_v; h is excluded.",
        "condition_stats": stats,
        "paired_contrasts": contrasts,
        "mmlu_kept_token_counts": [item.kept_token_count for item in features["mmlu_qa"]],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir / "report.md", stats, contrasts)
    primary = contrasts["MMLU Q/A: GSM Q/A minus GSM P/R"]
    print("Primary contrast (positive means GSM P/R is less similar to MMLU Q/A):")
    for metric, values in primary.items():
        interval = f", CI=[{values['ci95_low']:.6e}, {values['ci95_high']:.6e}]" if "ci95_low" in values else ""
        print(f"  {metric}: {values['mean']:.6e}{interval}")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
