"""Shapley attribution of exact CH1 template-score changes to H versus error.

For the original RH/CH1 score, R(v)=sum_t e_t(v) h_t and score is the aligned
dot product of R vectors.  This script preserves that exact score and, for
paired GSM Q/A and P/R sequences of the same token length, constructs the two
counterfactual representations E_QA^T H_PR and E_PR^T H_QA.  The average of
the two update orders gives an exact Shapley attribution of S_PR-S_QA to H and
prediction error E.  MMLU retains only its gold answer prediction position.
"""

from __future__ import annotations

import argparse
import contextlib
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
    DEFAULT_ARTIFACTS
    / "score_analysis_qwen25_base_mmlu5000_exact_ch1_component_attribution_answer_first_n50_seed42"
)


@dataclass
class TokenFeature:
    example_id: str
    token_count: int
    kept_token_count: int
    answer_label_index: int | None
    vocab_ids: torch.Tensor  # sorted [V]
    hidden: torch.Tensor  # float32 [T, D]
    error: torch.Tensor  # float32 [T, V]


@dataclass
class Representation:
    vocab_ids: torch.Tensor
    values: torch.Tensor  # bfloat16 [V, D], exactly the original CH1 representation format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute exact CH1 template-score changes to GSM hidden states versus errors."
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
        "--gsm_token_scope",
        choices=("full", "completion"),
        default="full",
        help="GSM positions: all prompt+completion tokens, or only gold completion tokens.",
    )
    parser.add_argument(
        "--normalize_hidden",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "L2-normalize each retained final RH hidden state before forming R(v). "
            "This is a scale-controlled diagnostic, not the raw original CH1 score."
        ),
    )
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def completion_label_indices(
    tokenizer, text: str, completion_char_start: int, max_length: int
) -> torch.Tensor:
    """Return retained label/logit positions for every token in a gold completion."""
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoded["input_ids"])
    completion_token_input_indices = [
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if end > start and end > completion_char_start
    ]
    if not completion_token_input_indices:
        raise ValueError("Could not locate GSM completion tokens from tokenizer offsets.")
    if tokenizer.eos_token_id is not None and (
        not token_ids or token_ids[-1] != tokenizer.eos_token_id
    ):
        token_ids.append(tokenizer.eos_token_id)
    trim_start = max(0, len(token_ids) - max_length)
    retained_length = min(len(token_ids), max_length)
    label_indices = [
        token_input_index - trim_start - 1
        for token_input_index in completion_token_input_indices
    ]
    label_indices = [
        index for index in label_indices if 0 <= index < retained_length - 1
    ]
    if not label_indices:
        raise ValueError("All GSM completion tokens were truncated or lack prediction positions.")
    return torch.tensor(label_indices, dtype=torch.long)

def extract_token_feature(
    model, tokenizer, sample: SamplePair, text: str, is_mmlu: bool, args, device: str
) -> TokenFeature:
    input_ids, _ = encode_text(tokenizer, text, args.max_length)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        logits, hidden, _ = forward_logits_and_hidden(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            gh_embedding_layers=None,
        )
    logits = logits[:, :-1, :].float()
    hidden = hidden[:, :-1, :].float()
    labels = input_ids[:, 1:]
    valid_mask = attention_mask[:, :-1].bool()
    keep_mask, log_denom = build_low_likelihood_position_mask(
        logits, labels, valid_mask, args.lowest_likelihood_ratio
    )
    answer_label_index = None
    score_mask = keep_mask
    if is_mmlu:
        answer_char_start = (
            sample.qa_answer_char_start if text == sample.qa_text else sample.pr_answer_char_start
        )
        answer_label_index = mmlu_answer_label_index(
            tokenizer, text, answer_char_start, args.max_length
        )
        if not keep_mask[0, answer_label_index]:
            raise ValueError("Answer position removed; use --lowest_likelihood_ratio 1.0.")
        score_mask = torch.zeros_like(keep_mask)
        score_mask[0, answer_label_index] = True
    elif args.gsm_token_scope == "completion":
        marker = "\n\nAnswer:" if text == sample.qa_text else "\n\nresult:"
        marker_index = text.rfind(marker)
        if marker_index < 0:
            raise ValueError(f"Could not locate GSM terminal marker {marker!r}.")
        completion_indices = completion_label_indices(
            tokenizer, text, marker_index + len(marker), args.max_length
        )
        if not bool(keep_mask[0, completion_indices].all()):
            raise ValueError("Completion position removed; use --lowest_likelihood_ratio 1.0.")
        score_mask = torch.zeros_like(keep_mask)
        score_mask[0, completion_indices] = True
    indices = torch.nonzero(score_mask[0], as_tuple=False).squeeze(-1)
    vocab_ids = build_batch_vocabulary(logits, score_mask, args.prediction_topk)
    selected_logits = logits[0, indices].index_select(-1, vocab_ids)
    selected_log_denom = log_denom[0, indices]
    if selected_log_denom.ndim == 1:
        selected_log_denom = selected_log_denom.unsqueeze(-1)
    probability = torch.exp(selected_logits - selected_log_denom)
    observed = (labels[0, indices].unsqueeze(-1) == vocab_ids.unsqueeze(0)).to(probability.dtype)
    error = torch.nan_to_num(observed - probability, nan=0.0, posinf=0.0, neginf=0.0)
    retained_hidden = torch.nan_to_num(hidden[0, indices], nan=0.0, posinf=0.0, neginf=0.0)
    if args.normalize_hidden:
        retained_hidden = retained_hidden / retained_hidden.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    feature = TokenFeature(
        example_id=sample.example_id,
        token_count=int(input_ids.shape[1]),
        kept_token_count=int(indices.numel()),
        answer_label_index=answer_label_index,
        vocab_ids=vocab_ids.detach().cpu().to(torch.int64),
        hidden=retained_hidden.detach().cpu().float(),
        error=error.detach().cpu().float(),
    )
    del logits, hidden, probability, error
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return feature


def make_representation(vocab_ids: torch.Tensor, error: torch.Tensor, hidden: torch.Tensor) -> Representation:
    values = torch.nan_to_num(error.transpose(0, 1) @ hidden, nan=0.0, posinf=0.0, neginf=0.0)
    return Representation(vocab_ids=vocab_ids, values=values.to(torch.bfloat16))


def expand_error(feature: TokenFeature, union_vocab: torch.Tensor) -> torch.Tensor:
    positions = torch.searchsorted(union_vocab, feature.vocab_ids)
    expanded = torch.zeros(
        (feature.error.shape[0], union_vocab.numel()), dtype=torch.float32
    )
    expanded[:, positions] = feature.error
    return expanded


def aligned_dot(left: Representation, right: Representation) -> float:
    positions = torch.searchsorted(right.vocab_ids, left.vocab_ids)
    valid = positions < right.vocab_ids.numel()
    if not valid.any():
        return 0.0
    left_index = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    right_index = positions[valid]
    matched = right.vocab_ids[right_index] == left.vocab_ids[left_index]
    if not matched.any():
        return 0.0
    return float(
        torch.sum(
            left.values[left_index[matched]].float() * right.values[right_index[matched]].float()
        ).item()
    )


def gsm_counterfactual_representations(qa: TokenFeature, pr: TokenFeature) -> dict[str, Representation]:
    if qa.example_id != pr.example_id:
        raise ValueError("GSM Q/A and P/R examples are not paired.")
    if qa.hidden.shape != pr.hidden.shape:
        raise ValueError(
            f"GSM token positions do not align for {qa.example_id}: {qa.hidden.shape} vs {pr.hidden.shape}"
        )
    vocab = torch.unique(torch.cat((qa.vocab_ids, pr.vocab_ids)), sorted=True)
    qa_error = expand_error(qa, vocab)
    pr_error = expand_error(pr, vocab)
    return {
        "qa": make_representation(vocab, qa_error, qa.hidden),
        "error_pr_hidden_qa": make_representation(vocab, pr_error, qa.hidden),
        "error_qa_hidden_pr": make_representation(vocab, qa_error, pr.hidden),
        "pr": make_representation(vocab, pr_error, pr.hidden),
    }


def bootstrap(matrices: dict[str, np.ndarray], seed: int, count: int) -> dict[str, dict[str, float]]:
    result = {name: {"mean": float(value.mean())} for name, value in matrices.items()}
    if count <= 0:
        return result
    shape_set = {value.shape for value in matrices.values()}
    if len(shape_set) != 1:
        raise ValueError("All matrices must share a shape for cluster bootstrap.")
    rows, columns = next(iter(shape_set))
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(count) for name in matrices}
    for draw in range(count):
        row_index = rng.integers(0, rows, size=rows)
        column_index = rng.integers(0, columns, size=columns)
        for name, value in matrices.items():
            samples[name][draw] = value[np.ix_(row_index, column_index)].mean()
    for name, values in samples.items():
        result[name]["ci95_low"] = float(np.quantile(values, 0.025))
        result[name]["ci95_high"] = float(np.quantile(values, 0.975))
    return result


def calculate_condition(
    mmlu: list[TokenFeature], gsm_qa: list[TokenFeature], gsm_pr: list[TokenFeature]
) -> dict[str, np.ndarray]:
    size = (len(mmlu), len(gsm_qa))
    scores = {name: np.empty(size, dtype=np.float64) for name in ("score_qa", "error_only", "hidden_only", "score_pr")}
    for column, (qa, pr) in enumerate(zip(gsm_qa, gsm_pr, strict=True)):
        gsm_repr = gsm_counterfactual_representations(qa, pr)
        for row, mmlu_item in enumerate(mmlu):
            mmlu_repr = make_representation(mmlu_item.vocab_ids, mmlu_item.error, mmlu_item.hidden)
            for name, value in gsm_repr.items():
                target = {
                    "qa": "score_qa",
                    "error_pr_hidden_qa": "error_only",
                    "error_qa_hidden_pr": "hidden_only",
                    "pr": "score_pr",
                }[name]
                scores[target][row, column] = aligned_dot(mmlu_repr, value)
    delta = scores["score_pr"] - scores["score_qa"]
    hidden_contribution = 0.5 * (
        (scores["hidden_only"] - scores["score_qa"])
        + (scores["score_pr"] - scores["error_only"])
    )
    error_contribution = 0.5 * (
        (scores["error_only"] - scores["score_qa"])
        + (scores["score_pr"] - scores["hidden_only"])
    )
    if not np.allclose(delta, hidden_contribution + error_contribution, rtol=1e-10, atol=1e-5):
        raise AssertionError("Exact CH1 Shapley components do not sum to score delta.")
    scores.update(
        {
            "score_delta_pr_minus_qa": delta,
            "hidden_contribution": hidden_contribution,
            "prediction_error_contribution": error_contribution,
        }
    )
    return scores


def report(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Exact CH1 H versus prediction-error attribution",
        "",
        (
            "MMLU retains its gold answer prediction token; GSM retains only its gold completion after the terminal Answer:/result: marker."
            if results["gsm_token_scope"] == "completion"
            else "MMLU retains only its gold answer prediction token; GSM retains all positions."
        ),
        "GSM Q/A and P/R have identical retained token lengths for all sampled pairs, so H and E are counterfactually recombined at matching token positions.",
        (
            "Each retained final RH hidden state is L2-normalized before R(v) is formed."
            if results["hidden_normalization"] == "l2_per_token"
            else "Final RH hidden states retain their original scale."
        ),
        "",
        "Direction is GSM P/R minus GSM Q/A. A positive delta means P/R increases the exact CH1 score (i.e., is less negative / more protective under the local first-order likelihood interpretation).",
        "",
        "| MMLU template | Quantity | Mean | 95% cluster-bootstrap CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for template, result in results["conditions"].items():
        for name in ("score_delta_pr_minus_qa", "hidden_contribution", "prediction_error_contribution"):
            values = result["bootstrap"][name]
            lines.append(
                f"| MMLU {template.upper()} | {name} | {values['mean']:.5e} | "
                f"[{values['ci95_low']:.5e}, {values['ci95_high']:.5e}] |"
            )
        lines.append("|  |  |  |  |")
    lines.extend(
        [
            "",
            "The attribution is exact: ΔS = ΔS_H + ΔS_E. It averages the H-first and E-first counterfactual paths, so it has no ordering preference.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.prediction_topk <= 0 or not 0 < args.lowest_likelihood_ratio <= 1:
        raise ValueError("Invalid --prediction_topk or --lowest_likelihood_ratio.")
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
        build_paired_samples(data_dir / "gsm8k_question_answer.jsonl", data_dir / "gsm8k_problem_result.jsonl", "gsm8k"),
        args.num_gsm,
        args.seed,
        "gsm8k",
    )
    device = resolve_device(args.device)
    print(f"Loading model on {device}: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args, device)
    features: dict[str, list[TokenFeature]] = {}
    for dataset_name, samples in (("mmlu", mmlu), ("gsm", gsm)):
        for variant in ("qa", "pr"):
            key = f"{dataset_name}_{variant}"
            features[key] = []
            print(f"Extracting {key}: {len(samples)}")
            for index, item in enumerate(samples, start=1):
                text = item.qa_text if variant == "qa" else item.pr_text
                features[key].append(
                    extract_token_feature(model, tokenizer, item, text, dataset_name == "mmlu", args, device)
                )
                if index % 10 == 0 or index == len(samples):
                    print(f"  {key}: {index}/{len(samples)}")

    conditions = {
        "qa": calculate_condition(features["mmlu_qa"], features["gsm_qa"], features["gsm_pr"]),
        "pr": calculate_condition(features["mmlu_pr"], features["gsm_qa"], features["gsm_pr"]),
    }
    summary_conditions = {}
    payload = {}
    for index, (template, matrices) in enumerate(conditions.items()):
        summary_conditions[template] = {
            "bootstrap": bootstrap(matrices, args.seed + 1000 * index, args.bootstrap_samples)
        }
        for name, matrix in matrices.items():
            payload[f"mmlu_{template}__{name}"] = matrix
    np.savez_compressed(output_dir / "component_attribution_matrices.npz", **payload)
    summary = {
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "seed": args.seed,
        "num_mmlu": args.num_mmlu,
        "num_gsm": args.num_gsm,
        "mmlu_token_scope": "answer_first",
        "gsm_token_scope": args.gsm_token_scope,
        "hidden_normalization": "l2_per_token" if args.normalize_hidden else "none",
        "score": (
            "exact CH1/RH with per-token L2-normalized hidden states, not h_dot * error_dot"
            if args.normalize_hidden
            else "exact original CH1/RH, not h_dot * error_dot"
        ),
        "attribution": "Shapley average of H-first and E-first counterfactual paths; delta = hidden + prediction_error.",
        "conditions": summary_conditions,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report(output_dir / "report.md", summary)
    primary = summary_conditions["qa"]["bootstrap"]
    print("MMLU Q/A exact CH1 attribution (GSM P/R minus Q/A):")
    for name in ("score_delta_pr_minus_qa", "hidden_contribution", "prediction_error_contribution"):
        values = primary[name]
        print(f"  {name}: {values['mean']:.6e} CI=[{values['ci95_low']:.6e}, {values['ci95_high']:.6e}]")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
