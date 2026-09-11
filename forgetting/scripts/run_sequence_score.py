#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "eval"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from analysis.sequence_score.config import SequenceScoreConfig
from analysis.sequence_score.data import build_gsm8k_supervised_sample, build_mmlu_observation, load_hf_subset
from analysis.sequence_score.features import ObservationFeatures, extract_observation_features, extract_update_features
from analysis.sequence_score.model_setup import load_base_model_and_template, model_architecture_summary
from analysis.sequence_score.validation import run_closed_form_validations, validate_causal_alignment
from analysis.sequence_score.vectorized import aggregate_pair_rows, score_update_against_observations


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_mmlu_all(cache_dir: str | None, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    candidates = [
        ("cais/mmlu", "all", split),
        ("cais/mmlu", "all", "validation" if split == "test" else split),
        ("lukaemon/mmlu", "all", split),
    ]
    errors = []
    for path, name, candidate_split in candidates:
        try:
            kwargs: dict[str, Any] = {"path": path, "name": name, "split": candidate_split}
            if cache_dir is not None:
                kwargs["cache_dir"] = cache_dir
            dataset = load_dataset(**kwargs)
            return [dict(dataset[i]) for i in range(len(dataset))]
        except Exception as exc:
            errors.append(f"{path}/{name}:{candidate_split}: {exc}")
    raise RuntimeError("Could not load MMLU all split:\n" + "\n".join(errors))


def sample_mmlu_by_subject(rows: list[dict[str, Any]], examples_per_subject: int, seed: int) -> dict[str, list[int]]:
    by_subject: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        subject = row.get("subject") or row.get("category") or "unknown"
        by_subject.setdefault(str(subject), []).append(idx)

    rng = random.Random(seed)
    sampled = {}
    for subject, indices in sorted(by_subject.items()):
        if len(indices) < examples_per_subject:
            raise ValueError(f"Subject {subject} has only {len(indices)} examples, need {examples_per_subject}.")
        sampled[subject] = sorted(rng.sample(indices, k=examples_per_subject))
    return sampled


def all_mmlu_by_subject(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    by_subject: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        subject = row.get("subject") or row.get("category") or "unknown"
        by_subject.setdefault(str(subject), []).append(idx)
    return dict(sorted(by_subject.items()))


def _mean_tensor(items: list[torch.Tensor], name: str) -> torch.Tensor:
    if not items:
        raise ValueError(f"Cannot average empty {name}.")
    return torch.stack([item.float() for item in items], dim=0).mean(dim=0)


def mean_observation_features(subject: str, features: list[ObservationFeatures]) -> ObservationFeatures:
    if not features:
        raise ValueError(f"Cannot build subject mean for empty subject {subject}.")
    representative = features[0]
    source_indices = [int(feature.metadata["source_index"]) for feature in features]
    target_counts: dict[str, int] = {}
    for feature in features:
        target = str(feature.metadata.get("target", ""))
        target_counts[target] = target_counts.get(target, 0) + 1

    return ObservationFeatures(
        prompt_ids=representative.prompt_ids,
        attention_mask=representative.attention_mask,
        option_token_ids=representative.option_token_ids,
        option_token_info=representative.option_token_info,
        option_gradients=_mean_tensor([feature.option_gradients for feature in features], "option gradients"),
        projected_option_gradients=_mean_tensor(
            [feature.projected_option_gradients for feature in features],
            "projected option gradients",
        ),
        ifmass_gradient=_mean_tensor([feature.ifmass_gradient for feature in features], "ifmass gradients"),
        projected_ifmass_gradient=_mean_tensor(
            [feature.projected_ifmass_gradient for feature in features],
            "projected ifmass gradients",
        ),
        hidden=_mean_tensor([feature.hidden for feature in features], "hidden states"),
        logits=_mean_tensor([feature.logits for feature in features], "logits"),
        metadata={
            "dataset_name": "mmlu",
            "subject": subject,
            "source_index": "subject_mean",
            "observation_mode": "subject_mean",
            "num_source_examples": len(features),
            "source_indices": source_indices,
            "representative_source_index": int(representative.metadata["source_index"]),
            "target_counts": target_counts,
            "target": "subject_mean",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cached/vectorized full GSM8K -> MMLU sequence-score experiment.")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--template", default="qwen3_nothink")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default="results/sequence_score/full_qwen3_4b")
    parser.add_argument("--mmlu_split", default="test")
    parser.add_argument("--num_mmlu_per_subject", type=int, default=25)
    parser.add_argument("--mmlu_observation_mode", choices=["sampled", "subject_mean"], default="sampled")
    parser.add_argument("--num_gsm8k", type=int, default=100)
    parser.add_argument("--gsm8k_candidate_pool_size", type=int, default=None, help="Restrict sampling to the first N GSM8K rows; use 300 to reproduce the reported runs.")
    parser.add_argument("--use_all_gsm8k", action="store_true", help="Use every GSM8K train example instead of sampling --num_gsm8k examples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--gradient_convention", choices=["logprob", "nll"], default="logprob")
    parser.add_argument("--kembd_mode", choices=["prefix", "full_sequence_scalar"], default="prefix")
    parser.add_argument("--update_token_mode", choices=["all_supervised", "first_final_answer_token", "first_response_token"], default="all_supervised")
    parser.add_argument("--final_answer_markers", nargs="+", default=["####", "Answer:"])
    parser.add_argument("--observation_chunk_size", type=int, default=16)
    parser.add_argument("--max_subjects", type=int, default=None, help="Debug option; leave unset for all subjects.")
    args = parser.parse_args()

    start_time = time.time()
    output_dir = Path(args.output_dir)
    config = SequenceScoreConfig(
        gradient_convention=args.gradient_convention,
        kembd_mode=args.kembd_mode,
        update_token_mode=args.update_token_mode,
        final_answer_markers=tuple(args.final_answer_markers),
    )

    loaded = load_base_model_and_template(
        repo_root=REPO_ROOT,
        model_name_or_path=args.base_model,
        template_name=args.template,
        cache_dir=args.cache_dir,
        cutoff_len=args.cutoff_len,
    )

    mmlu_rows = load_mmlu_all(args.cache_dir, args.mmlu_split)
    if args.mmlu_observation_mode == "subject_mean":
        sampled_mmlu = all_mmlu_by_subject(mmlu_rows)
    else:
        sampled_mmlu = sample_mmlu_by_subject(mmlu_rows, args.num_mmlu_per_subject, args.seed)
    if args.max_subjects is not None:
        sampled_mmlu = dict(list(sampled_mmlu.items())[: args.max_subjects])

    gsm8k_rows = load_hf_subset(
        "openai/gsm8k",
        "main",
        "train",
        None if args.use_all_gsm8k else args.gsm8k_candidate_pool_size,
        args.cache_dir,
    )
    rng = random.Random(args.seed)
    if args.use_all_gsm8k:
        gsm8k_indices = list(range(len(gsm8k_rows)))
    else:
        gsm8k_indices = sorted(rng.sample(range(len(gsm8k_rows)), k=args.num_gsm8k))

    write_json(
        output_dir / "config.json",
        {
            "base_model": args.base_model,
            "template": args.template,
            "mmlu_split": args.mmlu_split,
            "num_mmlu_per_subject": args.num_mmlu_per_subject,
            "mmlu_observation_mode": args.mmlu_observation_mode,
            "mmlu_subject_example_counts": {subject: len(indices) for subject, indices in sampled_mmlu.items()},
            "num_gsm8k": len(gsm8k_indices),
            "requested_num_gsm8k": args.num_gsm8k,
            "gsm8k_candidate_pool_size": args.gsm8k_candidate_pool_size,
            "use_all_gsm8k": args.use_all_gsm8k,
            "seed": args.seed,
            "cutoff_len": args.cutoff_len,
            "observation_chunk_size": args.observation_chunk_size,
            "update_token_mode": args.update_token_mode,
            "final_answer_markers": list(args.final_answer_markers),
            "score_config": config.to_dict(),
        },
    )
    write_json(
        output_dir / "repository_mapping.json",
        {
            "gsm8k_dataset_info": "data_eaft/dataset_info.json:gsm8k_sft",
            "gsm8k_converter": "src/llamafactory/data/converter.py:AlpacaDatasetConverter",
            "gsm8k_processor": "src/llamafactory/data/processor/supervised.py:SupervisedDatasetProcessor",
            "mmlu_prompt": "eval/prepare_datasets.py:_format_mmlu_prompt",
            "model_loader": "src/llamafactory/model/loader.py:load_model",
            "reference_math": "analysis/sequence_score/components.py:score_from_components",
        },
    )
    write_json(output_dir / "sampled_gsm8k_indices.json", {"indices": gsm8k_indices})
    write_json(output_dir / "sampled_mmlu_indices.json", sampled_mmlu)

    observations = []
    option_token_ids = {}
    subject_mean_rows = []
    for subject, indices in sampled_mmlu.items():
        subject_features = []
        for idx in indices:
            observation = build_mmlu_observation(
                repo_root=REPO_ROOT,
                raw_example=mmlu_rows[idx],
                tokenizer=loaded.tokenizer,
                template=loaded.template,
                subject=subject,
                source_index=idx,
            )
            features = extract_observation_features(loaded.model, loaded.tokenizer, loaded.template, observation, config)
            features.metadata["target"] = observation.target
            if args.mmlu_observation_mode == "subject_mean":
                subject_features.append(features)
            else:
                observations.append(features)
            option_token_ids[f"{subject}:{idx}"] = features.option_token_info

        if args.mmlu_observation_mode == "subject_mean":
            mean_features = mean_observation_features(subject, subject_features)
            observations.append(mean_features)
            subject_mean_rows.append(
                {
                    "subject": subject,
                    "mmlu_subject": subject,
                    "num_mmlu_examples": len(subject_features),
                    "representative_source_index": mean_features.metadata["representative_source_index"],
                    "source_index_min": min(mean_features.metadata["source_indices"]),
                    "source_index_max": max(mean_features.metadata["source_indices"]),
                    "target_counts": json.dumps(mean_features.metadata["target_counts"], sort_keys=True),
                }
            )

    write_json(output_dir / "option_token_ids.json", option_token_ids)
    if subject_mean_rows:
        write_csv(output_dir / "subject_mean_metadata.csv", subject_mean_rows)

    pair_rows = []
    update_diagnostics = []
    for update_number, idx in enumerate(gsm8k_indices):
        update_start = time.time()
        sample = build_gsm8k_supervised_sample(
            repo_root=REPO_ROOT,
            raw_example=gsm8k_rows[idx],
            tokenizer=loaded.tokenizer,
            template=loaded.template,
            cutoff_len=args.cutoff_len,
            template_name=args.template,
        )
        update = extract_update_features(loaded.model, sample, config, tokenizer=loaded.tokenizer)
        rows = score_update_against_observations(
            update,
            observations,
            config,
            observation_chunk_size=args.observation_chunk_size,
        )
        for row in rows:
            row["gsm8k_index"] = idx
            row["update_number"] = update_number
        pair_rows.extend(rows)
        update_diagnostics.append(
            {
                "gsm8k_index": idx,
                "update_number": update_number,
                "num_supervised_tokens": int(update.label_positions.numel()),
                "update_token_mode": update.metadata.get("update_token_mode"),
                "selected_token_text": update.metadata.get("selected_token_text"),
                "selected_token_id": update.metadata.get("selected_token_id"),
                "selected_label_position": update.metadata.get("selected_label_position"),
                "selected_supervised_offset": update.metadata.get("selected_supervised_offset"),
                "num_original_supervised_tokens": update.metadata.get("num_original_supervised_tokens"),
                "causal_alignment": validate_causal_alignment(update.label_positions, update.logit_positions),
                "runtime_seconds": time.time() - update_start,
            }
        )
        write_jsonl(output_dir / "pair_scores.jsonl", pair_rows)

    example_rows, subject_rows = aggregate_pair_rows(pair_rows)
    write_csv(output_dir / "pair_scores.csv", pair_rows)
    write_csv(output_dir / "mmlu_example_scores.csv", example_rows)
    write_csv(output_dir / "subject_scores.csv", subject_rows)
    write_json(
        output_dir / "diagnostics.json",
        {
            "architecture": model_architecture_summary(loaded.model),
            "closed_form_gradients": run_closed_form_validations(),
            "num_observations": len(observations),
            "num_pairs": len(pair_rows),
            "num_subjects": len(sampled_mmlu),
            "updates": update_diagnostics,
            "runtime_seconds": time.time() - start_time,
        },
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_subjects": len(sampled_mmlu),
                "num_observations": len(observations),
                "num_pairs": len(pair_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
