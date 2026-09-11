#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import random
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "eval"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from analysis.sequence_score.config import SequenceScoreConfig
from analysis.sequence_score.data import build_gsm8k_supervised_sample, build_mmlu_observation, load_hf_subset
from analysis.sequence_score.features import extract_observation_features, extract_update_features
from analysis.sequence_score.model_setup import load_base_model_and_template, model_architecture_summary
from analysis.sequence_score.pair_score import score_pair
from analysis.sequence_score.validation import run_closed_form_validations, validate_causal_alignment


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny correctness-first GSM8K -> MMLU sequence-score reference.")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--template", default="qwen3_nothink")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default="results/sequence_score/reference")
    parser.add_argument("--mmlu_subject", default="abstract_algebra")
    parser.add_argument("--num_mmlu", type=int, default=1)
    parser.add_argument("--num_gsm8k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--gradient_convention", choices=["logprob", "nll"], default="logprob")
    parser.add_argument("--kembd_mode", choices=["prefix", "full_sequence_scalar"], default="prefix")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    config = SequenceScoreConfig(gradient_convention=args.gradient_convention, kembd_mode=args.kembd_mode)

    loaded = load_base_model_and_template(
        repo_root=REPO_ROOT,
        model_name_or_path=args.base_model,
        template_name=args.template,
        cache_dir=args.cache_dir,
        cutoff_len=args.cutoff_len,
    )

    gsm8k_rows = load_hf_subset("openai/gsm8k", "main", "train", max(args.num_gsm8k * 5, args.num_gsm8k), args.cache_dir)
    mmlu_rows = load_hf_subset("cais/mmlu", args.mmlu_subject, "test", max(args.num_mmlu * 5, args.num_mmlu), args.cache_dir)
    gsm8k_indices = sorted(rng.sample(range(len(gsm8k_rows)), k=args.num_gsm8k))
    mmlu_indices = sorted(rng.sample(range(len(mmlu_rows)), k=args.num_mmlu))

    update_features = []
    update_diagnostics = []
    for idx in gsm8k_indices:
        sample = build_gsm8k_supervised_sample(
            repo_root=REPO_ROOT,
            raw_example=gsm8k_rows[idx],
            tokenizer=loaded.tokenizer,
            template=loaded.template,
            cutoff_len=args.cutoff_len,
            template_name=args.template,
        )
        features = extract_update_features(loaded.model, sample, config)
        update_features.append((idx, features))
        update_diagnostics.append(
            {
                "gsm8k_index": idx,
                "num_supervised_tokens": int(features.label_positions.numel()),
                "causal_alignment": validate_causal_alignment(features.label_positions, features.logit_positions),
                "first_label_token_id": int(features.target_ids[0].item()),
                "first_label_token_text": loaded.tokenizer.decode([int(features.target_ids[0].item())], skip_special_tokens=False),
            }
        )

    observation_features = []
    option_token_ids = {}
    for idx in mmlu_indices:
        observation = build_mmlu_observation(
            repo_root=REPO_ROOT,
            raw_example=mmlu_rows[idx],
            tokenizer=loaded.tokenizer,
            template=loaded.template,
            subject=args.mmlu_subject,
            source_index=idx,
        )
        features = extract_observation_features(loaded.model, loaded.tokenizer, loaded.template, observation, config)
        observation_features.append((idx, features))
        option_token_ids[f"{args.mmlu_subject}:{idx}"] = features.option_token_info

    pair_rows = []
    for gsm8k_idx, update in update_features:
        for mmlu_idx, observation in observation_features:
            result = score_pair(update, observation, config)
            pair_rows.append(
                {
                    "gsm8k_index": gsm8k_idx,
                    "mmlu_subject": args.mmlu_subject,
                    "mmlu_index": mmlu_idx,
                    "num_update_tokens": int(update.label_positions.numel()),
                    **{f"uniform_{k}": v for k, v in result["uniform_options"].to_dict().items() if k != "objective"},
                    **{f"ifmass_{k}": v for k, v in result["ifmass"].to_dict().items() if k != "objective"},
                    "diagnostics": result["diagnostics"],
                }
            )

    write_json(
        output_dir / "config.json",
        {
            "base_model": args.base_model,
            "template": args.template,
            "mmlu_subject": args.mmlu_subject,
            "num_mmlu": args.num_mmlu,
            "num_gsm8k": args.num_gsm8k,
            "seed": args.seed,
            "cutoff_len": args.cutoff_len,
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
            "chat_wrapper": "src/llamafactory/chat/hf_engine.py:HuggingfaceEngine._process_args",
            "model_loader": "src/llamafactory/model/loader.py:load_model",
        },
    )
    write_json(output_dir / "sampled_gsm8k_indices.json", {"indices": gsm8k_indices})
    write_json(output_dir / "sampled_mmlu_indices.json", {"subject": args.mmlu_subject, "indices": mmlu_indices})
    write_json(output_dir / "option_token_ids.json", option_token_ids)
    write_jsonl(output_dir / "pair_scores.jsonl", pair_rows)
    write_json(
        output_dir / "diagnostics.json",
        {
            "architecture": model_architecture_summary(loaded.model),
            "closed_form_gradients": run_closed_form_validations(),
            "updates": update_diagnostics,
            "num_pairs": len(pair_rows),
        },
    )

    print(json.dumps({"output_dir": str(output_dir), "num_pairs": len(pair_rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
