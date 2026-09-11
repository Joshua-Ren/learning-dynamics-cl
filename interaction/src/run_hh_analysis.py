import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from utils.config_read import load_config
from utils.hh_dataset_adapters import (
    make_hh_dataset_adapter,
    parse_csv_arg,
)
from utils.hidden_records import extract_token_hidden_records
from utils.hh_metrics import (
    compute_hidden_pair_metrics,
    compute_pair_metrics_for_layer,
    pair_metrics_to_rows,
    summarize_pair_metrics,
)
from utils.hh_sampling import sample_pair_indices_from_metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline HH hidden-state alignment analysis."
    )
    parser.add_argument("--config", type=str, default="interaction/configs/train_basic.yaml")
    parser.add_argument("--output_dir", type=str, default="./out/hh_analysis_debug")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train[:2]")
    parser.add_argument(
        "--hh_dataset_adapter",
        type=str,
        default="gsm8k",
        choices=[
            "gsm8k",
            "mmlu",
            "dolly",
            "gsm8k_mmlu",
            "gsm8k_dolly",
            "dolly_mmlu",
            "dolly_gsm8k",
        ],
        help="Dataset adapter used to build supervised token labels for HH analysis.",
    )
    parser.add_argument(
        "--mmlu_subjects",
        type=str,
        default=None,
        help="Comma-separated MMLU subject list. Defaults to the built-in small subject set.",
    )
    parser.add_argument(
        "--mixed_samples_per_dataset",
        type=int,
        default=1,
        help="Number of random examples to select from each dataset for mixed adapters.",
    )
    parser.add_argument(
        "--mixed_left_samples",
        type=int,
        default=None,
        help="Override number of random examples selected from the left/update dataset in mixed adapters.",
    )
    parser.add_argument(
        "--mixed_right_samples",
        type=int,
        default=None,
        help="Override number of random examples selected from the right/probe dataset in mixed adapters.",
    )
    parser.add_argument(
        "--mixed_gsm8k_split",
        type=str,
        default="train",
        help="GSM8K split used by mixed HH dataset adapters.",
    )
    parser.add_argument(
        "--mixed_mmlu_split",
        type=str,
        default="test",
        help="MMLU split used by mixed HH dataset adapters.",
    )
    parser.add_argument(
        "--mixed_dolly_split",
        type=str,
        default="train",
        help="Dolly split used by mixed HH dataset adapters.",
    )
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--layers", type=str, default="-1")
    parser.add_argument("--max_pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--pair_scope",
        type=str,
        default="all",
        choices=["all", "same", "cross", "cross_dataset"],
        help="Whether to sample all, same-sample, or cross-sample record pairs.",
    )
    parser.add_argument("--include_self", action="store_true")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--debug_token_alignment", action="store_true")
    parser.add_argument("--save_pair_samples", action="store_true")
    parser.add_argument("--save_debug_tensors", action="store_true")
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow Hugging Face model/dataset downloads. Defaults to local cache only.",
    )
    return parser.parse_args()


def pair_scope_to_filters(pair_scope):
    if pair_scope == "all":
        return None, None
    if pair_scope == "same":
        return True, None
    if pair_scope == "cross":
        return False, None
    if pair_scope == "cross_dataset":
        return None, True
    raise ValueError(f"Unknown pair_scope: {pair_scope}")


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metadata_value(metadata, index, key):
    value = metadata[index].get(key)
    if isinstance(value, torch.Tensor):
        return value.item() if value.dim() == 0 else value.tolist()
    return value


def attach_pair_metadata(metric_rows, metadata):
    rows = []
    metadata_keys = [
        "sample_index",
        "example_id",
        "raw_index",
        "sequence_index",
        "hidden_pos",
        "label_pos",
        "input_token_id",
        "label_token_id",
        "input_token_str",
        "label_token_str",
        "hidden_attention_mask",
        "label_attention_mask",
        "dataset",
        "domain",
    ]
    for row in metric_rows:
        out = dict(row)
        left_index = row["left_index"]
        right_index = row["right_index"]
        for key in metadata_keys:
            out[f"left_{key}"] = metadata_value(metadata, left_index, key)
            out[f"right_{key}"] = metadata_value(metadata, right_index, key)
        rows.append(out)
    return rows


def add_metric_variant(rows, variant):
    out = []
    for row in rows:
        updated = dict(row)
        updated["metric_variant"] = variant
        out.append(updated)
    return out


def compute_centered_pair_metrics(hidden_by_layer, pair_indices):
    centered = {}
    for layer_idx, hidden in hidden_by_layer.items():
        hidden_cpu = hidden.detach().cpu().float()
        hidden_centered = hidden_cpu - hidden_cpu.mean(dim=0, keepdim=True)
        centered[layer_idx] = compute_pair_metrics_for_layer(hidden_centered, pair_indices)
    return centered


def make_token_alignment_rows(metadata, max_rows=20):
    rows = []
    for row in metadata[:max_rows]:
        rows.append(
            {
                "example_id": row.get("example_id", row.get("sample_index")),
                "hidden_pos": row["hidden_pos"],
                "target_pos": row["label_pos"],
                "input_token_at_hidden_pos": row["input_token_str"],
                "target_token": row["label_token_str"],
                "label_value": row["label_token_id"],
                "hidden_attention_mask": row.get("hidden_attention_mask"),
                "target_attention_mask": row.get("label_attention_mask"),
                "position_policy": "causal_lm_next_token: hidden_pos = target_pos - 1",
            }
        )
    return rows


def make_pair_sample_rows(metadata, pair_indices, pair_scope, pair_metrics, max_rows=20):
    pairs = pair_indices[:max_rows].detach().cpu()
    final_layer = max(pair_metrics.keys()) if pair_metrics else None
    rows = []

    for pair_offset, (left, right) in enumerate(pairs.tolist()):
        left_row = metadata[left]
        right_row = metadata[right]
        out = {
            "pair_type": pair_scope,
            "i": left,
            "j": right,
            "token_i": left_row.get("label_token_str"),
            "token_j": right_row.get("label_token_str"),
            "token_id_i": left_row.get("label_token_id"),
            "token_id_j": right_row.get("label_token_id"),
            "example_id_i": left_row.get("example_id", left_row.get("sample_index")),
            "example_id_j": right_row.get("example_id", right_row.get("sample_index")),
            "pos_i": left_row.get("label_pos"),
            "pos_j": right_row.get("label_pos"),
            "domain_i": left_row.get("domain"),
            "domain_j": right_row.get("domain"),
            "dataset_i": left_row.get("dataset"),
            "dataset_j": right_row.get("dataset"),
        }
        if final_layer is not None:
            out["raw_cosine_layer_minus_1"] = float(pair_metrics[final_layer]["cosine"][pair_offset].item())
            out["raw_cosine_layer_index"] = final_layer
        rows.append(out)

    return rows


def save_debug_tensors(output_dir, hidden_by_layer, pair_indices):
    layer_idx = max(hidden_by_layer.keys())
    torch.save(hidden_by_layer[layer_idx], output_dir / f"debug_hidden_layer_{layer_idx}.pt")
    torch.save(pair_indices.detach().cpu(), output_dir / "debug_pair_indices.pt")


def load_dataset_and_tokenizer(
    config,
    model_name,
    max_length,
    local_files_only,
    adapter_name,
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
        dataset = adapter.load_dataset(
            split=config["runtime_dataset_split"],
            local_files_only=local_files_only,
        )
        return dataset, adapter
    except OSError as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        print(
            "\nFailed to load the dataset or tokenizer from Hugging Face "
            f"({mode}).\n"
            "By default this runner avoids network downloads. If this model "
            "and dataset are not already cached, rerun with `--allow_download` "
            "on a machine where downloads are approved, or pass `--model_name` "
            "for a cached/local model path.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _prefix_mixed_sample_metadata(samples, dataset_name):
    out = []
    for sample_index, sample in enumerate(samples):
        item = dict(sample)
        original_id = item.get("example_id", sample_index)
        item["example_id"] = f"{dataset_name}:{original_id}"
        item["dataset"] = dataset_name
        out.append(item)
    return out


def _select_random_rows(rows, num_examples, seed):
    if num_examples < 0:
        raise ValueError("mixed_samples_per_dataset must be non-negative.")
    if num_examples >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), k=num_examples))
    return [rows[idx] for idx in indices]


def _process_selected_raw_rows(adapter, raw_rows):
    processed = []
    for raw_row in raw_rows:
        sample = adapter.process_raw_example(raw_row)
        if (sample["labels"] != -100).sum().item() == 0:
            raise ValueError(
                "Selected raw example produced zero supervised labels: "
                f"{raw_row['dataset']}:{raw_row['example_id']}"
            )
        processed.append(sample)
    return processed


def _selected_raw_summary(raw_rows):
    return [
        {
            "dataset": row.get("dataset"),
            "example_id": row.get("example_id"),
            "domain": row.get("domain"),
            "raw_index": row.get("raw_index"),
        }
        for row in raw_rows
    ]


def load_mixed_dataset_and_tokenizer(
    config,
    model_name,
    max_length,
    local_files_only,
    mmlu_subjects,
    left_adapter_name,
    right_adapter_name,
    left_split,
    right_split,
    samples_per_dataset,
    seed,
    left_samples=None,
    right_samples=None,
):
    try:
        left_adapter = make_hh_dataset_adapter(
            adapter_name=left_adapter_name,
            model_name_or_path=model_name,
            max_length=max_length,
            config=config,
            local_files_only=local_files_only,
            mmlu_subjects=mmlu_subjects,
        )
        right_adapter = make_hh_dataset_adapter(
            adapter_name=right_adapter_name,
            model_name_or_path=model_name,
            max_length=max_length,
            config=config,
            local_files_only=local_files_only,
            mmlu_subjects=mmlu_subjects,
        )

        left_dataset = left_adapter.load_dataset(
            split=left_split,
            local_files_only=local_files_only,
        )
        right_dataset = right_adapter.load_dataset(
            split=right_split,
            local_files_only=local_files_only,
        )

        left_count = samples_per_dataset if left_samples is None else left_samples
        right_count = samples_per_dataset if right_samples is None else right_samples
        left_raw_rows = _select_random_rows(
            list(left_adapter.iter_raw_examples(left_dataset)),
            left_count,
            seed,
        )
        right_raw_rows = _select_random_rows(
            list(right_adapter.iter_raw_examples(right_dataset)),
            right_count,
            seed + 1,
        )

        left_samples = _prefix_mixed_sample_metadata(
            _process_selected_raw_rows(left_adapter, left_raw_rows),
            left_adapter_name,
        )
        right_samples = _prefix_mixed_sample_metadata(
            _process_selected_raw_rows(right_adapter, right_raw_rows),
            right_adapter_name,
        )
        selected_raw_examples = (
            _selected_raw_summary(left_raw_rows)
            + _selected_raw_summary(right_raw_rows)
        )
        return left_samples + right_samples, left_adapter, selected_raw_examples
    except OSError as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        print(
            "\nFailed to load a mixed dataset or tokenizer from Hugging Face "
            f"({mode}).\n"
            "By default this runner avoids network downloads. If this model "
            "and dataset are not already cached, rerun with `--allow_download` "
            "on a machine where downloads are approved, or pass `--model_name` "
            "for a cached/local model path.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def load_model(model_name, local_files_only, device):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ).to(device)
        model.eval()
        return model
    except OSError as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        print(
            "\nFailed to load the model from Hugging Face "
            f"({mode}).\n"
            "By default this runner avoids network downloads. If this model "
            "is not already cached, rerun with `--allow_download` on a machine "
            "where downloads are approved, or pass `--model_name` for a cached/local model path.\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def collect_hidden_records(model, tokenizer, processed_dataset, max_samples, layers, device):
    all_metadata = []
    hidden_chunks_by_layer = {}
    n_samples = min(max_samples, len(processed_dataset))

    for sample_index in range(n_samples):
        sample = processed_dataset[sample_index]
        batch = {
            "input_ids": sample["input_ids"].unsqueeze(0).to(device),
            "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
            "labels": sample["labels"].unsqueeze(0).to(device),
            "sample_index": torch.tensor([sample_index], device=device),
        }
        if "example_id" in sample:
            batch["example_id"] = [sample["example_id"]]
        if "raw_index" in sample:
            batch["raw_index"] = [sample["raw_index"]]
        if "dataset" in sample:
            batch["dataset"] = [sample["dataset"]]
        if "domain" in sample:
            batch["domain"] = [sample["domain"]]
        metadata, hidden_by_layer = extract_token_hidden_records(
            model=model,
            batch=batch,
            tokenizer=tokenizer,
            layers=layers,
        )
        all_metadata.extend(metadata)
        for layer_idx, hidden in hidden_by_layer.items():
            hidden_chunks_by_layer.setdefault(layer_idx, []).append(hidden)

    hidden_by_layer = {
        layer_idx: torch.cat(chunks, dim=0) if chunks else torch.empty(0, 0)
        for layer_idx, chunks in hidden_chunks_by_layer.items()
    }
    return all_metadata, hidden_by_layer


def main():
    args = parse_args()
    config = load_config(args.config)
    config["runtime_dataset_split"] = args.dataset_split
    adapter_name = args.hh_dataset_adapter or config.get("hh_dataset_adapter", "gsm8k")
    mmlu_subjects = parse_csv_arg(args.mmlu_subjects)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name or config["model"]["name"]
    max_length = config["model"]["max_length"]
    local_files_only = not args.allow_download
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    selected_raw_examples = None

    mixed_adapter_specs = {
        "gsm8k_mmlu": ("gsm8k", "mmlu", args.mixed_gsm8k_split, args.mixed_mmlu_split),
        "gsm8k_dolly": ("gsm8k", "dolly", args.mixed_gsm8k_split, args.mixed_dolly_split),
        "dolly_mmlu": ("dolly", "mmlu", args.mixed_dolly_split, args.mixed_mmlu_split),
        "dolly_gsm8k": ("dolly", "gsm8k", args.mixed_dolly_split, args.mixed_gsm8k_split),
    }
    if adapter_name in mixed_adapter_specs:
        left_adapter_name, right_adapter_name, left_split, right_split = mixed_adapter_specs[adapter_name]
        processed_dataset, processor, selected_raw_examples = load_mixed_dataset_and_tokenizer(
            config=config,
            model_name=model_name,
            max_length=max_length,
            local_files_only=local_files_only,
            mmlu_subjects=mmlu_subjects,
            left_adapter_name=left_adapter_name,
            right_adapter_name=right_adapter_name,
            left_split=left_split,
            right_split=right_split,
            samples_per_dataset=args.mixed_samples_per_dataset,
            seed=args.seed,
            left_samples=args.mixed_left_samples,
            right_samples=args.mixed_right_samples,
        )
        max_samples_for_hidden = len(processed_dataset)
    else:
        dataset, processor = load_dataset_and_tokenizer(
            config=config,
            model_name=model_name,
            max_length=max_length,
            local_files_only=local_files_only,
            adapter_name=adapter_name,
            mmlu_subjects=mmlu_subjects,
        )
        processed_dataset = processor.process_dataset(dataset)
        max_samples_for_hidden = args.max_samples

    if not processed_dataset:
        raise ValueError("No processed samples were produced.")

    model = load_model(
        model_name=model_name,
        local_files_only=local_files_only,
        device=device,
    )

    metadata, hidden_by_layer = collect_hidden_records(
        model=model,
        tokenizer=processor.tokenizer,
        processed_dataset=processed_dataset,
        max_samples=max_samples_for_hidden,
        layers=args.layers,
        device=device,
    )
    if not metadata:
        raise ValueError("No hidden records were collected.")

    pair_indices = sample_pair_indices_from_metadata(
        metadata=metadata,
        max_pairs=args.max_pairs,
        seed=args.seed,
        include_self=args.include_self,
        directed=args.directed,
        same_sample=pair_scope_to_filters(args.pair_scope)[0],
        different_dataset=pair_scope_to_filters(args.pair_scope)[1],
    )
    pair_metrics = compute_hidden_pair_metrics(hidden_by_layer, pair_indices)
    centered_pair_metrics = compute_centered_pair_metrics(hidden_by_layer, pair_indices)
    summaries = {
        "raw": summarize_pair_metrics(pair_metrics),
        "centered": summarize_pair_metrics(centered_pair_metrics),
    }
    metric_rows = attach_pair_metadata(
        add_metric_variant(pair_metrics_to_rows(pair_metrics), "raw")
        + add_metric_variant(pair_metrics_to_rows(centered_pair_metrics), "centered"),
        metadata=metadata,
    )

    run_config = {
        "model_name": model_name,
        "hh_dataset_adapter": adapter_name,
        "mmlu_subjects": mmlu_subjects,
        "dataset": config["dataset"],
        "dataset_split": args.dataset_split,
        "mixed_gsm8k_split": args.mixed_gsm8k_split,
        "mixed_mmlu_split": args.mixed_mmlu_split,
        "mixed_dolly_split": args.mixed_dolly_split,
        "mixed_samples_per_dataset": args.mixed_samples_per_dataset,
        "mixed_left_samples": args.mixed_left_samples,
        "mixed_right_samples": args.mixed_right_samples,
        "max_samples": args.max_samples,
        "effective_max_samples": max_samples_for_hidden,
        "layers": args.layers,
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "pair_scope": args.pair_scope,
        "include_self": args.include_self,
        "directed": args.directed,
        "local_files_only": local_files_only,
        "num_records": len(metadata),
        "num_pairs": int(pair_indices.shape[0]),
        "selected_layers": sorted(hidden_by_layer.keys()),
    }
    if selected_raw_examples is not None:
        run_config["selected_raw_examples"] = selected_raw_examples

    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "summary.json", summaries)
    write_jsonl(output_dir / "metadata.jsonl", metadata)
    write_csv(output_dir / "pair_metrics.csv", metric_rows)

    if args.debug_token_alignment:
        token_alignment_rows = make_token_alignment_rows(metadata, max_rows=20)
        write_jsonl(output_dir / "debug_token_alignment.jsonl", token_alignment_rows)
        print("Debug token alignment rows:")
        for row in token_alignment_rows:
            print(row)

    if args.save_pair_samples:
        write_jsonl(
            output_dir / "debug_pair_samples.jsonl",
            make_pair_sample_rows(
                metadata=metadata,
                pair_indices=pair_indices,
                pair_scope=args.pair_scope,
                pair_metrics=pair_metrics,
                max_rows=20,
            ),
        )

    if args.save_debug_tensors:
        save_debug_tensors(output_dir, hidden_by_layer, pair_indices)

    print(f"Saved HH analysis outputs to {output_dir}")
    print(f"num_records: {len(metadata)}")
    print(f"num_pairs: {int(pair_indices.shape[0])}")
    print(f"selected_layers: {sorted(hidden_by_layer.keys())}")


if __name__ == "__main__":
    main()
