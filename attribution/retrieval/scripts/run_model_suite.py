#!/usr/bin/env python
"""Run RH and Both/all-GH retrieval on MMMLU and GSM8K for one model."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

RETRIEVAL_DIR = Path(__file__).resolve().parents[1]
ATTRIBUTION_DIR = RETRIEVAL_DIR.parent
REPO_DIR = ATTRIBUTION_DIR.parent
SCORER = RETRIEVAL_DIR / "src" / "run_forvalue_translation_top4.py"
DEFAULT_OUTPUT_ROOT = REPO_DIR / "outputs" / "attribution" / "retrieval" / "model_suites"

BENCHMARKS = (
    {
        "name": "mmmlu_rh",
        "data": RETRIEVAL_DIR / "data" / "medical_translated_test_top10.json",
        "max_length": 192,
        "channels": ("rh",),
    },
    {
        "name": "mmmlu_both_all_gh",
        "data": RETRIEVAL_DIR / "data" / "medical_translated_test_top10.json",
        "max_length": 192,
        "channels": ("rh", "gh"),
    },
    {
        "name": "gsm8k_first50_rh",
        "data": RETRIEVAL_DIR / "data" / "gsm8k_train_first50_translated_manual.json",
        "max_length": 768,
        "channels": ("rh",),
    },
    {
        "name": "gsm8k_first50_both_all_gh",
        "data": RETRIEVAL_DIR / "data" / "gsm8k_train_first50_translated_manual.json",
        "max_length": 768,
        "channels": ("rh", "gh"),
    },
)


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model_name).strip("-").lower()
    if not slug:
        raise ValueError("model name does not contain any filename-safe characters")
    return slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Hugging Face model ID or local path.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prediction-topk", type=int, default=16)
    parser.add_argument("--train-score-chunk", type=int, default=8)
    parser.add_argument("--embed-device", default="auto")
    parser.add_argument("--score-device", default="auto")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads. The default requires a local cache.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, benchmark: dict, output_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(SCORER),
        "--data_path",
        str(benchmark["data"]),
        "--output_path",
        str(output_path),
        "--model_name",
        args.model_name,
        "--max_length",
        str(benchmark["max_length"]),
        "--batch_size",
        str(args.batch_size),
        "--prediction_topk",
        str(args.prediction_topk),
        "--train_score_chunk",
        str(args.train_score_chunk),
        "--embed_device",
        args.embed_device,
        "--score_device",
        args.score_device,
        "--scoring_method",
        "forvalue",
        "--readout_channels",
        *benchmark["channels"],
        "--retrieval_query_field",
        "qa",
    ]
    if "gh" in benchmark["channels"]:
        command.extend(
            [
                "--gh_embedding_layers",
                "all",
                "--gh_layer_index_mode",
                "bottom",
                "--gh_use_input_layernorm",
            ]
        )
    if not args.allow_download:
        command.append("--local_files_only")
    return command


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / model_slug(args.model_name)
    outputs = {
        benchmark["name"]: output_dir / f"{benchmark['name']}.json"
        for benchmark in BENCHMARKS
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing results:\n  {joined}\nPass --overwrite to replace them.")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for benchmark in BENCHMARKS:
        output_path = outputs[benchmark["name"]]
        command = build_command(args, benchmark, output_path)
        print("+", shlex.join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True, cwd=REPO_DIR)
        summary = json.loads(output_path.read_text(encoding="utf-8"))["summary"]
        summaries.append(
            (
                benchmark["name"],
                float(summary["top4_item_accuracy_mean"]),
                float(summary["top4_all_same_source_rate"]),
                int(summary["queries_with_all_4_same_source"]),
                int(summary["num_queries"]),
            )
        )

    if summaries:
        print("\nresult\ttop4_item_accuracy\tall_four_rate\tall_four_count")
        for name, item_accuracy, all_four_rate, count, total in summaries:
            print(f"{name}\t{item_accuracy:.4f}\t{all_four_rate:.4f}\t{count}/{total}")
        print(f"\nresults written to {output_dir}")


if __name__ == "__main__":
    main()
