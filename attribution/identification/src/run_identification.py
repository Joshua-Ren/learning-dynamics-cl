#!/usr/bin/env python
"""Run the class-balanced data-identification benchmark."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


IDENTIFICATION_DIR = Path(__file__).resolve().parents[1]
ATTRIBUTION_DIR = IDENTIFICATION_DIR.parent
REPO_DIR = ATTRIBUTION_DIR.parent
DATA_DIR = IDENTIFICATION_DIR / "data"
SCORER = ATTRIBUTION_DIR / "common" / "forvalue_streaming_ghrh.py"
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "attribution" / "identification"

DATASETS = ("grammars", "math_without_reason", "math_with_reason")
PRESETS = {
    "ch1": {"readout_channel": "rh", "input_layernorm": False},
    "ch1-ch2": {"readout_channel": "both", "input_layernorm": False},
    "ch1-ch2-inputln": {"readout_channel": "both", "input_layernorm": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=(*DATASETS, "all"),
        default=["all"],
        help="Datasets to run; defaults to all three in sequence.",
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="ch1-ch2-inputln")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--embed-device", default="cuda:0")
    parser.add_argument("--score-device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_datasets(requested: list[str]) -> list[str]:
    if not requested or requested == ["all"]:
        return list(DATASETS)
    if "all" in requested:
        raise ValueError("Use `all` by itself, or list individual datasets.")
    return list(dict.fromkeys(requested))


def build_command(
    args: argparse.Namespace,
    dataset: str,
    result_path: Path,
    time_path: Path,
) -> list[str]:
    preset = PRESETS[args.preset]
    command = [
        sys.executable,
        str(SCORER),
        "--model_name",
        args.model_name,
        "--dataset_name",
        dataset,
        "--dataset_dir",
        str(DATA_DIR),
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
        "--vocab_mode",
        "total_unique",
        "--lowest_likelihood_ratio",
        "1.0",
        "--train_score_chunk",
        "16",
        "--n_class",
        "10",
        "--n_sample_per_class",
        "90",
        "--embed_device",
        args.embed_device,
        "--score_device",
        args.score_device,
        "--readout_channel",
        preset["readout_channel"],
        "--gh_embedding_layers",
        "all",
        "--gh_layer_index_mode",
        "bottom",
        "--result_path",
        str(result_path),
        "--time_path",
        str(time_path),
    ]
    if preset["input_layernorm"]:
        command.append("--gh_use_input_layernorm")
    if not args.allow_download:
        command.append("--local_files_only")
    return command


def main() -> None:
    args = parse_args()
    datasets = resolve_datasets(args.datasets)
    args.output_dir = args.output_dir.resolve()

    for dataset in datasets:
        stem = f"{dataset}_{args.preset.replace('-', '_')}"
        result_path = args.output_dir / f"{stem}.json"
        time_path = args.output_dir / f"{stem}_time.json"
        if not args.dry_run:
            if not args.overwrite and (result_path.exists() or time_path.exists()):
                raise FileExistsError(
                    f"refusing to overwrite {result_path} or {time_path}; pass --overwrite"
                )
            args.output_dir.mkdir(parents=True, exist_ok=True)

        command = build_command(args, dataset, result_path, time_path)
        print("+", shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=REPO_DIR)


if __name__ == "__main__":
    main()
