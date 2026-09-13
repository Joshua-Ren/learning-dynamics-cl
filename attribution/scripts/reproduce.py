#!/usr/bin/env python
"""Portable launchers for the identification, memory, and retrieval experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import reproduce_core as _core
from reproduce_core import *  # noqa: F401,F403 - retain the original public launcher API


IDENTIFICATION_DIR = ATTRIBUTION_DIR / "identification"
IDENTIFICATION_SRC_DIR = IDENTIFICATION_DIR / "src"
IDENTIFICATION_DATASETS = (
    "grammars",
    "math_without_reason",
    "math_with_reason",
)
IDENTIFICATION_PRESETS = ("ch1", "ch1-ch2", "ch1-ch2-inputln")


def reproduce_identification(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(IDENTIFICATION_SRC_DIR / "run_identification.py"),
        *args.datasets,
        "--preset",
        args.preset,
        "--model-name",
        args.model_name,
        "--embed-device",
        args.embed_device,
        "--score-device",
        args.score_device,
    ]
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])
    if args.allow_download:
        command.append("--allow-download")
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry-run")

    # Execute even for a dry run so the task launcher expands every dataset into
    # the exact scorer command without loading a model.
    run_command(command, dry_run=False)


def identification_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} identification",
        description="Run the balanced data-identification benchmark.",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=(*IDENTIFICATION_DATASETS, "all"),
        default=["all"],
    )
    parser.add_argument("--preset", choices=IDENTIFICATION_PRESETS, default="ch1-ch2-inputln")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--embed-device", default="cuda:0")
    parser.add_argument("--score-device", default="cuda:0")
    add_execution_args(parser)
    return parser


def list_runs() -> None:
    _core.list_runs()
    print("\nIdentification benchmark:")
    print("  model          Qwen/Qwen2.5-1.5B")
    print(f"  datasets       {', '.join(IDENTIFICATION_DATASETS)}")
    print(f"  presets        {', '.join(IDENTIFICATION_PRESETS)}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "identification":
        reproduce_identification(identification_parser().parse_args(sys.argv[2:]))
    elif sys.argv[1:] == ["list"]:
        list_runs()
    else:
        _core.main()


if __name__ == "__main__":
    main()
