#!/usr/bin/env python
"""Portable launchers for the released agent-memory and retrieval experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ATTRIBUTION_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ATTRIBUTION_DIR.parent
AGENT_MEMORY_DIR = ATTRIBUTION_DIR / "agent_memory"
RETRIEVAL_DIR = ATTRIBUTION_DIR / "retrieval"
AGENT_MEMORY_SRC_DIR = AGENT_MEMORY_DIR / "src"
RETRIEVAL_SRC_DIR = RETRIEVAL_DIR / "src"
OUTPUT_DIR = REPO_DIR / "outputs" / "attribution"

MEMORY_RUNS = {
    "qwen25_1p5b": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "data": "gsm8k_train_first200_qwen25_1p5b_instruct_wrong_codex_manual_translated.json",
        "scores": {
            "rh": "forvalue_gsm8k_train_first200_qwen25_1p5b_instruct_wrong_question_query_rh.json",
            "gh": "forvalue_gsm8k_train_first200_qwen25_1p5b_instruct_wrong_question_query_gh_all.json",
            "both": "forvalue_gsm8k_train_first200_qwen25_1p5b_instruct_wrong_question_query_both_all.json",
            "native_last_embedding_mean": "forvalue_gsm8k_train_first200_qwen25_1p5b_instruct_wrong_question_query_native_last_embedding_mean.json",
        },
    },
    "llama32_3b": {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "data": "gsm8k_train_first200_llama32_3b_instruct_wrong_codex_manual_translated.json",
        "scores": {
            "rh": "forvalue_gsm8k_train_first200_wrong_question_query_rh.json",
            "gh": "forvalue_gsm8k_train_first200_wrong_question_query_gh_all.json",
            "both": "forvalue_gsm8k_train_first200_wrong_question_query_both_all.json",
            "native_last_embedding_mean": "forvalue_gsm8k_train_first200_llama32_3b_instruct_wrong_question_query_native_last_embedding_mean.json",
        },
    },
    "qwen3_1p7b": {
        "model": "Qwen/Qwen3-1.7B",
        "data": "gsm8k_train_first200_qwen3_1p7b_wrong_translated.json",
        "scores": {
            "rh": "forvalue_gsm8k_train_first200_qwen3_1p7b_wrong_question_query_rh.json",
            "gh": "forvalue_gsm8k_train_first200_qwen3_1p7b_wrong_question_query_gh_all.json",
            "both": "forvalue_gsm8k_train_first200_qwen3_1p7b_wrong_question_query_both_all.json",
            "native_last_embedding_mean": "forvalue_gsm8k_train_first200_qwen3_1p7b_wrong_question_query_native_last_embedding_mean.json",
        },
    },
    "qwen3_4b": {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "data": "gsm8k_train_first400_qwen3_4b_instruct_2507_wrong_translated.json",
        "scores": {
            "rh": "forvalue_gsm8k_train_first400_qwen3_4b_instruct_2507_wrong_question_query_rh.json",
            "gh_top2_near_lmhead": "forvalue_gsm8k_train_first400_qwen3_4b_instruct_2507_wrong_question_query_gh_top2_near_lmhead.json",
            "both_top2_near_lmhead": "forvalue_gsm8k_train_first400_qwen3_4b_instruct_2507_wrong_question_query_both_top2_near_lmhead.json",
            "native_last_embedding_mean": "forvalue_gsm8k_train_first400_qwen3_4b_instruct_2507_wrong_question_query_native_last_embedding_mean.json",
        },
    },
}

RETRIEVAL_SUITES = {
    "qwen3_gsm8k": [
        "forvalue_gsm8k_train50_top4_qwen3_4b_rh.json",
        "forvalue_gsm8k_train50_top4_qwen3_4b_gh_all.json",
        "forvalue_gsm8k_train50_top4_qwen3_4b_both_all.json",
    ],
    "qwen3_medical": [
        "forvalue_translation_top4_all_english_rh.json",
        "forvalue_translation_top4_all_english_gh_all.json",
        "forvalue_translation_top4_all_english_both_all.json",
    ],
    "llama32_medical": [
        "forvalue_translation_top4_all_english_llama32_3b_rh.json",
        "forvalue_translation_top4_all_english_llama32_3b_gh_all.json",
        "forvalue_translation_top4_all_english_llama32_3b_both_all.json",
    ],
}

MEMORY_DATA_BY_MODEL = {
    spec["model"]: AGENT_MEMORY_DIR / "data" / spec["data"]
    for spec in MEMORY_RUNS.values()
}


def score_files() -> list[Path]:
    return sorted([
        *(AGENT_MEMORY_DIR / "scores").glob("*.json"),
        *(RETRIEVAL_DIR / "scores").glob("*.json"),
    ])


def resolve_score(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate.resolve()
    matches = [path for path in score_files() if path.name == name or path.stem == name]
    if len(matches) != 1:
        choices = ", ".join(path.name for path in matches) or "none"
        raise ValueError(f"score reference {name!r} matched {len(matches)} files: {choices}")
    return matches[0]


def score_dataset(reference: Path, summary: dict) -> Path:
    name = reference.name
    if "translation_top4_all_english" in name:
        return RETRIEVAL_DIR / "data" / "medical_translated_test_top10.json"
    if "gsm8k_train50_top4" in name:
        return RETRIEVAL_DIR / "data" / "gsm8k_train_first50_translated_manual.json"
    model = summary.get("model_name") or summary.get("model")
    try:
        return MEMORY_DATA_BY_MODEL[model]
    except KeyError as exc:
        raise ValueError(f"cannot map score model to a released dataset: {model!r}") from exc


def add_local_only(command: list[str], allow_download: bool) -> None:
    if not allow_download:
        command.append("--local_files_only")


def run_command(command: list[str], dry_run: bool) -> None:
    print("+", shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=REPO_DIR)


def build_score_command(
    reference: Path,
    output: Path,
    allow_download: bool,
    embed_device: str,
    score_device: str,
) -> list[str]:
    payload = json.loads(reference.read_text(encoding="utf-8"))
    summary = payload["summary"]
    model = summary.get("model_name") or summary.get("model")
    dataset = score_dataset(reference, summary)
    scoring_method = summary.get("scoring_method") or "forvalue"
    readout_channels = list(summary.get("readout_channels") or [])

    command = [
        sys.executable,
        str(RETRIEVAL_SRC_DIR / "run_forvalue_translation_top4.py"),
        "--data_path",
        str(dataset),
        "--output_path",
        str(output),
        "--model_name",
        str(model),
        "--retrieval_query_field",
        str(summary.get("retrieval_query_field") or "qa"),
        "--max_length",
        str(summary.get("max_length") or 192),
        "--batch_size",
        str(summary.get("batch_size") or 4),
        "--prediction_topk",
        str(summary.get("prediction_topk") or 16),
        "--train_score_chunk",
        "8",
        "--embed_device",
        embed_device,
        "--score_device",
        score_device,
    ]
    add_local_only(command, allow_download)

    if scoring_method == "native_last_embedding":
        command.extend(
            [
                "--scoring_method",
                "native_last_embedding",
                "--native_pooling",
                str(summary.get("native_pooling") or "mean"),
            ]
        )
        return command

    command.extend(["--scoring_method", "forvalue"])
    if not readout_channels:
        raise ValueError(f"ForValue reference has no readout_channels: {reference}")
    command.extend(["--readout_channels", *readout_channels])
    if "gh" in readout_channels:
        layers = summary.get("gh_embedding_layers")
        if not layers:
            raise ValueError(f"GH reference has no resolved layers: {reference}")
        command.extend(["--gh_embedding_layers", *[str(layer) for layer in layers]])
        command.extend(
            ["--gh_layer_index_mode", str(summary.get("gh_layer_index_mode") or "bottom")]
        )
        if summary.get("gh_use_input_layernorm"):
            command.append("--gh_use_input_layernorm")
    return command


def reproduce_score(args: argparse.Namespace, reference_name: str, output: Path | None = None) -> None:
    reference = resolve_score(reference_name)
    experiment = "agent_memory" if AGENT_MEMORY_DIR in reference.parents else "retrieval"
    destination = output or OUTPUT_DIR / experiment / "scores" / reference.name
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}; pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        build_score_command(
            reference,
            destination,
            args.allow_download,
            args.embed_device,
            args.score_device,
        ),
        args.dry_run,
    )


def reproduce_memory(args: argparse.Namespace, random_baseline: bool = False) -> None:
    spec = MEMORY_RUNS[args.run_id]
    data = AGENT_MEMORY_DIR / "data" / spec["data"]
    suffix = "random" if random_baseline else "top1"
    output = args.output or OUTPUT_DIR / "agent_memory" / f"{args.run_id}_{suffix}.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    if random_baseline:
        command = [
            sys.executable,
            str(AGENT_MEMORY_SRC_DIR / "run_top1_agent_random_selection.py"),
            "--data_path",
            str(data),
            "--model_name",
            spec["model"],
            "--output_path",
            str(output),
            "--random_seed",
            "42",
            "--retrieval_query_field",
            "question",
            "--max_prompt_length",
            "2048",
            "--max_new_tokens",
            "512",
            "--generation_batch_size",
            str(args.generation_batch_size),
        ]
    else:
        command = [
            sys.executable,
            str(AGENT_MEMORY_SRC_DIR / "run_top1_agent_from_saved_selection.py"),
            "--data_path",
            str(data),
            "--model_name",
            spec["model"],
            "--output_path",
            str(output),
            "--retrieval_query_field",
            "question",
            "--max_prompt_length",
            "2048",
            "--max_new_tokens",
            "512",
            "--generation_batch_size",
            str(args.generation_batch_size),
        ]
        for channel, filename in spec["scores"].items():
            command.extend(
                [
                    "--selection",
                    f"{channel}={AGENT_MEMORY_DIR / 'scores' / filename}",
                ]
            )
    add_local_only(command, args.allow_download)
    run_command(command, args.dry_run)


def list_runs() -> None:
    print("Memory runs:")
    for run_id, spec in MEMORY_RUNS.items():
        print(f"  {run_id:14s} {spec['model']}")
    print("\nRetrieval suites:")
    for suite_id, references in RETRIEVAL_SUITES.items():
        print(f"  {suite_id:14s} {len(references)} score runs")
    print("\nSaved score references:")
    for path in score_files():
        print(f"  {path.relative_to(ATTRIBUTION_DIR)}")


def add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List released runs and saved scores.")

    score_parser = subparsers.add_parser("score", help="Recompute one saved retrieval score.")
    score_parser.add_argument("reference", help="Saved score filename, stem, or path.")
    score_parser.add_argument("--output", type=Path)
    score_parser.add_argument("--embed-device", default="auto")
    score_parser.add_argument("--score-device", default="auto")
    add_execution_args(score_parser)

    suite_parser = subparsers.add_parser("retrieval-suite", help="Recompute a three-score suite.")
    suite_parser.add_argument("suite_id", choices=tuple(RETRIEVAL_SUITES))
    suite_parser.add_argument("--embed-device", default="auto")
    suite_parser.add_argument("--score-device", default="auto")
    add_execution_args(suite_parser)

    for name, help_text in (
        ("memory", "Run memory generation from saved deterministic scores."),
        ("random", "Run the seeded random-memory control."),
    ):
        run_parser = subparsers.add_parser(name, help=help_text)
        run_parser.add_argument("run_id", choices=tuple(MEMORY_RUNS))
        run_parser.add_argument("--output", type=Path)
        run_parser.add_argument("--generation-batch-size", type=int, default=8)
        add_execution_args(run_parser)

    args = parser.parse_args()
    if args.command == "list":
        list_runs()
    elif args.command == "score":
        reproduce_score(args, args.reference, args.output)
    elif args.command == "retrieval-suite":
        for reference in RETRIEVAL_SUITES[args.suite_id]:
            reproduce_score(args, reference)
    elif args.command == "memory":
        reproduce_memory(args, random_baseline=False)
    elif args.command == "random":
        reproduce_memory(args, random_baseline=True)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
