#!/usr/bin/env python
"""Validate the attribution release, including data identification artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import validate_release_core as _core
from reproduce import ATTRIBUTION_DIR, IDENTIFICATION_DIR
from validate_release_core import *  # noqa: F401,F403 - retain the original validator API


def validate_identification() -> int:
    reproduction_path = (
        IDENTIFICATION_DIR / "reference" / "qwen25_1p5b_reproduction.tsv"
    )
    rows = read_tsv(reproduction_path)
    require(len(rows) == 9, f"expected 9 identification rows: {reproduction_path}")

    metric_fields = {
        "auc_mean": "proposed_auc",
        "auc_std": "proposed_auc_std",
        "recall_mean": "proposed_recall",
        "recall_std": "proposed_recall_std",
    }
    seen = set()
    for row in rows:
        run_key = (row["dataset"], row["preset"])
        require(run_key not in seen, f"duplicate identification row: {run_key}")
        seen.add(run_key)

        result_path = ATTRIBUTION_DIR / row["source_artifact"]
        time_path = ATTRIBUTION_DIR / row["time_artifact"]
        require(result_path.is_file(), f"identification result missing: {result_path}")
        require(time_path.is_file(), f"identification timing missing: {time_path}")
        result = read_json(result_path)
        timing = read_json(time_path)

        for reference_field, result_field in metric_fields.items():
            require(
                close(result[result_field], float(row[reference_field]), 1e-12),
                f"identification metric mismatch ({result_field}): {result_path}",
            )
        require(
            close(timing[row["model"]], float(row["total_sec"]), 1e-12),
            f"identification timing mismatch: {time_path}",
        )
        require(
            result["readout_channels"] == row["readout_channels"].split(","),
            f"identification channels mismatch: {result_path}",
        )
        expected_input_norm = row["gh_input_layernorm"].lower() == "true"
        require(
            bool(result["gh_use_input_layernorm"]) == expected_input_norm,
            f"identification input RMSNorm mismatch: {result_path}",
        )
        require(
            result["vocab_mode"] == row["vocab_mode"],
            f"identification vocabulary mismatch: {result_path}",
        )
        if "gh" in result["readout_channels"]:
            require(
                result["gh_embedding_layers"] == list(range(1, 28)),
                f"identification GH layers mismatch: {result_path}",
            )
            require(
                result["gh_layer_index_mode"] == "bottom",
                f"identification GH indexing mismatch: {result_path}",
            )

        for split, expected_n in (("train", 900), ("test", 100)):
            dataset_dir = (
                IDENTIFICATION_DIR / "data" / f"{row['dataset']}_{split}.hf"
            )
            state_path = dataset_dir / "state.json"
            info_path = dataset_dir / "dataset_info.json"
            require(state_path.is_file(), f"dataset state missing: {state_path}")
            require(info_path.is_file(), f"dataset info missing: {info_path}")
            state = read_json(state_path)
            files = state.get("_data_files", [])
            require(files, f"dataset has no Arrow shards: {dataset_dir}")
            for item in files:
                require(
                    (dataset_dir / item["filename"]).is_file(),
                    f"dataset shard missing: {dataset_dir / item['filename']}",
                )
            require(int(row[f"n_{split}"]) == expected_n, f"bad declared {split} size")

    paper_path = IDENTIFICATION_DIR / "reference" / "paper_table_excerpt.tsv"
    paper_rows = read_tsv(paper_path)
    require(len(paper_rows) == 6, f"expected 6 paper rows: {paper_path}")
    require(
        {row["dataset"] for row in paper_rows}
        == {"grammars", "math_without_reason", "math_with_reason"},
        f"paper identification datasets mismatch: {paper_path}",
    )
    require(
        {row["method"] for row in paper_rows} == {"CH1", "CH1+CH2"},
        f"paper identification methods mismatch: {paper_path}",
    )
    return len(rows)


def main() -> None:
    _core.main()
    identification_rows = validate_identification()
    print(f"identification OK: {identification_rows} measured rows and 6 paper rows")


if __name__ == "__main__":
    main()
