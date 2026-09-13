#!/usr/bin/env python
"""Validate released datasets, saved scores, references, and reproductions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from reproduce import (
    AGENT_MEMORY_DIR,
    ATTRIBUTION_DIR,
    MEMORY_RUNS,
    RETRIEVAL_DIR,
    score_dataset,
)

LANGUAGES = ("zh", "fr", "ko", "es")
MANIFEST_PATH = ATTRIBUTION_DIR / "MANIFEST.json"
IGNORED_PARTS = {"__pycache__", ".pytest_cache"}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def close(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def released_files() -> list[Path]:
    files = []
    for path in ATTRIBUTION_DIR.rglob("*"):
        if not path.is_file() or path == MANIFEST_PATH:
            continue
        if any(part in IGNORED_PARTS for part in path.parts) or path.suffix == ".pyc":
            continue
        files.append(path)
    return sorted(files)


def write_manifest() -> None:
    payload = {
        "schema_version": 1,
        "files": [
            {
                "path": str(path.relative_to(ATTRIBUTION_DIR)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in released_files()
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {MANIFEST_PATH} ({len(payload['files'])} files)")


def validate_manifest() -> int:
    require(MANIFEST_PATH.is_file(), "MANIFEST.json is missing; run --write-manifest")
    manifest = read_json(MANIFEST_PATH)
    entries = {entry["path"]: entry for entry in manifest["files"]}
    actual = {
        str(path.relative_to(ATTRIBUTION_DIR)): path for path in released_files()
    }
    require(set(entries) == set(actual), "manifest file list does not match the release tree")
    for relative, path in actual.items():
        entry = entries[relative]
        require(entry["bytes"] == path.stat().st_size, f"size mismatch: {relative}")
        require(entry["sha256"] == sha256(path), f"sha256 mismatch: {relative}")
    return len(actual)


def translated_gsm_rows(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    dataset = data.get("source_dataset", "openai/gsm8k")
    return [(f"{dataset}:{row['index']}", row) for row in data["records"]]


def translated_medical_rows(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for dataset, dataset_rows in data["datasets"].items():
        rows.extend((f"{dataset}:{row['index']}", row) for row in dataset_rows)
    return rows


def translated_rows(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if "datasets" in data:
        return translated_medical_rows(data)
    require("records" in data, "translation dataset has neither records nor datasets")
    return translated_gsm_rows(data)


def validate_translation_dataset(path: Path) -> int:
    data = read_json(path)
    rows = translated_rows(data)
    require(rows, f"empty dataset: {path}")
    source_ids = [source_id for source_id, _ in rows]
    require(len(source_ids) == len(set(source_ids)), f"duplicate source ids: {path}")
    for source_id, row in rows:
        translations = row.get("translations")
        require(isinstance(translations, dict), f"missing translations for {source_id}")
        for language in LANGUAGES:
            fields = translations.get(language)
            require(isinstance(fields, dict), f"missing {language} translation for {source_id}")
            require(bool(fields.get("question") or fields.get("input")), f"empty question for {source_id}/{language}")
            require("answer" in fields or "A" in fields, f"empty answer/options for {source_id}/{language}")
    declared = data.get("num_source_records")
    if declared is not None:
        require(int(declared) == len(rows), f"num_source_records mismatch: {path}")
    wrong_indices = data.get("wrong_indices")
    if wrong_indices is not None:
        require(
            [int(row["index"]) for _, row in rows] == [int(index) for index in wrong_indices],
            f"wrong_indices order mismatch: {path}",
        )
    return len(rows)


def validate_source_pool(path: Path) -> int:
    data = read_json(path)
    rows = data.get("records", [])
    require(rows, f"empty source pool: {path}")
    indices = [int(row["index"]) for row in rows]
    require(indices == list(range(len(rows))), f"source pool is not a contiguous prefix: {path}")
    for row in rows:
        source = row.get("source")
        require(isinstance(source, dict), f"missing source record: {path}:{row.get('index')}")
        require(bool(source.get("question")), f"missing question: {path}:{row.get('index')}")
        require("answer" in source, f"missing answer: {path}:{row.get('index')}")
    return len(rows)


def candidate_metadata(data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for source_id, row in translated_rows(data):
        for language in LANGUAGES:
            candidates.append({"source_id": source_id, "language": language})
    return candidates


def validate_score(path: Path) -> tuple[int, float, float]:
    score = read_json(path)
    summary = score.get("summary", {})
    per_query = score.get("per_query", [])
    dataset_path = score_dataset(path, summary)
    require(dataset_path.is_file(), f"mapped dataset missing for {path.name}: {dataset_path}")
    dataset = read_json(dataset_path)
    query_rows = translated_rows(dataset)
    candidates = candidate_metadata(dataset)

    require(len(per_query) == len(query_rows), f"query count mismatch: {path}")
    require(int(summary.get("num_queries", -1)) == len(query_rows), f"summary query count mismatch: {path}")
    require(
        int(summary.get("num_translation_candidates", -1)) == len(candidates),
        f"candidate count mismatch: {path}",
    )

    same_counts = []
    for query_index, (query_result, (source_id, _)) in enumerate(zip(per_query, query_rows)):
        query = query_result.get("query", {})
        require(query.get("source_id") == source_id, f"query order/source mismatch: {path}:{query_index}")
        top4 = query_result.get("top4", [])
        require(len(top4) == 4, f"top4 length mismatch: {path}:{query_index}")
        seen = set()
        computed_same = 0
        for rank, item in enumerate(top4, start=1):
            train_index = int(item["train_index"])
            require(0 <= train_index < len(candidates), f"bad train_index: {path}:{query_index}")
            require(train_index not in seen, f"duplicate top4 train_index: {path}:{query_index}")
            seen.add(train_index)
            expected = candidates[train_index]
            require(item.get("source_id") == expected["source_id"], f"candidate source mismatch: {path}:{query_index}")
            require(item.get("language") == expected["language"], f"candidate language mismatch: {path}:{query_index}")
            require(int(item.get("rank", rank)) == rank, f"rank mismatch: {path}:{query_index}")
            same_source = expected["source_id"] == source_id
            require(bool(item.get("same_source")) == same_source, f"same_source mismatch: {path}:{query_index}")
            computed_same += int(same_source)
        require(int(query_result["top4_same_source_count"]) == computed_same, f"top4 count mismatch: {path}:{query_index}")
        same_counts.append(computed_same)

    item_accuracy = sum(same_counts) / (4 * len(same_counts))
    all_source_rate = sum(count == 4 for count in same_counts) / len(same_counts)
    require(close(summary["top4_item_accuracy_mean"], item_accuracy), f"top4 item metric mismatch: {path}")
    require(close(summary["top4_all_same_source_rate"], all_source_rate), f"all-source metric mismatch: {path}")
    require(int(summary["queries_with_all_4_same_source"]) == sum(count == 4 for count in same_counts), f"all-source count mismatch: {path}")
    return len(per_query), item_accuracy, all_source_rate


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_baseline_references() -> int:
    rows = read_tsv(AGENT_MEMORY_DIR / "reference" / "memory_baselines.tsv")
    for row in rows:
        path = AGENT_MEMORY_DIR / "reference" / "baselines" / row["baseline_file"]
        payload = read_json(path)
        summary = payload["summary"]
        require(int(summary["num_queries"]) == int(row["n"]), f"baseline n mismatch: {path}")
        require(int(summary["correct"]) == int(row["correct"]), f"baseline correct mismatch: {path}")
        require(close(summary["accuracy"], float(row["accuracy"])), f"baseline accuracy mismatch: {path}")
        require(len(summary["wrong_indices"]) == int(row["self_wrong_count"]), f"baseline wrong count mismatch: {path}")
    return len(rows)


def validate_memory_references() -> int:
    rows = read_tsv(AGENT_MEMORY_DIR / "reference" / "memory_top1.tsv")
    for row in rows:
        require(int(row["answer_correct"]) / int(row["n"]) == float(row["answer_accuracy"]), f"answer ratio mismatch: {row['run_id']}/{row['selector']}")
        require(int(row["selection_correct"]) / int(row["n"]) == float(row["selection_accuracy"]), f"selection ratio mismatch: {row['run_id']}/{row['selector']}")
        if row["score_file"] == "-":
            continue
        path = AGENT_MEMORY_DIR / "scores" / row["score_file"]
        score = read_json(path)
        top1_correct = sum(bool(item["top4"][0]["same_source"]) for item in score["per_query"])
        require(len(score["per_query"]) == int(row["n"]), f"memory score n mismatch: {path}")
        require(top1_correct == int(row["selection_correct"]), f"top1 selection mismatch: {path}")
    return len(rows)


def validate_translation_references() -> int:
    rows = read_tsv(RETRIEVAL_DIR / "reference" / "translation_top4.tsv")
    checked = 0
    for row in rows:
        path = ATTRIBUTION_DIR / row["source_artifact"]
        require(path.is_file(), f"translation reference missing: {path}")
        if path.suffix != ".json":
            continue
        summary = read_json(path)["summary"]
        require(int(summary["num_queries"]) == int(row["num_queries"]), f"translation n mismatch: {path}")
        require(close(summary["top4_item_accuracy_mean"], float(row["top4_item_accuracy_mean"]), 1e-4), f"translation item metric mismatch: {path}")
        require(close(summary["top4_all_same_source_rate"], float(row["top4_all_same_source_rate"]), 1e-4), f"translation all-source mismatch: {path}")
        checked += 1
    return checked


def validate_release() -> None:
    translated = 0
    pools = 0
    for path in sorted([
        *(AGENT_MEMORY_DIR / "data").rglob("*.json"),
        *(RETRIEVAL_DIR / "data").rglob("*.json"),
    ]):
        payload = read_json(path)
        translations = payload.get("records", [{}])[0].get("translations", {}) if payload.get("records") else {}
        if "records" in payload and not set(LANGUAGES).issubset(translations):
            pools += validate_source_pool(path)
        else:
            translated += validate_translation_dataset(path)

    scores = 0
    score_queries = 0
    for path in sorted([
        *(AGENT_MEMORY_DIR / "scores").rglob("*.json"),
        *(RETRIEVAL_DIR / "scores").rglob("*.json"),
    ]):
        count, _, _ = validate_score(path)
        scores += 1
        score_queries += count

    baseline_rows = validate_baseline_references()
    memory_rows = validate_memory_references()
    translation_rows = validate_translation_references()
    manifest_files = validate_manifest()
    print(
        "release OK: "
        f"{translated} translated rows, {pools} source-pool rows, "
        f"{scores} score files/{score_queries} scored queries, "
        f"{baseline_rows} baselines, {memory_rows} memory metrics, "
        f"{translation_rows} JSON retrieval references, {manifest_files} manifested files"
    )


def compare_scores(reference: Path, candidate: Path, strict_ranking: bool) -> None:
    left = read_json(reference)
    right = read_json(candidate)
    require(len(left["per_query"]) == len(right["per_query"]), "candidate query count differs")
    top1_equal = 0
    top4_equal = 0
    for left_query, right_query in zip(left["per_query"], right["per_query"]):
        left_indices = [int(item["train_index"]) for item in left_query["top4"]]
        right_indices = [int(item["train_index"]) for item in right_query["top4"]]
        top1_equal += int(left_indices[0] == right_indices[0])
        top4_equal += int(left_indices == right_indices)
    for key in ("top4_item_accuracy_mean", "top4_all_same_source_rate"):
        require(close(left["summary"][key], right["summary"][key], 1e-6), f"score metric differs: {key}")
    total = len(left["per_query"])
    print(f"score metrics match; top1 rankings {top1_equal}/{total}, exact top4 rankings {top4_equal}/{total}")
    if strict_ranking:
        require(top4_equal == total, "strict top4 ranking comparison failed")


def compare_memory_result(run_id: str, candidate: Path, answer_tolerance: int) -> None:
    expected = {
        row["selector"]: row
        for row in read_tsv(AGENT_MEMORY_DIR / "reference" / "memory_top1.tsv")
        if row["run_id"] == run_id
    }
    payload = read_json(candidate)
    if payload["summary"].get("selection_method") == "random_uniform":
        channels = {"random_seed42": payload["channels"]["random"]}
    else:
        channels = payload["channels"]
    require(channels, f"no result channels found in {candidate}")
    for selector, result in channels.items():
        require(selector in expected, f"unexpected selector for {run_id}: {selector}")
        row = expected[selector]
        summary = result["summary"]
        require(int(summary["top1_retrieval_correct"]) == int(row["selection_correct"]), f"selection count differs: {selector}")
        answer_delta = abs(int(summary["answer_correct"]) - int(row["answer_correct"]))
        require(answer_delta <= answer_tolerance, f"answer count differs by {answer_delta}: {selector}")
        print(f"{run_id}/{selector}: selection exact, answer delta={answer_delta}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--compare-score", nargs=2, metavar=("REFERENCE", "CANDIDATE"), type=Path)
    parser.add_argument("--strict-ranking", action="store_true")
    parser.add_argument("--memory-result", nargs=2, metavar=("RUN_ID", "RESULT"))
    parser.add_argument("--answer-tolerance", type=int, default=1)
    args = parser.parse_args()

    if args.write_manifest:
        write_manifest()
    validate_release()
    if args.compare_score:
        compare_scores(args.compare_score[0], args.compare_score[1], args.strict_ranking)
    if args.memory_result:
        compare_memory_result(
            args.memory_result[0], Path(args.memory_result[1]), args.answer_tolerance
        )


if __name__ == "__main__":
    main()
