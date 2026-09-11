from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-tokenize an existing packed CPT stream with a different tokenizer."
    )
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--source_tokenizer_path", required=True)
    parser.add_argument("--target_tokenizer_path", required=True)
    parser.add_argument("--source_tokenizer_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--target_tokenizer_name", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--roundtrip_checks", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if not (source_root / "manifests" / "dataset_manifest.json").is_file():
        raise FileNotFoundError(f"Not a hybrid CPT root: {source_root}")
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output already exists and is non-empty: {output_root}")
    if args.max_seq_length <= 0:
        raise ValueError("--max_seq_length must be positive")

    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer_path, local_files_only=True)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer_path, local_files_only=True)
    if source_tokenizer.eos_token_id is None or target_tokenizer.eos_token_id is None:
        raise RuntimeError("Both tokenizers must define eos_token_id")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source_root / "manifests" / "dataset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    reports: dict[str, Any] = {}
    reports["train"] = convert_stream(
        input_paths=sorted((source_root / "train").glob("shard_*.jsonl")),
        output_dir=output_root / "train",
        split="train",
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_seq_length=args.max_seq_length,
        roundtrip_checks=args.roundtrip_checks,
    )
    probe_reports = {}
    for domain_dir in sorted((source_root / "probes").iterdir()):
        if not domain_dir.is_dir():
            continue
        probe_reports[domain_dir.name] = convert_stream(
            input_paths=sorted(domain_dir.glob("probe_*.jsonl")),
            output_dir=output_root / "probes" / domain_dir.name,
            split="probe",
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            max_seq_length=args.max_seq_length,
            roundtrip_checks=args.roundtrip_checks,
        )

    copied_manifest = output_root / "manifests" / "source_dataset_manifest.json"
    copied_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_manifest_path, copied_manifest)
    manifest = {
        "serialization": "decoded source-tokenizer text with source EOS translated to target EOS, then re-tokenized",
        "source_root": str(source_root),
        "source_dataset_manifest_sha256": sha256_file(source_manifest_path),
        "source_tokenizer": {"name": args.source_tokenizer_name, "eos_token_id": source_tokenizer.eos_token_id, "vocab_size": source_tokenizer.vocab_size},
        "target_tokenizer": {"name": args.target_tokenizer_name, "eos_token_id": target_tokenizer.eos_token_id, "vocab_size": target_tokenizer.vocab_size},
        "max_seq_length": args.max_seq_length,
        "source_mixture": source_manifest["domains"],
        "train": reports["train"],
        "probes": probe_reports,
        "validation": {
            "source_roundtrip_checks_per_stream": args.roundtrip_checks,
            "source_dataset_reloaded": False,
            "source_examples_and_packed_stream_order": "preserved from the materialized source CPT stream",
        },
    }
    write_json(output_root / "manifests" / "serialization_manifest.json", manifest)
    print(json.dumps({"output_root": str(output_root), "train": reports["train"], "probes": probe_reports}, indent=2, sort_keys=True))


def convert_stream(
    input_paths: list[Path],
    output_dir: Path,
    split: str,
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    roundtrip_checks: int,
) -> dict[str, Any]:
    if not input_paths:
        raise FileNotFoundError(f"No input shards for {output_dir}")
    writer = PackedWriter(output_dir, "shard" if split == "train" else "probe", max_seq_length)
    input_rows = 0
    input_tokens = 0
    roundtrip_checked = 0
    roundtrip_passed = 0
    roundtrip_mismatched = 0
    source_labels: set[str] = set()
    for row in iter_jsonl(input_paths):
        source_ids = [int(token_id) for token_id in row["input_ids"]]
        if not source_ids:
            continue
        if roundtrip_checked < roundtrip_checks:
            roundtrip_checked += 1
            if verify_source_roundtrip(source_ids, source_tokenizer):
                roundtrip_passed += 1
            else:
                roundtrip_mismatched += 1
        target_ids = translate_ids(source_ids, source_tokenizer, target_tokenizer)
        writer.add(target_ids, str(row.get("source", "unknown")), split)
        input_rows += 1
        input_tokens += len(source_ids)
        source_labels.add(str(row.get("source", "unknown")))
    writer.close()
    return {
        "input_paths": [str(path) for path in input_paths],
        "input_rows": input_rows,
        "input_tokens": input_tokens,
        "output_rows": writer.rows_written,
        "output_tokens": writer.tokens_written,
        "source_labels": sorted(source_labels),
        "source_roundtrip_checks": roundtrip_checked,
        "source_roundtrip_checks_passed": roundtrip_passed,
        "source_roundtrip_checks_mismatched": roundtrip_mismatched,
        "source_roundtrip_note": "A mismatch is expected when a packed record begins or ends inside a tokenizer merge; decoded text and EOS boundaries are still preserved.",
        "output_shards": writer.shards,
    }


def translate_ids(
    source_ids: list[int],
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    translated: list[int] = []
    start = 0
    source_eos = int(source_tokenizer.eos_token_id)
    target_eos = int(target_tokenizer.eos_token_id)
    for index, token_id in enumerate(source_ids):
        if token_id != source_eos:
            continue
        translated.extend(tokenize_source_segment(source_ids[start:index], source_tokenizer, target_tokenizer))
        translated.append(target_eos)
        start = index + 1
    translated.extend(tokenize_source_segment(source_ids[start:], source_tokenizer, target_tokenizer))
    return translated


def tokenize_source_segment(
    source_ids: list[int],
    source_tokenizer: PreTrainedTokenizerBase,
    target_tokenizer: PreTrainedTokenizerBase,
) -> list[int]:
    if not source_ids:
        return []
    text = source_tokenizer.decode(source_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return [int(token_id) for token_id in target_tokenizer(text, add_special_tokens=False)["input_ids"]]


def verify_source_roundtrip(source_ids: list[int], source_tokenizer: PreTrainedTokenizerBase) -> bool:
    rebuilt: list[int] = []
    start = 0
    eos_token_id = int(source_tokenizer.eos_token_id)
    for index, token_id in enumerate(source_ids):
        if token_id != eos_token_id:
            continue
        rebuilt.extend(reencode_source_segment(source_ids[start:index], source_tokenizer))
        rebuilt.append(eos_token_id)
        start = index + 1
    rebuilt.extend(reencode_source_segment(source_ids[start:], source_tokenizer))
    return rebuilt == source_ids


def reencode_source_segment(source_ids: list[int], source_tokenizer: PreTrainedTokenizerBase) -> list[int]:
    if not source_ids:
        return []
    text = source_tokenizer.decode(source_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return [int(token_id) for token_id in source_tokenizer(text, add_special_tokens=False)["input_ids"]]


class PackedWriter:
    def __init__(self, output_dir: Path, prefix: str, max_seq_length: int) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_seq_length = max_seq_length
        self.buffer: list[int] = []
        self.source_labels: set[str] = set()
        self.shard_index = 0
        self.rows_written = 0
        self.tokens_written = 0
        self.shards: list[dict[str, Any]] = []
        self._handle = None
        self._current_path: Path | None = None
        self._current_tokens = 0
        self._shard_token_limit = 8_388_608
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, token_ids: list[int], source_label: str, split: str) -> None:
        self.buffer.extend(token_ids)
        self.source_labels.add(source_label)
        while len(self.buffer) >= self.max_seq_length:
            chunk = self.buffer[: self.max_seq_length]
            del self.buffer[: self.max_seq_length]
            source = source_label if len(self.source_labels) == 1 else "mixed"
            self._write({"input_ids": chunk, "attention_mask": [1] * len(chunk), "token_count": len(chunk), "source": source, "split": split, "chunk_index": self.rows_written})
            self.source_labels.clear()

    def _write(self, row: dict[str, Any]) -> None:
        if self._handle is None or self._current_tokens + row["token_count"] > self._shard_token_limit:
            self._rotate()
        assert self._handle is not None
        self._handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        self._current_tokens += int(row["token_count"])
        self.rows_written += 1
        self.tokens_written += int(row["token_count"])

    def _rotate(self) -> None:
        self._close_handle()
        self._current_path = self.output_dir / f"{self.prefix}_shard_{self.shard_index:04d}.jsonl"
        self._handle = self._current_path.open("w", encoding="utf-8")
        self._current_tokens = 0
        self.shard_index += 1

    def _close_handle(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        assert self._current_path is not None
        self.shards.append({"path": str(self._current_path), "tokens": self._current_tokens})
        self._handle = None

    def close(self) -> None:
        self._close_handle()


def iter_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
