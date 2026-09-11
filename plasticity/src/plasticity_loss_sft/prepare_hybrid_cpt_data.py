from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_SOURCES = {
    "general": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "target_tokens": 120_000_000,
        "text_columns": ["text"],
    },
    "scientific": {
        "dataset": "slinusc/PubMedAbstractsSubset",
        "config": None,
        "split": "train",
        "target_tokens": 60_000_000,
        "text_columns": ["abstract", "text", "content", "contents"],
        "title_columns": ["title", "article_title"],
        "id_columns": ["PMID", "pmid", "id", "doi", "article_id"],
    },
    "math": {
        "dataset": "open-web-math/open-web-math",
        "config": None,
        "split": "train",
        "target_tokens": 45_000_000,
        "text_columns": ["text"],
    },
    "code": {
        "dataset": "codeparrot/codeparrot-clean-train",
        "config": None,
        "split": "train",
        "target_tokens": 45_000_000,
        "text_columns": ["content", "code", "text"],
        "substitution_note": "Replaced codeparrot/github-code because datasets>=4 refuses dataset scripts; replaced bigcode/the-stack-smol because it is gated.",
    },
    "qa": {
        "dataset": "Open-Orca/OpenOrca",
        "config": None,
        "split": "train",
        "target_tokens": 30_000_000,
        "text_columns": ["response", "output", "answer", "completion", "text"],
        "prompt_columns": ["question", "instruction", "prompt", "system_prompt"],
    },
}

DEFAULT_DOWNSTREAM_PATHS = (
    "data/prepared_subsets/gsm8k/train.jsonl",
    "data/prepared_subsets/gsm8k/probe.jsonl",
    "data/prepared_subsets/mbpp/train.jsonl",
    "data/prepared_subsets/mbpp/probe.jsonl",
    "data/prepared_subsets/dolly_qa/train.jsonl",
    "data/prepared_subsets/dolly_qa/probe.jsonl",
)


@dataclass
class DomainStats:
    domain: str
    dataset: str
    config: str | None
    split: str
    target_train_tokens: int
    actual_train_tokens: int = 0
    probe_tokens: int = 0
    train_documents: int = 0
    probe_documents: int = 0
    train_chunks: int = 0
    probe_chunks: int = 0
    skipped_documents: int = 0
    duplicate_documents: int = 0
    contamination_matches: int = 0
    output_shards: list[dict[str, Any]] | None = None
    probe_shards: list[dict[str, Any]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a 300M-token hybrid CPT dataset.")
    parser.add_argument("--output_root", default="__CPT_DATA_ROOT__/hybrid_300m")
    parser.add_argument("--cache_dir", default="__CPT_DATA_ROOT__/hf_cache")
    parser.add_argument("--raw_tmp_dir", default="__CPT_DATA_ROOT__/raw_tmp")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--probe_tokens_per_domain", type=int, default=750_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min_doc_tokens", type=int, default=64)
    parser.add_argument("--shard_tokens", type=int, default=8_388_608)
    parser.add_argument("--final_shard_tokens", type=int, default=8_388_608)
    parser.add_argument("--source_specs", default=None, help="Optional JSON file overriding source specs.")
    parser.add_argument("--downstream_paths", nargs="+", default=list(DEFAULT_DOWNSTREAM_PATHS))
    parser.add_argument("--domains", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--max_examples_per_domain", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cache(args)
    output_root = Path(args.output_root)
    train_domain_root = output_root / "train_by_domain"
    final_train_root = output_root / "train"
    probe_root = output_root / "probes"
    manifest_root = output_root / "manifests"
    for path in (train_domain_root, final_train_root, probe_root, manifest_root, Path(args.raw_tmp_dir)):
        path.mkdir(parents=True, exist_ok=True)

    specs = load_source_specs(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, cache_dir=args.cache_dir)
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"Tokenizer {args.tokenizer_name} has no EOS token.")

    downstream_hashes = load_downstream_hashes([Path(path) for path in args.downstream_paths])
    domain_stats = []
    for domain in args.domains:
        if domain not in specs:
            raise ValueError(f"Unknown domain {domain!r}. Available domains: {sorted(specs)}")
        stats = build_domain(
            domain=domain,
            spec=specs[domain],
            tokenizer=tokenizer,
            args=args,
            train_domain_root=train_domain_root,
            probe_root=probe_root,
            downstream_hashes=downstream_hashes,
        )
        domain_stats.append(stats)

    final_shards = interleave_domain_shards(
        domain_stats=domain_stats,
        train_domain_root=train_domain_root,
        final_train_root=final_train_root,
        args=args,
    )
    manifest = build_manifest(args, specs, domain_stats, final_shards)
    write_json(manifest_root / "dataset_manifest.json", manifest)
    write_readme(manifest_root / "README.md", manifest)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


def configure_cache(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))


def load_source_specs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    specs = json.loads(json.dumps(DEFAULT_SOURCES))
    if args.source_specs:
        override = json.loads(Path(args.source_specs).read_text(encoding="utf-8"))
        for domain, values in override.items():
            specs.setdefault(domain, {}).update(values)
    return specs


def build_domain(
    domain: str,
    spec: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
    train_domain_root: Path,
    probe_root: Path,
    downstream_hashes: set[str],
) -> DomainStats:
    domain_train_root = train_domain_root / domain
    domain_probe_root = probe_root / domain
    stats_path = domain_train_root / "domain_manifest.json"
    if stats_path.is_file() and not args.overwrite:
        return domain_stats_from_json(stats_path)
    if args.dry_run:
        return DomainStats(
            domain=domain,
            dataset=str(spec.get("dataset") or spec.get("local_jsonl")),
            config=spec.get("config"),
            split=str(spec.get("split", "train")),
            target_train_tokens=int(spec["target_tokens"]),
        )

    for path in (domain_train_root, domain_probe_root):
        path.mkdir(parents=True, exist_ok=True)

    stats = DomainStats(
        domain=domain,
        dataset=str(spec.get("dataset") or spec.get("local_jsonl")),
        config=spec.get("config"),
        split=str(spec.get("split", "train")),
        target_train_tokens=int(spec["target_tokens"]),
        output_shards=[],
        probe_shards=[],
    )
    source = load_streaming_dataset(spec, args.cache_dir)
    seen_doc_ids: set[str] = set()
    probe_writer = ShardWriter(domain_probe_root, "probe", args.shard_tokens)
    train_writer = ShardWriter(domain_train_root, "train", args.shard_tokens)
    train_buffer: list[int] = []
    probe_buffer: list[int] = []
    train_doc_lengths: list[int] = []
    probe_doc_lengths: list[int] = []
    target_train_tokens = full_chunk_target(stats.target_train_tokens, args.max_seq_length)
    target_probe_tokens = full_chunk_target(args.probe_tokens_per_domain, args.max_seq_length)

    for source_index, row in enumerate(source):
        if args.max_examples_per_domain is not None and source_index >= args.max_examples_per_domain:
            break
        text, raw_id = prepare_source_text(row, spec, source_index)
        if not text:
            stats.skipped_documents += 1
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) < args.min_doc_tokens:
            stats.skipped_documents += 1
            continue
        doc_id = stable_id(domain, raw_id or source_index, text[:256])
        if doc_id in seen_doc_ids:
            stats.duplicate_documents += 1
            continue
        seen_doc_ids.add(doc_id)
        if normalized_hash(text) in downstream_hashes:
            stats.contamination_matches += 1

        doc_tokens = list(map(int, token_ids)) + [int(tokenizer.eos_token_id)]
        if stats.probe_tokens < target_probe_tokens:
            probe_doc_lengths.append(len(doc_tokens))
            stats.probe_documents += 1
            probe_buffer.extend(doc_tokens)
            stats.probe_tokens += flush_chunks(
                writer=probe_writer,
                buffer=probe_buffer,
                domain=domain,
                split="probe",
                max_seq_length=args.max_seq_length,
                token_limit=target_probe_tokens,
            )
            stats.probe_chunks = probe_writer.chunk_count
            continue

        if stats.actual_train_tokens >= target_train_tokens:
            break
        train_doc_lengths.append(len(doc_tokens))
        stats.train_documents += 1
        train_buffer.extend(doc_tokens)
        stats.actual_train_tokens += flush_chunks(
            writer=train_writer,
            buffer=train_buffer,
            domain=domain,
            split="train",
            max_seq_length=args.max_seq_length,
            token_limit=target_train_tokens,
        )
        stats.train_chunks = train_writer.chunk_count
        if stats.actual_train_tokens >= target_train_tokens:
            break

    if stats.probe_tokens < target_probe_tokens:
        raise RuntimeError(
            f"Domain {domain} only produced {stats.probe_tokens} probe tokens; "
            f"target is {target_probe_tokens}."
        )
    if stats.actual_train_tokens < target_train_tokens:
        raise RuntimeError(
            f"Domain {domain} only produced {stats.actual_train_tokens} train tokens; "
            f"target is {target_train_tokens}."
        )

    probe_writer.close()
    train_writer.close()
    stats.output_shards = train_writer.shards
    stats.probe_shards = probe_writer.shards
    stats_dict = stats.__dict__ | {
        "avg_train_doc_tokens": safe_mean(train_doc_lengths),
        "median_train_doc_tokens": safe_median(train_doc_lengths),
        "avg_probe_doc_tokens": safe_mean(probe_doc_lengths),
        "median_probe_doc_tokens": safe_median(probe_doc_lengths),
    }
    write_json(stats_path, stats_dict)
    return DomainStats(**{key: stats_dict[key] for key in DomainStats.__dataclass_fields__})


def load_streaming_dataset(spec: Mapping[str, Any], cache_dir: str) -> Iterable[Mapping[str, Any]]:
    if spec.get("local_jsonl"):
        return load_dataset(
            "json",
            data_files=str(spec["local_jsonl"]),
            split=spec.get("split", "train"),
            streaming=True,
            cache_dir=cache_dir,
        )
    load_args: list[Any] = [spec["dataset"]]
    if spec.get("config"):
        load_args.append(spec["config"])
    load_kwargs = {
        "split": spec.get("split", "train"),
        "streaming": True,
        "cache_dir": cache_dir,
    }
    if spec.get("data_dir"):
        load_kwargs["data_dir"] = spec["data_dir"]
    return load_dataset(*load_args, **load_kwargs)


def domain_stats_from_json(path: Path) -> DomainStats:
    data = json.loads(path.read_text(encoding="utf-8"))
    fields = DomainStats.__dataclass_fields__
    return DomainStats(**{key: data[key] for key in fields if key in data})


def prepare_source_text(row: Mapping[str, Any], spec: Mapping[str, Any], source_index: int) -> tuple[str, str]:
    prompt = first_text(row, spec.get("prompt_columns", []))
    body = first_text(row, spec.get("text_columns", ["text", "content", "code", "response"]))
    title = first_text(row, spec.get("title_columns", []))
    if prompt and body:
        text = f"Question/instruction:\n{prompt}\n\nResponse:\n{body}"
    elif title and body and title != body:
        text = f"{title}\n\n{body}"
    else:
        text = body or prompt or title
    raw_id = first_text(row, spec.get("id_columns", ["id", "sha", "url"])) or str(source_index)
    return clean_text(text), raw_id


def flush_chunks(
    writer: "ShardWriter",
    buffer: list[int],
    domain: str,
    split: str,
    max_seq_length: int,
    token_limit: int,
) -> int:
    added_tokens = 0
    while len(buffer) >= max_seq_length and writer.token_count + max_seq_length <= token_limit:
        chunk = buffer[:max_seq_length]
        del buffer[:max_seq_length]
        writer.write(
            {
                "input_ids": chunk,
                "attention_mask": [1] * len(chunk),
                "token_count": len(chunk),
                "source": domain,
                "split": split,
                "chunk_index": writer.chunk_count,
            }
        )
        added_tokens += max_seq_length
    return added_tokens


class ShardWriter:
    def __init__(self, output_dir: Path, prefix: str, shard_tokens: int) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_tokens = shard_tokens
        self.shard_index = 0
        self.chunk_count = 0
        self.token_count = 0
        self.current_token_count = 0
        self.handle = None
        self.current_path: Path | None = None
        self.shards: list[dict[str, Any]] = []

    def write(self, row: Mapping[str, Any]) -> None:
        if self.handle is None or self.current_token_count + int(row["token_count"]) > self.shard_tokens:
            self.rotate()
        assert self.handle is not None
        self.handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        self.current_token_count += int(row["token_count"])
        self.token_count += int(row["token_count"])
        self.chunk_count += 1

    def rotate(self) -> None:
        self.close()
        self.current_path = self.output_dir / f"{self.prefix}_shard_{self.shard_index:04d}.jsonl"
        self.handle = self.current_path.open("w", encoding="utf-8")
        self.current_token_count = 0
        self.shard_index += 1

    def close(self) -> None:
        if self.handle is None:
            return
        self.handle.close()
        assert self.current_path is not None
        self.shards.append(
            {
                "path": str(self.current_path),
                "tokens": self.current_token_count,
            }
        )
        self.handle = None


def interleave_domain_shards(
    domain_stats: list[DomainStats],
    train_domain_root: Path,
    final_train_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    manifest_path = final_train_root / "train_manifest.json"
    if manifest_path.is_file() and not args.overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))["shards"]
    if args.dry_run:
        return []

    rng = random.Random(args.seed)
    iterators = {}
    remaining = {}
    for stats in domain_stats:
        paths = sorted((train_domain_root / stats.domain).glob("train_shard_*.jsonl"))
        iterators[stats.domain] = iter_jsonl_files(paths)
        remaining[stats.domain] = int(stats.actual_train_tokens)

    writer = ShardWriter(final_train_root, "shard", args.final_shard_tokens)
    while any(tokens > 0 for tokens in remaining.values()):
        active = [domain for domain, tokens in remaining.items() if tokens > 0]
        weights = [remaining[domain] for domain in active]
        domain = rng.choices(active, weights=weights, k=1)[0]
        try:
            row = next(iterators[domain])
        except StopIteration:
            remaining[domain] = 0
            continue
        writer.write(row)
        remaining[domain] -= int(row["token_count"])
    writer.close()
    write_json(manifest_path, {"shards": writer.shards, "seed": args.seed})
    return writer.shards


def iter_jsonl_files(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def build_manifest(
    args: argparse.Namespace,
    specs: Mapping[str, Any],
    domain_stats: list[DomainStats],
    final_shards: list[dict[str, Any]],
) -> dict[str, Any]:
    domain_rows = []
    total_train_tokens = sum(stats.actual_train_tokens for stats in domain_stats)
    total_probe_tokens = sum(stats.probe_tokens for stats in domain_stats)
    for stats in domain_stats:
        domain_manifest = Path(args.output_root) / "train_by_domain" / stats.domain / "domain_manifest.json"
        row = json.loads(domain_manifest.read_text(encoding="utf-8")) if domain_manifest.is_file() else stats.__dict__
        row["realized_train_fraction"] = (
            row["actual_train_tokens"] / total_train_tokens if total_train_tokens else None
        )
        domain_rows.append(row)
    return {
        "summary": {
            "output_root": args.output_root,
            "tokenizer_name": args.tokenizer_name,
            "seed": args.seed,
            "max_seq_length": args.max_seq_length,
            "total_train_tokens": total_train_tokens,
            "total_probe_tokens": total_probe_tokens,
            "final_train_shards": len(final_shards),
            "final_train_path": str(Path(args.output_root) / "train"),
            "single_stream_path": str(Path(args.output_root) / "train"),
        },
        "sources": specs,
        "domains": domain_rows,
        "final_shards": final_shards,
        "validation": {
            "probe_reserved_before_training": True,
            "downstream_overlap_check": "normalized exact-text hash over local GSM8K/MBPP/Dolly examples",
            "home_directory_cache_allowed": False,
        },
    }


def write_readme(path: Path, manifest: Mapping[str, Any]) -> None:
    rows = []
    total = manifest["summary"]["total_train_tokens"]
    for domain in manifest["domains"]:
        frac = 100 * domain["actual_train_tokens"] / total if total else 0.0
        rows.append(
            "| {domain} | {train:,} | {probe:,} | {docs:,} | {frac:.2f}% |".format(
                domain=domain["domain"],
                train=domain["actual_train_tokens"],
                probe=domain["probe_tokens"],
                docs=domain["train_documents"],
                frac=frac,
            )
        )
    text = "\n".join(
        [
            "# Hybrid 300M CPT Dataset",
            "",
            f"Tokenizer: `{manifest['summary']['tokenizer_name']}`",
            f"Seed: `{manifest['summary']['seed']}`",
            f"Train tokens: `{manifest['summary']['total_train_tokens']:,}`",
            f"Probe tokens: `{manifest['summary']['total_probe_tokens']:,}`",
            "",
            "| Domain | Train tokens | Probe tokens | Train docs | Fraction |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "Final interleaved training shards are under `train/`.",
            "Domain-specific intermediate shards are under `train_by_domain/`.",
            "Fixed held-out probes are under `probes/`.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def load_downstream_hashes(paths: list[Path]) -> set[str]:
    hashes = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for text in collect_text_values(row):
                    hashes.add(normalized_hash(text))
    return hashes


def collect_text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        out = []
        for inner in value.values():
            out.extend(collect_text_values(inner))
        return out
    if isinstance(value, list):
        out = []
        for inner in value:
            out.extend(collect_text_values(inner))
        return out
    return []


def first_text(row: Mapping[str, Any], columns: Iterable[str]) -> str:
    for column in columns:
        if column not in row:
            continue
        value = normalize_text_value(row[column])
        if value:
            return value
    return ""


def normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(normalize_text_value(item) for item in value if normalize_text_value(item)).strip()
    return str(value).strip()


def clean_text(value: str) -> str:
    return "\n".join(line.strip() for line in value.replace("\r", "\n").split("\n") if line.strip())


def normalized_hash(value: str) -> str:
    normalized = " ".join(value.lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def stable_id(*parts: object) -> str:
    text = "::".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def full_chunk_target(tokens: int, max_seq_length: int) -> int:
    return (tokens // max_seq_length) * max_seq_length


def safe_mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_median(values: list[int]) -> float:
    return float(median(values)) if values else 0.0


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
