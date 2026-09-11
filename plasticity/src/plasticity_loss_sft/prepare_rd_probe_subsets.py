from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean, median
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_PROBES = {
    "pubmed": {
        "role": "covered",
        "kind": "local_messages",
        "paths": [
            "data/prepared_subsets/bio/train.jsonl",
            "data/prepared_subsets/bio/probe.jsonl",
        ],
        "task": "pubmed",
        "source_dataset": "local:data/prepared_subsets/bio",
        "formatting": "Existing Bio/PubMed title-to-abstract chat format.",
    },
    "math": {
        "role": "covered",
        "kind": "local_messages",
        "paths": [
            "data/prepared_subsets/gsm8k/train.jsonl",
            "data/prepared_subsets/gsm8k/probe.jsonl",
        ],
        "task": "math",
        "source_dataset": "local:data/prepared_subsets/gsm8k",
        "formatting": "Existing GSM8K question-to-solution chat format.",
    },
    "qa": {
        "role": "covered",
        "kind": "local_messages",
        "paths": [
            "data/prepared_subsets/dolly_qa/train.jsonl",
            "data/prepared_subsets/dolly_qa/probe.jsonl",
        ],
        "task": "qa",
        "source_dataset": "local:data/prepared_subsets/dolly_qa",
        "formatting": "Existing Dolly-QA instruction/reference-to-answer chat format.",
    },
    "code": {
        "role": "unsupported",
        "kind": "local_messages",
        "paths": [
            "data/prepared_subsets/mbpp/train.jsonl",
            "data/prepared_subsets/mbpp/probe.jsonl",
        ],
        "task": "code",
        "source_dataset": "local:data/prepared_subsets/mbpp",
        "formatting": "Existing MBPP prompt/tests-to-code chat format.",
    },
    "multilingual": {
        "role": "unsupported",
        "kind": "translation",
        "dataset": "Helsinki-NLP/opus-100",
        "config": "en-zh",
        "split": "train",
        "task": "multilingual",
        "source_dataset": "Helsinki-NLP/opus-100/en-zh",
        "formatting": "Translate English source text into Chinese target text.",
    },
    "legal": {
        "role": "unsupported",
        "kind": "legal_summarization",
        "dataset": "satesilka/LegalSumm",
        "config": None,
        "split": "validation",
        "task": "legal",
        "source_dataset": "satesilka/LegalSumm",
        "formatting": "Summarize a legal document; target is the human-written legal summary.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed R_D probe subsets for Section 6.")
    parser.add_argument("--output_root", default="__CPT_DATA_ROOT__/rd_probes")
    parser.add_argument("--cache_dir", default="__CPT_DATA_ROOT__/hf_cache")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--target_examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min_target_tokens", type=int, default=16)
    parser.add_argument("--max_input_tokens", type=int, default=768)
    parser.add_argument("--max_target_tokens", type=int, default=512)
    parser.add_argument("--max_stream_examples", type=int, default=20000)
    parser.add_argument("--probe_specs", default=None, help="Optional JSON file overriding probe specs.")
    parser.add_argument("--hybrid_cpt_train_root", default="__CPT_DATA_ROOT__/hybrid_300m/train")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cache(args.cache_dir)
    output_root = Path(args.output_root)
    manifest_root = output_root / "manifest"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    specs = load_specs(args.probe_specs)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, cache_dir=args.cache_dir)
    hybrid_hashes, hybrid_checked = load_hybrid_hashes(Path(args.hybrid_cpt_train_root))

    summaries = []
    for probe_name, spec in specs.items():
        probe_dir = output_root / probe_name
        probe_path = probe_dir / "probe.jsonl"
        stats_path = probe_dir / "stats.json"
        if probe_path.is_file() and stats_path.is_file() and not args.overwrite:
            summaries.append(json.loads(stats_path.read_text(encoding="utf-8")))
            continue
        probe_dir.mkdir(parents=True, exist_ok=True)
        if spec["kind"] == "local_messages":
            records = build_local_messages_probe(probe_name, spec, tokenizer, args)
        elif spec["kind"] == "translation":
            records = build_translation_probe(probe_name, spec, tokenizer, args)
        elif spec["kind"] == "legal_summarization":
            records = build_legal_summarization_probe(probe_name, spec, tokenizer, args)
        else:
            raise ValueError(f"Unknown probe kind for {probe_name}: {spec['kind']}")

        for row in records:
            text_hash = normalized_hash(messages_text(row["messages"]))
            row.setdefault("metadata", {})["exact_overlap_with_hybrid_cpt"] = text_hash in hybrid_hashes if hybrid_checked else None

        write_jsonl(probe_path, records)
        stats = summarize_probe(probe_name, spec, records, tokenizer, args, hybrid_checked)
        write_json(stats_path, stats)
        summaries.append(stats)

    manifest = {
        "output_root": str(output_root),
        "tokenizer_name": args.tokenizer_name,
        "seed": args.seed,
        "target_examples": args.target_examples,
        "min_target_tokens": args.min_target_tokens,
        "max_input_tokens": args.max_input_tokens,
        "max_target_tokens": args.max_target_tokens,
        "hybrid_cpt_train_root": args.hybrid_cpt_train_root,
        "hybrid_overlap_checked": hybrid_checked,
        "probes": summaries,
    }
    write_json(manifest_root / "probes.json", manifest)
    write_readme(manifest_root / "README.md", manifest)
    print(json.dumps({"output_root": str(output_root), "probes": concise_rows(summaries)}, indent=2, sort_keys=True))


def configure_cache(cache_dir: str) -> None:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / "xdg"))


def load_specs(path: str | None) -> dict[str, dict[str, Any]]:
    specs = json.loads(json.dumps(DEFAULT_PROBES))
    if path:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        for name, values in overrides.items():
            specs.setdefault(name, {}).update(values)
    return specs


def build_local_messages_probe(
    probe_name: str,
    spec: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    for path_value in spec["paths"]:
        path = Path(path_value)
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                messages = normalize_messages(row.get("messages") or [])
                if not valid_messages(messages, tokenizer, args):
                    continue
                key = normalized_hash(messages_text(messages))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(make_record(probe_name, spec, row.get("example_id"), messages, row.get("metadata", {})))
    rng = random.Random(args.seed + stable_int(probe_name))
    rng.shuffle(candidates)
    return candidates[: args.target_examples]


def build_translation_probe(
    probe_name: str,
    spec: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for index, row in enumerate(stream_dataset(spec, args)):
        translation = row.get("translation") or {}
        if not isinstance(translation, Mapping):
            continue
        source = first_present(translation, ("en", "eng"))
        target = first_present(translation, ("zh", "zho", "cmn", "ja", "jpn"))
        if not source or not target:
            continue
        messages = [
            {"role": "user", "content": "Translate the following English text into Chinese.\n\n" + clean_text(source)},
            {"role": "assistant", "content": clean_text(target)},
        ]
        if not valid_messages(messages, tokenizer, args):
            continue
        key = normalized_hash(messages_text(messages))
        if key in seen:
            continue
        seen.add(key)
        records.append(make_record(probe_name, spec, row.get("id") or index, messages, {"source_index": index}))
        if len(records) >= args.target_examples:
            break
    ensure_enough(probe_name, records, args.target_examples)
    return records


def build_legal_summarization_probe(
    probe_name: str,
    spec: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for index, row in enumerate(stream_dataset(spec, args)):
        document = first_present(row, ("text", "document", "article", "input"))
        summary = first_present(row, ("summary", "target", "output", "answer"))
        if not document or not summary:
            continue
        document = truncate_to_tokens(clean_text(document), tokenizer, args.max_input_tokens)
        messages = [
            {"role": "user", "content": "Summarize the following legal document.\n\n" + document},
            {"role": "assistant", "content": clean_text(summary)},
        ]
        if not valid_messages(messages, tokenizer, args):
            continue
        key = normalized_hash(messages_text(messages))
        if key in seen:
            continue
        seen.add(key)
        records.append(make_record(probe_name, spec, row.get("id") or index, messages, {"source_index": index}))
        if len(records) >= args.target_examples:
            break
    ensure_enough(probe_name, records, args.target_examples)
    return records


def stream_dataset(spec: Mapping[str, Any], args: argparse.Namespace) -> Iterable[Mapping[str, Any]]:
    load_args = [spec["dataset"]]
    if spec.get("config"):
        load_args.append(spec["config"])
    dataset = load_dataset(*load_args, split=spec.get("split", "train"), streaming=True, cache_dir=args.cache_dir)
    for index, row in enumerate(dataset):
        if index >= args.max_stream_examples:
            break
        yield row


def make_record(
    probe_name: str,
    spec: Mapping[str, Any],
    example_id: Any,
    messages: list[dict[str, str]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    stable = stable_id(probe_name, example_id, messages_text(messages))
    return {
        "task": spec.get("task", probe_name),
        "subset": "probe",
        "example_id": stable,
        "messages": messages,
        "metadata": dict(metadata)
        | {
            "probe_name": probe_name,
            "role": spec["role"],
            "source_dataset": spec.get("source_dataset") or spec.get("dataset"),
            "formatting": spec.get("formatting"),
        },
    }


def valid_messages(messages: list[dict[str, str]], tokenizer: PreTrainedTokenizerBase, args: argparse.Namespace) -> bool:
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return False
    prompt_tokens = len(tokenizer(messages[0]["content"], add_special_tokens=False)["input_ids"])
    target_tokens = len(tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"])
    if target_tokens < args.min_target_tokens:
        return False
    if target_tokens > args.max_target_tokens:
        return False
    if prompt_tokens > args.max_input_tokens and messages[0]["content"].startswith("Summarize the following legal document"):
        return False
    return True


def summarize_probe(
    probe_name: str,
    spec: Mapping[str, Any],
    records: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
    hybrid_checked: bool,
) -> dict[str, Any]:
    input_lengths = []
    target_lengths = []
    overlaps = 0
    for row in records:
        messages = row["messages"]
        input_lengths.append(len(tokenizer(messages[0]["content"], add_special_tokens=False)["input_ids"]))
        target_lengths.append(len(tokenizer(messages[-1]["content"], add_special_tokens=False)["input_ids"]))
        overlaps += int(bool(row.get("metadata", {}).get("exact_overlap_with_hybrid_cpt")))
    return {
        "probe": probe_name,
        "role": spec["role"],
        "dataset": spec.get("source_dataset") or spec.get("dataset"),
        "split": spec.get("split"),
        "config": spec.get("config"),
        "path": str(Path(args.output_root) / probe_name / "probe.jsonl"),
        "examples": len(records),
        "input_tokens": sum(input_lengths),
        "target_tokens": sum(target_lengths),
        "avg_input_tokens": mean(input_lengths) if input_lengths else 0.0,
        "avg_target_tokens": mean(target_lengths) if target_lengths else 0.0,
        "median_input_tokens": median(input_lengths) if input_lengths else 0.0,
        "median_target_tokens": median(target_lengths) if target_lengths else 0.0,
        "min_target_tokens": min(target_lengths) if target_lengths else 0,
        "max_target_tokens": max(target_lengths) if target_lengths else 0,
        "seed": args.seed,
        "formatting": spec.get("formatting"),
        "hybrid_overlap_checked": hybrid_checked,
        "exact_overlap_with_hybrid_cpt": overlaps if hybrid_checked else None,
    }


def load_hybrid_hashes(path: Path) -> tuple[set[str], bool]:
    if not path.exists():
        return set(), False
    hashes = set()
    paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for shard in paths:
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "text" in row:
                    hashes.add(normalized_hash(str(row["text"])))
    return hashes, True


def normalize_messages(messages: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    out = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = clean_text(str(message.get("content") or ""))
        if role and content:
            out.append({"role": role, "content": content})
    return out


def truncate_to_tokens(text: str, tokenizer: PreTrainedTokenizerBase, max_tokens: int) -> str:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def ensure_enough(probe_name: str, records: list[dict[str, Any]], target: int) -> None:
    if len(records) < target:
        raise RuntimeError(f"Probe {probe_name} produced {len(records)} examples, below target {target}.")


def first_present(row: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row:
            value = normalize_text(row[key])
            if value:
                return value
    return ""


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def clean_text(value: str) -> str:
    return "\n".join(line.strip() for line in value.replace("\r", "\n").split("\n") if line.strip())


def messages_text(messages: list[Mapping[str, str]]) -> str:
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def normalized_hash(value: str) -> str:
    return hashlib.sha1(" ".join(value.lower().split()).encode("utf-8")).hexdigest()


def stable_id(*parts: object) -> str:
    return hashlib.sha1("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def stable_int(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def write_readme(path: Path, manifest: Mapping[str, Any]) -> None:
    rows = []
    for probe in manifest["probes"]:
        rows.append(
            "| {probe} | {role} | {dataset} | {examples:,} | {target_tokens:,} |".format(
                probe=probe["probe"],
                role=probe["role"],
                dataset=probe["dataset"],
                examples=probe["examples"],
                target_tokens=probe["target_tokens"],
            )
        )
    text = "\n".join(
        [
            "# Fixed R_D Probe Subsets",
            "",
            f"Tokenizer: `{manifest['tokenizer_name']}`",
            f"Seed: `{manifest['seed']}`",
            f"Target examples per probe: `{manifest['target_examples']}`",
            f"Hybrid CPT overlap checked: `{manifest['hybrid_overlap_checked']}`",
            "",
            "| Probe | Role | Dataset | Examples | Target tokens |",
            "|---|---|---|---:|---:|",
            *rows,
            "",
            "Each probe is saved as `probe.jsonl` with the existing chat `messages` schema.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def concise_rows(summaries: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "probe": row["probe"],
            "role": row["role"],
            "examples": row["examples"],
            "target_tokens": row["target_tokens"],
            "path": row["path"],
        }
        for row in summaries
    ]


if __name__ == "__main__":
    main()
