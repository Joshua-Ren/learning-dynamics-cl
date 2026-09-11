from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_TEXT_COLUMNS = ("abstract", "content", "contents", "text", "article")
DEFAULT_TITLE_COLUMNS = ("title", "article_title")
DEFAULT_ID_COLUMNS = ("PMID", "pmid", "id", "doi", "article_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare PubMed CPT chunks and held-out supervised Bio train/probe subsets."
    )
    parser.add_argument("--source_dataset", default="slinusc/PubMedAbstractsSubset")
    parser.add_argument("--source_config", default=None)
    parser.add_argument("--source_split", default="train")
    parser.add_argument(
        "--local_jsonl",
        default=None,
        help="Optional local JSONL source. When set, source_dataset/source_config are ignored.",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default="data/cpt_pubmed")
    parser.add_argument("--bio_output_dir", default="data/prepared_subsets/bio")
    parser.add_argument("--tokenizer_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_cpt_tokens", type=int, default=100_000_000)
    parser.add_argument("--bio_train_size", type=int, default=1000)
    parser.add_argument("--bio_probe_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min_doc_tokens", type=int, default=64)
    parser.add_argument("--max_source_docs", type=int, default=None)
    parser.add_argument(
        "--text_columns",
        default="auto",
        help="Comma-separated body/text columns, or 'auto'. First non-empty column is used.",
    )
    parser.add_argument(
        "--title_columns",
        default="auto",
        help="Comma-separated title columns, or 'auto'. First non-empty column is used.",
    )
    parser.add_argument(
        "--id_columns",
        default="auto",
        help="Comma-separated stable id columns, or 'auto'. First non-empty column is used.",
    )
    parser.add_argument("--year_column", default="year")
    parser.add_argument("--min_year", type=int, default=None)
    parser.add_argument("--cpt_output_name", default="cpt_train.jsonl")
    parser.add_argument("--print_examples", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_seq_length <= 1:
        raise ValueError("--max_seq_length must be greater than 1")
    if args.max_cpt_tokens <= 0:
        raise ValueError("--max_cpt_tokens must be positive")
    target_cpt_tokens = (args.max_cpt_tokens // args.max_seq_length) * args.max_seq_length
    if target_cpt_tokens == 0:
        raise ValueError("--max_cpt_tokens must be at least --max_seq_length for fixed-length CPT chunks")
    if args.bio_train_size < 0 or args.bio_probe_size < 0:
        raise ValueError("Bio subset sizes must be non-negative")

    output_dir = Path(args.output_dir)
    bio_output_dir = Path(args.bio_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bio_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, cache_dir=args.cache_dir)
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"Tokenizer {args.tokenizer_name} has no eos_token_id")

    source = load_source_dataset(args)
    text_columns = parse_column_list(args.text_columns, DEFAULT_TEXT_COLUMNS)
    title_columns = parse_column_list(args.title_columns, DEFAULT_TITLE_COLUMNS)
    id_columns = parse_column_list(args.id_columns, DEFAULT_ID_COLUMNS)

    heldout_needed = args.bio_train_size + args.bio_probe_size
    heldout_docs: list[PreparedDocument] = []
    heldout_ids: set[str] = set()
    cpt_doc_count = 0
    cpt_token_count = 0
    cpt_chunk_count = 0
    skipped_docs = 0
    seen_doc_ids: set[str] = set()
    buffer_ids: list[int] = []
    buffer_doc_ids: list[str] = []

    cpt_path = output_dir / args.cpt_output_name
    with cpt_path.open("w", encoding="utf-8") as cpt_handle:
        for source_index, row in enumerate(source):
            if args.max_source_docs is not None and source_index >= args.max_source_docs:
                break
            prepared = prepare_document(
                row=row,
                source_index=source_index,
                tokenizer=tokenizer,
                text_columns=text_columns,
                title_columns=title_columns,
                id_columns=id_columns,
                year_column=args.year_column,
                min_year=args.min_year,
                min_doc_tokens=args.min_doc_tokens,
            )
            if prepared is None:
                skipped_docs += 1
                continue
            if prepared.doc_id in seen_doc_ids:
                skipped_docs += 1
                continue
            seen_doc_ids.add(prepared.doc_id)

            if len(heldout_docs) < heldout_needed:
                heldout_docs.append(prepared)
                heldout_ids.add(prepared.doc_id)
                continue

            assert prepared.doc_id not in heldout_ids
            doc_token_ids = prepared.token_ids + [int(tokenizer.eos_token_id)]
            buffer_ids.extend(doc_token_ids)
            buffer_doc_ids.extend([prepared.doc_id] * len(doc_token_ids))
            cpt_doc_count += 1

            while len(buffer_ids) >= args.max_seq_length and cpt_token_count < target_cpt_tokens:
                chunk_ids = buffer_ids[: args.max_seq_length]
                chunk_doc_ids = buffer_doc_ids[: args.max_seq_length]
                del buffer_ids[: args.max_seq_length]
                del buffer_doc_ids[: args.max_seq_length]
                cpt_chunk_count += 1
                cpt_token_count += args.max_seq_length
                cpt_handle.write(
                    json.dumps(
                        {
                            "input_ids": chunk_ids,
                            "attention_mask": [1] * len(chunk_ids),
                            "token_count": len(chunk_ids),
                            "chunk_index": cpt_chunk_count - 1,
                            "doc_ids": unique_in_order(chunk_doc_ids),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )
            if cpt_token_count >= target_cpt_tokens:
                break

    if len(heldout_docs) < heldout_needed:
        raise RuntimeError(
            f"Only collected {len(heldout_docs)} valid held-out Bio docs, but {heldout_needed} are required. "
            "Lower min_doc_tokens or use a larger source."
        )
    if cpt_chunk_count == 0:
        raise RuntimeError("CPT preparation produced zero chunks. Lower min_doc_tokens/max_seq_length or increase source docs.")

    bio_train_docs = heldout_docs[: args.bio_train_size]
    bio_probe_docs = heldout_docs[args.bio_train_size : heldout_needed]
    bio_train_records = [bio_record(doc, "train") for doc in bio_train_docs]
    bio_probe_records = [bio_record(doc, "probe") for doc in bio_probe_docs]
    write_jsonl(bio_output_dir / "train.jsonl", bio_train_records)
    write_jsonl(bio_output_dir / "probe.jsonl", bio_probe_records)

    selected_ids = {
        "seed": args.seed,
        "source_dataset": source_name(args),
        "train_ids": [doc.doc_id for doc in bio_train_docs],
        "probe_ids": [doc.doc_id for doc in bio_probe_docs],
        "heldout_ids_disjoint_from_cpt": True,
        "heldout_strategy": "first_valid_documents_are_reserved_before_cpt_streaming",
    }
    write_json(bio_output_dir / "selected_ids.json", selected_ids)

    bio_stats = {
        "train": compute_bio_stats(bio_train_records, tokenizer),
        "probe": compute_bio_stats(bio_probe_records, tokenizer),
    }
    write_json(bio_output_dir / "stats.json", bio_stats)

    manifest = {
        "source_dataset": source_name(args),
        "source_split": args.source_split,
        "tokenizer_name": args.tokenizer_name,
        "max_seq_length": args.max_seq_length,
        "requested_max_cpt_tokens": args.max_cpt_tokens,
        "target_cpt_tokens_full_chunks": target_cpt_tokens,
        "actual_cpt_tokens": cpt_token_count,
        "approx_total_tokens": cpt_token_count,
        "cpt_chunks": cpt_chunk_count,
        "cpt_documents": cpt_doc_count,
        "cpt_path": str(cpt_path),
        "bio_output_dir": str(bio_output_dir),
        "bio_train_path": str(bio_output_dir / "train.jsonl"),
        "bio_probe_path": str(bio_output_dir / "probe.jsonl"),
        "bio_train_size": len(bio_train_records),
        "bio_probe_size": len(bio_probe_records),
        "heldout_documents": len(heldout_docs),
        "heldout_doc_ids_sha1": stable_id(*[doc.doc_id for doc in heldout_docs]),
        "skipped_documents": skipped_docs,
        "unique_source_documents_seen": len(seen_doc_ids),
        "min_doc_tokens": args.min_doc_tokens,
        "min_year": args.min_year,
        "text_columns": text_columns,
        "title_columns": title_columns,
        "id_columns": id_columns,
        "bio_stats": bio_stats,
        "format": {
            "cpt": "jsonl rows with input_ids, attention_mask, token_count, chunk_index, doc_ids",
            "bio": "existing supervised messages JSONL compatible with load_instruction_dataset",
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    print_examples("bio_train", bio_train_records, args.print_examples)
    print_examples("bio_probe", bio_probe_records, args.print_examples)


def load_source_dataset(args: argparse.Namespace) -> Iterable[Mapping[str, Any]]:
    if args.local_jsonl:
        return load_dataset("json", data_files=str(args.local_jsonl), split="train", streaming=True)
    load_args: list[Any] = [args.source_dataset]
    if args.source_config:
        load_args.append(args.source_config)
    return load_dataset(
        *load_args,
        split=args.source_split,
        streaming=True,
        cache_dir=args.cache_dir,
    )


def parse_column_list(value: str, default: tuple[str, ...]) -> list[str]:
    if value == "auto":
        return list(default)
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise ValueError("Column list cannot be empty")
    return columns


class PreparedDocument:
    def __init__(
        self,
        doc_id: str,
        title: str,
        body: str,
        text: str,
        token_ids: list[int],
        metadata: dict[str, Any],
    ) -> None:
        self.doc_id = doc_id
        self.title = title
        self.body = body
        self.text = text
        self.token_ids = token_ids
        self.metadata = metadata


def prepare_document(
    row: Mapping[str, Any],
    source_index: int,
    tokenizer: PreTrainedTokenizerBase,
    text_columns: list[str],
    title_columns: list[str],
    id_columns: list[str],
    year_column: str,
    min_year: int | None,
    min_doc_tokens: int,
) -> PreparedDocument | None:
    year = row.get(year_column)
    if min_year is not None and year is not None:
        try:
            if int(str(year)[:4]) < min_year:
                return None
        except ValueError:
            return None

    title = first_text(row, title_columns)
    body = first_text(row, text_columns)
    if not body:
        return None
    if body == title:
        text = body
    elif title:
        text = f"{title}\n\n{body}"
    else:
        text = body
    text = clean_text(text)
    body = clean_text(body)
    title = clean_text(title)
    if not text or not body:
        return None

    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) < min_doc_tokens:
        return None
    raw_id = first_text(row, id_columns) or stable_id(source_index, title, body[:256])
    doc_id = stable_id(raw_id)
    return PreparedDocument(
        doc_id=doc_id,
        title=title,
        body=body,
        text=text,
        token_ids=list(token_ids),
        metadata={
            "source_index": source_index,
            "source_doc_id": raw_id,
            "title": title,
            "year": year,
            "token_count": len(token_ids),
        },
    )


def first_text(row: Mapping[str, Any], columns: list[str]) -> str:
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


def bio_record(doc: PreparedDocument, subset: str) -> dict[str, Any]:
    if doc.title:
        user = (
            "Provide the PubMed abstract for the biomedical article with this title.\n\n"
            f"Title: {doc.title}"
        )
        assistant = doc.body
    else:
        user, assistant = split_body_for_supervision(doc.body)
    return {
        "task": "bio",
        "subset": subset,
        "example_id": doc.doc_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": doc.metadata | {"source": "pubmed_heldout", "heldout_from_cpt": True},
    }


def split_body_for_supervision(body: str) -> tuple[str, str]:
    words = body.split()
    split_index = max(1, min(len(words) // 3, len(words) - 16))
    prefix = " ".join(words[:split_index])
    suffix = " ".join(words[split_index:])
    return (
        "Continue this biomedical abstract.\n\n" + prefix,
        suffix,
    )


def compute_bio_stats(records: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> dict[str, int | float]:
    prompt_lengths = []
    assistant_lengths = []
    for record in records:
        prompt_lengths.append(len(tokenizer(record["messages"][0]["content"], add_special_tokens=False)["input_ids"]))
        assistant_lengths.append(len(tokenizer(record["messages"][1]["content"], add_special_tokens=False)["input_ids"]))
    return {
        "count": len(records),
        "avg_prompt_tokens": mean(prompt_lengths) if prompt_lengths else 0.0,
        "avg_assistant_tokens": mean(assistant_lengths) if assistant_lengths else 0.0,
        "max_prompt_tokens": max(prompt_lengths) if prompt_lengths else 0,
        "max_assistant_tokens": max(assistant_lengths) if assistant_lengths else 0,
        "min_assistant_tokens": min(assistant_lengths) if assistant_lengths else 0,
    }


def print_examples(label: str, records: list[dict[str, Any]], count: int) -> None:
    if count <= 0:
        return
    print("=" * 80)
    print(label)
    for record in records[:count]:
        user = record["messages"][0]["content"].replace("\n", " ")
        assistant = record["messages"][1]["content"].replace("\n", " ")
        print(f"- {record['example_id']}")
        print(f"  user: {user[:300]}")
        print(f"  assistant: {assistant[:300]}")


def unique_in_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def source_name(args: argparse.Namespace) -> str:
    if args.local_jsonl:
        return str(args.local_jsonl)
    if args.source_config:
        return f"{args.source_dataset}/{args.source_config}"
    return args.source_dataset


def stable_id(*parts: object) -> str:
    text = "::".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
