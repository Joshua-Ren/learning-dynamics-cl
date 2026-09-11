from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

from transformers import AutoTokenizer


TASKS = ("gsm8k", "mbpp", "dolly_qa")
SPLITS = ("train", "probe")
DEFAULT_INPUT_DIR = "data/prepared_subsets"
DEFAULT_OUTPUT_DIR = "data/multilingual_subsets"
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare multilingual fixed subsets: GSM8K zh, MBPP en, Dolly QA fr."
    )
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument(
        "--translator",
        choices=("openai", "passthrough"),
        default="passthrough",
        help="Use passthrough only for smoke tests. Use openai for real translation.",
    )
    parser.add_argument("--openai_model", default=os.environ.get("OPENAI_TRANSLATION_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--openai_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--openai_base_url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--sample_examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--tokenizer_name", default=DEFAULT_TOKENIZER)
    parser.add_argument("--skip_token_stats", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output split files before processing. Default resumes and skips existing IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tasks = parse_csv(args.tasks)
    splits = parse_csv(args.splits)
    translator = build_translator(args)

    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "tasks": tasks,
        "splits": splits,
        "translator": args.translator,
        "openai_model": args.openai_model if args.translator == "openai" else None,
        "max_examples": args.max_examples,
        "outputs": {},
    }

    for task in tasks:
        report["outputs"][task] = {}
        for split in splits:
            input_path = input_dir / task / f"{split}.jsonl"
            output_path = output_dir / task / f"{split}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite and output_path.exists():
                output_path.unlink()
            split_report = process_split(
                input_path=input_path,
                output_path=output_path,
                task=task,
                split=split,
                translator=translator,
                max_examples=args.max_examples,
                sleep_seconds=args.sleep_seconds,
            )
            report["outputs"][task][split] = split_report

    validation = validate_outputs(input_dir, output_dir, tasks, splits, args.max_examples)
    report["validation"] = validation
    report["samples"] = sample_examples(output_dir, tasks, splits, args.sample_examples, args.seed)
    if not args.skip_token_stats:
        report["token_length_stats"] = compute_token_length_stats(output_dir, tasks, splits, args.tokenizer_name)

    write_json(output_dir / "manifest.json", report)
    write_samples_markdown(output_dir / "sample_translations.md", report["samples"])
    print(json.dumps(compact_report(report), indent=2, ensure_ascii=False, sort_keys=True))
    print(f"Wrote multilingual manifest: {output_dir / 'manifest.json'}")
    print(f"Wrote sample translations: {output_dir / 'sample_translations.md'}")


def process_split(
    input_path: Path,
    output_path: Path,
    task: str,
    split: str,
    translator: Callable[[str, str], str],
    max_examples: int | None,
    sleep_seconds: float,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    existing_ids = load_existing_ids(output_path)
    input_count = count_jsonl(input_path)
    target_count = min(input_count, max_examples) if max_examples is not None else input_count
    written = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as source, output_path.open("a", encoding="utf-8") as sink:
        for index, record in enumerate(iter_jsonl(source)):
            if max_examples is not None and index >= max_examples:
                break
            example_id = str(record["example_id"])
            if example_id in existing_ids:
                skipped += 1
                continue
            translated = translate_record(record, task, translator)
            sink.write(json.dumps(translated, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            written += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_count": input_count,
        "target_count": target_count,
        "existing_skipped": skipped,
        "new_written": written,
        "output_count": count_jsonl(output_path),
    }


def translate_record(
    record: dict[str, Any],
    task: str,
    translator: Callable[[str, str], str],
) -> dict[str, Any]:
    if task == "mbpp":
        target_language = "English"
        translated_messages = [dict(message) for message in record["messages"]]
        translation_status = "unchanged"
    elif task == "gsm8k":
        target_language = "Chinese"
        translated_messages = translate_messages(record["messages"], target_language, translator, preserve_gsm8k=True)
        translation_status = "translated"
    elif task == "dolly_qa":
        target_language = "French"
        translated_messages = translate_messages(record["messages"], target_language, translator, preserve_gsm8k=False)
        translation_status = "translated"
    else:
        raise ValueError(f"Unsupported task: {task}")

    metadata = dict(record.get("metadata") or {})
    metadata["multilingual"] = {
        "source_task": record.get("task"),
        "target_task": task,
        "target_language": target_language,
        "translation_status": translation_status,
    }
    return {
        **record,
        "messages": translated_messages,
        "original_messages": record["messages"],
        "metadata": metadata,
        "task": task,
        "language": target_language,
    }


def translate_messages(
    messages: list[dict[str, str]],
    target_language: str,
    translator: Callable[[str, str], str],
    preserve_gsm8k: bool,
) -> list[dict[str, str]]:
    translated = []
    for message in messages:
        content = str(message["content"])
        protected, placeholders = protect_structured_spans(content, preserve_gsm8k=preserve_gsm8k)
        translated_content = translator(protected, target_language)
        translated_content = restore_structured_spans(translated_content, placeholders)
        translated.append({"role": str(message["role"]), "content": translated_content})
    return translated


def protect_structured_spans(text: str, preserve_gsm8k: bool) -> tuple[str, dict[str, str]]:
    spans: list[tuple[int, int]] = []
    patterns = [
        r"https?://\S+",
        r"`[^`]*`",
        r"```.*?```",
    ]
    if preserve_gsm8k:
        patterns.extend(
            [
                r"<<[^>]+>>",
                r"####\s*[-+]?\d[\d,]*(?:\.\d+)?",
                r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*[-+*/=]\s*[-+]?\d[\d,]*(?:\.\d+)?)+",
                r"(?:[$€£]\s*)?[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)",
            ]
        )
    else:
        patterns.append(r"[-+]?\d[\d,]*(?:\.\d+)?")

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            spans.append((match.start(), match.end()))
    spans = merge_spans(spans)

    placeholders = {}
    pieces = []
    last = 0
    for index, (start, end) in enumerate(spans):
        placeholder = f"[[KEEP_{index:04d}]]"
        pieces.append(text[last:start])
        pieces.append(placeholder)
        placeholders[placeholder] = text[start:end]
        last = end
    pieces.append(text[last:])
    return "".join(pieces), placeholders


def restore_structured_spans(text: str, placeholders: dict[str, str]) -> str:
    for placeholder, value in placeholders.items():
        text = text.replace(placeholder, value)
    return text


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def build_translator(args: argparse.Namespace) -> Callable[[str, str], str]:
    if args.translator == "passthrough":
        return lambda text, _target_language: text
    if args.translator == "openai":
        api_key = os.environ.get(args.openai_api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {args.openai_api_key_env} for --translator openai")

        def translate(text: str, target_language: str) -> str:
            return openai_translate(
                text=text,
                target_language=target_language,
                api_key=api_key,
                base_url=args.openai_base_url,
                model=args.openai_model,
                max_retries=args.max_retries,
            )

        return translate
    raise ValueError(f"Unsupported translator: {args.translator}")


def openai_translate(
    text: str,
    target_language: str,
    api_key: str,
    base_url: str,
    model: str,
    max_retries: int,
) -> str:
    system = (
        "You are a careful translation engine for ML datasets. Translate only natural-language text. "
        "Do not change placeholders like [[KEEP_0000]], code formatting, equations, numbers, URLs, or final answer markers. "
        "Return only the translated text, with no explanation."
    )
    user = f"Translate this text to {target_language}:\n\n{text}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    for attempt in range(max_retries):
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
                return str(body["choices"][0]["message"]["content"]).strip()
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == max_retries - 1:
                detail = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"OpenAI translation failed with HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            if attempt == max_retries - 1:
                raise RuntimeError(f"OpenAI translation failed: {error}") from error
        time.sleep(min(2**attempt, 30))
    raise RuntimeError("OpenAI translation failed after retries")


def validate_outputs(
    input_dir: Path,
    output_dir: Path,
    tasks: list[str],
    splits: list[str],
    max_examples: int | None,
) -> dict[str, Any]:
    report = {}
    for task in tasks:
        report[task] = {}
        for split in splits:
            input_path = input_dir / task / f"{split}.jsonl"
            output_path = output_dir / task / f"{split}.jsonl"
            source = list(iter_jsonl_path(input_path))
            output = list(iter_jsonl_path(output_path))
            expected = min(len(source), max_examples) if max_examples is not None else len(source)
            errors = []
            if len(output) != expected:
                errors.append(f"count mismatch: expected {expected}, got {len(output)}")
            source_by_id = {str(record["example_id"]): record for record in source[:expected]}
            for record in output:
                errors.extend(validate_record(record, source_by_id.get(str(record["example_id"])), task))
            if errors:
                raise RuntimeError(f"Validation failed for {task}/{split}: {errors[:10]}")
            report[task][split] = {
                "input_count": len(source),
                "expected_output_count": expected,
                "output_count": len(output),
                "valid": True,
            }
    return report


def validate_record(record: dict[str, Any], source: dict[str, Any] | None, task: str) -> list[str]:
    errors = []
    if source is None:
        return [f"missing source for example_id={record.get('example_id')}"]
    if record.get("example_id") != source.get("example_id"):
        errors.append("example_id changed")
    if len(record.get("messages", [])) != len(source.get("messages", [])):
        errors.append("message count changed")
        return errors
    for index, message in enumerate(record["messages"]):
        if not str(message.get("content", "")).strip():
            errors.append(f"empty translated message {index}")
        if message.get("role") != source["messages"][index].get("role"):
            errors.append(f"role changed at message {index}")
    if task == "mbpp" and record["messages"] != source["messages"]:
        errors.append("MBPP messages changed")
    if task == "gsm8k":
        for index, message in enumerate(record["messages"]):
            source_content = source["messages"][index]["content"]
            target_content = message["content"]
            if canonical_numbers(source_content) != canonical_numbers(target_content):
                errors.append(f"GSM8K numbers changed at message {index}")
            if re.findall(r"<<[^>]+>>", source_content) != re.findall(r"<<[^>]+>>", target_content):
                errors.append(f"GSM8K <<...>> annotations changed at message {index}")
            if extract_final_answer(source_content) != extract_final_answer(target_content):
                errors.append(f"GSM8K final #### answer changed at message {index}")
    return errors


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)", text)


def canonical_numbers(text: str) -> list[str]:
    return [canonical_number(value) for value in extract_numbers(text)]


def canonical_number(value: str) -> str:
    normalized = value.replace(",", "")
    if normalized.startswith("."):
        normalized = "0" + normalized
    if normalized.startswith("-."):
        normalized = "-0" + normalized[1:]
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def extract_final_answer(text: str) -> str | None:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    return match.group(1) if match else None


def sample_examples(
    output_dir: Path,
    tasks: list[str],
    splits: list[str],
    sample_count: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    samples = {}
    for task in tasks:
        if task not in {"gsm8k", "dolly_qa"}:
            continue
        pool = []
        for split in splits:
            path = output_dir / task / f"{split}.jsonl"
            if path.exists():
                for record in iter_jsonl_path(path):
                    pool.append({"split": split, "record": record})
        rng.shuffle(pool)
        samples[task] = pool[:sample_count]
    return samples


def compute_token_length_stats(
    output_dir: Path,
    tasks: list[str],
    splits: list[str],
    tokenizer_name: str,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    report = {}
    for task in tasks:
        report[task] = {}
        for split in splits:
            lengths = []
            for record in iter_jsonl_path(output_dir / task / f"{split}.jsonl"):
                text = tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=False)
                lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
            report[task][split] = summarize_lengths(lengths)
    return report


def summarize_lengths(lengths: list[int]) -> dict[str, float | int | None]:
    if not lengths:
        return {"count": 0, "mean": None, "median": None, "max": None, "p95": None}
    ordered = sorted(lengths)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return {
        "count": len(lengths),
        "mean": mean(lengths),
        "median": median(lengths),
        "max": max(lengths),
        "p95": ordered[p95_index],
    }


def write_samples_markdown(path: Path, samples: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Multilingual Translation Samples", ""]
    for task, rows in samples.items():
        lines.extend([f"## {task}", ""])
        for row in rows:
            record = row["record"]
            lines.extend(
                [
                    f"### {row['split']} / {record['example_id']}",
                    "",
                    "**Original user:**",
                    "",
                    record["original_messages"][0]["content"],
                    "",
                    "**Translated user:**",
                    "",
                    record["messages"][0]["content"],
                    "",
                    "**Original assistant:**",
                    "",
                    record["original_messages"][1]["content"],
                    "",
                    "**Translated assistant:**",
                    "",
                    record["messages"][1]["content"],
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_dir": report["output_dir"],
        "translator": report["translator"],
        "outputs": report["outputs"],
        "validation": report["validation"],
        "token_length_stats": report.get("token_length_stats"),
    }


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(record["example_id"]) for record in iter_jsonl_path(path)}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl_path(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        yield from iter_jsonl(handle)


def iter_jsonl(handle: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in handle:
        line = line.strip()
        if line:
            yield json.loads(line)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
