#!/usr/bin/env python
"""Translate a no-memory baseline’s GSM8K mistakes for memory retrieval."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TypeVar

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LANGUAGES = {
    "zh": {"name": "Chinese", "code": "zho_Hans"},
    "fr": {"name": "French", "code": "fra_Latn"},
    "ko": {"name": "Korean", "code": "kor_Hang"},
    "es": {"name": "Spanish", "code": "spa_Latn"},
}
FIELDS = ("question", "answer")
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate GSM8K baseline mistakes into zh/fr/ko/es for memory experiments."
    )
    parser.add_argument(
        "--source_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reuse_translation_path",
        type=Path,
        action="append",
        default=None,
        help=(
            "Existing translation JSON to reuse. Repeat the option to merge multiple files; "
            "earlier files take precedence for duplicate records/languages."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=tuple(LANGUAGES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--source_max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def batched(items: Sequence[T], batch_size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_model_and_tokenizer(args: argparse.Namespace, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        for name in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, name):
                setattr(model.generation_config, name, None)
    return tokenizer, model


def build_record_prompt(tokenizer, record: Dict[str, str], target_language: str) -> str:
    payload = json.dumps({field: record[field] for field in FIELDS}, ensure_ascii=False, indent=2)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise translation engine for GSM8K math word problems. "
                "Translate text only. Preserve all numbers, arithmetic expressions, "
                "<<...>> calculation markers, and #### final-answer markers exactly. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Translate every JSON value independently into {target_language}. "
                "Keep exactly the same keys: question, answer. Do not solve the problem, "
                "do not change the math, and do not add explanations. Return only JSON.\n"
                f"{payload}"
            ),
        },
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        f"Translate every JSON value independently into {target_language}. "
        "Keep exactly the same keys and return only JSON.\n"
        f"{payload}\nJSON:"
    )


def build_text_prompt(tokenizer, text: str, target_language: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise translation engine for GSM8K math word problems. "
                "Translate text only and preserve all numbers, equations, <<...>>, and #### markers."
            ),
        },
        {
            "role": "user",
            "content": f"Translate only this text into {target_language}. Return only the translation.\n{text}",
        },
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Translate this text into {target_language}. Return only the translation.\n{text}\nTranslation:"


def generate_outputs(
    prompts: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    source_max_length: int,
    max_new_tokens: int,
    description: str,
) -> List[str]:
    outputs_text: List[str] = []
    for batch in tqdm(list(batched(prompts, batch_size)), desc=description):
        inputs = tokenizer(
            list(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=source_max_length,
        ).to(device)
        input_length = inputs.input_ids.shape[1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        outputs_text.extend(tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True))
        del inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs_text


def parse_json_translation(text: str) -> Dict[str, str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        raise ValueError("No JSON object found")
    parsed = json.loads(match.group(0))
    return {field: str(parsed[field]).strip() for field in FIELDS}


def translate_records(
    records: List[Dict[str, str]],
    tokenizer,
    model,
    device: str,
    target_language: str,
    batch_size: int,
    source_max_length: int,
    max_new_tokens: int,
) -> List[Dict[str, str]]:
    prompts = [build_record_prompt(tokenizer, record, target_language) for record in records]
    decoded = generate_outputs(
        prompts=prompts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        source_max_length=source_max_length,
        max_new_tokens=max_new_tokens,
        description=f"Translating records to {target_language}",
    )

    translated_records: List[Dict[str, str]] = []
    for record, output in zip(records, decoded):
        try:
            translated_records.append(parse_json_translation(output))
        except Exception:
            field_prompts = [build_text_prompt(tokenizer, record[field], target_language) for field in FIELDS]
            field_outputs = generate_outputs(
                prompts=field_prompts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=min(batch_size, len(FIELDS)),
                source_max_length=source_max_length,
                max_new_tokens=max_new_tokens,
                description=f"Fallback translating fields to {target_language}",
            )
            translated_records.append({field: text.strip().strip('"') for field, text in zip(FIELDS, field_outputs)})
    return translated_records


def load_reusable_translations(path: Path) -> Dict[int, Dict[str, Dict[str, str]]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(record["index"]): record.get("translations", {})
        for record in data.get("records", [])
        if "index" in record
    }


def write_output(path: Path, output: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_path}")

    source = json.loads(args.source_path.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_path.read_text(encoding="utf-8"))
    wrong_indices = set(int(index) for index in baseline["summary"]["wrong_indices"])
    reuse_paths = args.reuse_translation_path or []
    reusable: Dict[int, Dict[str, Dict[str, str]]] = {}
    for reuse_path in reuse_paths:
        for index, translations in load_reusable_translations(reuse_path).items():
            merged = reusable.setdefault(index, {})
            for language, fields in translations.items():
                merged.setdefault(language, fields)

    selected = [record for record in source["records"] if int(record["index"]) in wrong_indices]
    output_records: List[Dict[str, object]] = []
    missing_by_language = {language: [] for language in args.languages}
    missing_positions = {language: [] for language in args.languages}

    for record in selected:
        index = int(record["index"])
        translations = {"en": dict(record["source"])}
        for language in args.languages:
            reused = reusable.get(index, {}).get(language)
            if reused and all(field in reused for field in FIELDS):
                translations[language] = {field: str(reused[field]) for field in FIELDS}
            else:
                missing_by_language[language].append({field: str(record["source"][field]) for field in FIELDS})
                missing_positions[language].append(len(output_records))
        output_records.append(
            {
                "index": index,
                "source": {field: str(record["source"][field]) for field in FIELDS},
                "translations": translations,
            }
        )

    output: Dict[str, object] = {
        "source_dataset": source.get("source_dataset", "openai/gsm8k"),
        "subset": "main",
        "split": source.get("split", "train"),
        "source_path": str(args.source_path),
        "baseline_path": str(args.baseline_path),
        "source_filter": (
            f"{baseline.get('summary', {}).get('model_name', 'baseline model')} "
            "greedy incorrect on GSM8K train first200"
        ),
        "translation_backend": "causal_local_lm",
        "translation_model": args.model_name,
        "reuse_translation_paths": [str(path) for path in reuse_paths if path.exists()],
        "source_language": "en",
        "record_level": True,
        "fields": list(FIELDS),
        "languages": {
            "en": {"name": "English", "code": "eng_Latn"},
            **{language: LANGUAGES[language] for language in args.languages},
        },
        "num_source_records": len(output_records),
        "wrong_indices": [int(record["index"]) for record in output_records],
        "records": output_records,
    }
    write_output(args.output_path, output)

    total_missing = sum(len(items) for items in missing_by_language.values())
    print(f"wrong examples: {len(output_records)}")
    print(
        "reused translations from "
        f"{len([path for path in reuse_paths if path.exists()])} files: "
        f"{len(output_records) * len(args.languages) - total_missing}"
    )
    print(f"translations to generate: {total_missing}")
    if total_missing == 0:
        print(f"wrote {args.output_path}")
        return

    device = resolve_device(args.device)
    tokenizer, model = load_model_and_tokenizer(args, device)
    for language in args.languages:
        records_to_translate = missing_by_language[language]
        if not records_to_translate:
            continue
        translated = translate_records(
            records=records_to_translate,
            tokenizer=tokenizer,
            model=model,
            device=device,
            target_language=LANGUAGES[language]["name"],
            batch_size=args.batch_size,
            source_max_length=args.source_max_length,
            max_new_tokens=args.max_new_tokens,
        )
        for position, translated_record in zip(missing_positions[language], translated):
            output_records[position]["translations"][language] = translated_record
        write_output(args.output_path, output)
        print(f"updated {language}: {len(translated)} records")

    write_output(args.output_path, output)
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
