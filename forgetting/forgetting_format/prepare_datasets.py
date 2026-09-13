import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from common import stable_example_id, write_jsonl


MMLU_LETTERS = ["A", "B", "C", "D"]

# MMLU evaluation variants preserve questions, choices, targets, and the
# letter-only instruction. Only prompt-boundary labels differ.
MMLU_PROMPT_FORMATS = {
    "question_answer": {
        "question_label": "Question",
        "response_label": "Answer",
    },
    "problem_result": {
        "question_label": "Problem",
        "response_label": "result",
    },
}

# These two variants are the only intended training difference in the experiment.
# In particular, both retain the original GSM8K completion (including ``####``).
GSM8K_PROMPT_FORMATS = {
    "question_answer": {
        "question_label": "Question",
        "response_label": "Answer",
    },
    "problem_result": {
        "question_label": "Problem",
        "response_label": "result",
    },
}


def _load_first_available(candidates: list[tuple[str, str | None, str]], cache_dir: str | None) -> tuple[Dataset, str]:
    errors = []
    for path, name, split in candidates:
        try:
            kwargs = {"path": path, "split": split}
            if name is not None:
                kwargs["name"] = name
            if cache_dir is not None:
                kwargs["cache_dir"] = cache_dir
            return load_dataset(**kwargs), f"{path}/{name or ''}:{split}"
        except Exception as exc:
            errors.append(f"{path}/{name or ''}:{split}: {exc}")

    raise RuntimeError("Could not load any dataset candidate:\n" + "\n".join(errors))


def _limit(records: list[dict[str, Any]], max_samples: int | None) -> list[dict[str, Any]]:
    return records if max_samples is None else records[:max_samples]


def _format_mmlu_prompt(row: dict[str, Any], variant: str) -> tuple[str, str]:
    try:
        format_spec = MMLU_PROMPT_FORMATS[variant]
    except KeyError as exc:
        allowed = ", ".join(MMLU_PROMPT_FORMATS)
        raise ValueError(f"Unknown MMLU prompt variant {variant!r}; choose one of {allowed}") from exc

    choices = list(row["choices"])
    answer = row["answer"]
    if isinstance(answer, int):
        target = MMLU_LETTERS[answer]
    else:
        target = str(answer).strip().upper()

    prompt = "\n".join(
        [
            "Answer the following multiple-choice question.",
            "Give only the letter A, B, C, or D.",
            "",
            f"{format_spec['question_label']}: {row['question']}",
            f"A. {choices[0]}",
            f"B. {choices[1]}",
            f"C. {choices[2]}",
            f"D. {choices[3]}",
            "",
            f"{format_spec['response_label']}:",
        ]
    )
    return prompt, target


def prepare_mmlu(output_dir: Path, cache_dir: str | None, max_samples: int | None, variant: str) -> None:
    if variant not in MMLU_PROMPT_FORMATS:
        allowed = ", ".join(MMLU_PROMPT_FORMATS)
        raise ValueError(f"Unknown MMLU prompt variant {variant!r}; choose one of {allowed}")

    dataset, source = _load_first_available(
        [
            ("cais/mmlu", "all", "test"),
            ("cais/mmlu", "all", "validation"),
            ("lukaemon/mmlu", "all", "test"),
        ],
        cache_dir,
    )
    records = []
    for idx, row in enumerate(dataset):
        prompt, target = _format_mmlu_prompt(row, variant)
        subject = row.get("subject") or row.get("category") or "unknown"
        records.append(
            {
                "dataset_name": "mmlu",
                "format_variant": variant,
                "example_id": stable_example_id("mmlu", subject, idx),
                "prompt": prompt,
                "target": target,
                "metadata": {
                    "source": source,
                    "subject": subject,
                    "question": row["question"],
                    "choices": list(row["choices"]),
                },
            }
        )

    filename = "mmlu.jsonl" if variant == "question_answer" else f"mmlu_{variant}.jsonl"
    write_jsonl(output_dir / filename, _limit(records, max_samples))


def prepare_mmlu_auxiliary_train_sft(
    output_dir: Path,
    cache_dir: str | None,
    num_samples: int,
    seed: int,
    variant: str = "question_answer",
) -> Path:
    """Sample MMLU auxiliary_train into completion-only SFT records."""
    if variant not in MMLU_PROMPT_FORMATS:
        allowed = ", ".join(MMLU_PROMPT_FORMATS)
        raise ValueError(f"Unknown MMLU prompt variant {variant!r}; choose one of {allowed}")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    dataset, source = _load_first_available(
        [("cais/mmlu", "all", "auxiliary_train")],
        cache_dir,
    )
    if num_samples > len(dataset):
        raise ValueError(
            f"Requested {num_samples} MMLU auxiliary-train examples, but only {len(dataset)} exist."
        )

    source_indices = random.Random(seed).sample(range(len(dataset)), num_samples)
    selection_digest = hashlib.sha256(",".join(map(str, source_indices)).encode("utf-8")).hexdigest()

    records = []
    for source_row in source_indices:
        row = dataset[source_row]
        prompt, target = _format_mmlu_prompt(row, variant)
        subject = row.get("subject") or row.get("category") or "unknown"
        records.append(
            {
                "dataset_name": "mmlu_auxiliary_train",
                "format_variant": variant,
                "example_id": stable_example_id("mmlu_auxiliary_train", subject, source_row),
                "source_row": source_row,
                "prompt": prompt,
                "completion": f" {target}",
                "target": target,
                "metadata": {
                    "source": source,
                    "split": "auxiliary_train",
                    "subject": subject,
                    "question": row["question"],
                    "choices": list(row["choices"]),
                },
            }
        )

    filename = f"mmlu_auxiliary_train_{variant}_n{num_samples}_seed{seed}.jsonl"
    output_path = output_dir / filename
    write_jsonl(output_path, records)
    manifest = {
        "source": source,
        "split": "auxiliary_train",
        "num_samples": num_samples,
        "seed": seed,
        "format_variant": variant,
        "output_file": str(output_path.resolve()),
        "sample_indices_sha256": selection_digest,
    }
    with output_path.with_suffix(".manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output_path



def _load_gsm8k(
    data_path: str,
    dataset_config: str | None,
    split: str,
    cache_dir: str | None,
) -> tuple[Dataset, str]:
    """Load a Hub dataset, a saved DatasetDict, or a local JSON/JSONL file."""
    local_path = Path(data_path).expanduser()
    if local_path.exists():
        if local_path.is_file():
            dataset = load_dataset("json", data_files=str(local_path), split="train")
            return dataset, f"json:{local_path}"
        loaded = load_from_disk(str(local_path))
        if isinstance(loaded, DatasetDict):
            if split not in loaded:
                available = ", ".join(loaded.keys())
                raise KeyError(f"GSM8K split {split!r} not in {local_path}; available: {available}")
            return loaded[split], f"disk:{local_path}:{split}"
        if not isinstance(loaded, Dataset):
            raise TypeError(f"Unsupported dataset object loaded from {local_path}")
        return loaded, f"disk:{local_path}"

    kwargs: dict[str, Any] = {"path": data_path, "split": split}
    if dataset_config:
        kwargs["name"] = dataset_config
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(**kwargs), f"{data_path}/{dataset_config or ''}:{split}"


def _format_gsm8k_prompt(question: str, variant: str) -> str:
    try:
        format_spec = GSM8K_PROMPT_FORMATS[variant]
    except KeyError as exc:
        allowed = ", ".join(GSM8K_PROMPT_FORMATS)
        raise ValueError(f"Unknown GSM8K prompt variant {variant!r}; choose one of {allowed}") from exc

    # Keep both leading and internal whitespace identical across variants. This
    # makes the two labels the only prompt-template intervention.
    return "\n".join(
        [
            "",
            f"{format_spec['question_label']}: {question}",
            "",
            f"{format_spec['response_label']}:",
        ]
    )


def _completion_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["example_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["completion"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def prepare_gsm8k(
    output_dir: Path,
    cache_dir: str | None,
    max_samples: int | None,
    data_path: str,
    dataset_config: str | None,
    split: str,
    variants: list[str],
) -> None:
    """Materialize paired GSM8K SFT data with only the prompt labels changed."""
    if not variants:
        raise ValueError("At least one --gsm8k_variants value is required")
    unknown = sorted(set(variants) - set(GSM8K_PROMPT_FORMATS))
    if unknown:
        raise ValueError(f"Unknown GSM8K variant(s): {', '.join(unknown)}")

    dataset, source = _load_gsm8k(data_path, dataset_config, split, cache_dir)
    base_records = []
    for idx, row in enumerate(dataset):
        if "question" not in row or "answer" not in row:
            raise KeyError("GSM8K records must contain question and answer fields")
        base_records.append(
            {
                "example_id": stable_example_id("gsm8k", split, idx),
                "source_row": idx,
                "question": str(row["question"]),
                # A leading space gives tokenizers a clean prompt/completion boundary.
                "completion": " " + str(row["answer"]).lstrip(),
            }
        )
    base_records = _limit(base_records, max_samples)

    manifest: dict[str, Any] = {
        "dataset_name": "gsm8k",
        "source": source,
        "split": split,
        "num_examples": len(base_records),
        "mmlu_response_marker": "Answer:",
        "variants": {},
        "design": (
            "Each variant has the same source_row/example_id ordering and byte-identical "
            "GSM8K completion. Only question_label and response_label differ."
        ),
    }
    for variant in variants:
        format_spec = GSM8K_PROMPT_FORMATS[variant]
        records = []
        for base in base_records:
            records.append(
                {
                    "dataset_name": "gsm8k",
                    "format_variant": variant,
                    "example_id": base["example_id"],
                    "source_row": base["source_row"],
                    "prompt": _format_gsm8k_prompt(base["question"], variant),
                    "completion": base["completion"],
                    "metadata": {
                        "source": source,
                        "question": base["question"],
                        "question_label": format_spec["question_label"],
                        "response_label": format_spec["response_label"],
                        "mmlu_marker_exact_match": format_spec["response_label"] == "Answer",
                    },
                }
            )
        output_path = output_dir / f"gsm8k_{variant}.jsonl"
        write_jsonl(output_path, records)
        manifest["variants"][variant] = {
            "path": str(output_path),
            **format_spec,
            "mmlu_marker_exact_match": format_spec["response_label"] == "Answer",
            "completion_sha256": _completion_hash(records),
        }

    with (output_dir / "gsm8k_format_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def prepare_ifeval(output_dir: Path, cache_dir: str | None, max_samples: int | None) -> None:
    dataset, source = _load_first_available(
        [
            ("google/IFEval", None, "train"),
            ("HuggingFaceH4/ifeval", None, "train"),
        ],
        cache_dir,
    )
    records = []
    for idx, row in enumerate(dataset):
        prompt = row.get("prompt") or row.get("input") or row.get("instruction")
        records.append(
            {
                "dataset_name": "ifeval",
                "example_id": stable_example_id("ifeval", idx),
                "prompt": prompt,
                "target": "",
                "metadata": {
                    "source": source,
                    "instruction_id_list": row.get("instruction_id_list"),
                    "kwargs": row.get("kwargs"),
                },
            }
        )

    write_jsonl(output_dir / "ifeval.jsonl", _limit(records, max_samples))


def _dolly_prompt(row: dict[str, Any], task: str) -> str:
    parts = []
    if task == "classification":
        parts.append("Answer the following classification task.")
        parts.append("Return only the requested answer, not the task category.")
    else:
        parts.append("Answer the question using the provided context when it is useful.")

    parts.extend(["", f"Instruction: {row['instruction']}"])
    if row.get("context"):
        parts.extend(["", f"Context: {row['context']}"])
    parts.extend(["", "Answer:"])
    return "\n".join(parts)


def prepare_dolly(output_dir: Path, cache_dir: str | None, max_samples: int | None) -> None:
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train", cache_dir=cache_dir)
    by_category = defaultdict(list)
    for row in dataset:
        by_category[row.get("category", "")].append(row)

    classification_records = []
    for idx, row in enumerate(by_category["classification"]):
        classification_records.append(
            {
                "dataset_name": "dolly_classification",
                "example_id": stable_example_id("dolly_classification", idx),
                "prompt": _dolly_prompt(row, "classification"),
                "target": row["response"],
                "metadata": {
                    "source": "databricks/databricks-dolly-15k:train",
                    "category": row.get("category"),
                    "instruction": row.get("instruction"),
                    "context": row.get("context"),
                },
            }
        )

    closed_qa_records = []
    for idx, row in enumerate(by_category["closed_qa"]):
        closed_qa_records.append(
            {
                "dataset_name": "dolly_closed_qa",
                "example_id": stable_example_id("dolly_closed_qa", idx),
                "prompt": _dolly_prompt(row, "closed_qa"),
                "target": row["response"],
                "metadata": {
                    "source": "databricks/databricks-dolly-15k:train",
                    "category": row.get("category"),
                    "instruction": row.get("instruction"),
                    "context": row.get("context"),
                },
            }
        )

    write_jsonl(output_dir / "dolly_classification.jsonl", _limit(classification_records, max_samples))
    write_jsonl(output_dir / "dolly_closed_qa.jsonl", _limit(closed_qa_records, max_samples))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized eval datasets for EAFT comparison.")
    parser.add_argument("--output_dir", default="eval_data")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--datasets",
        default="gsm8k,mmlu,ifeval,dolly_classification,dolly_closed_qa",
        help="Comma-separated subset of gsm8k,mmlu,mmlu_sft,ifeval,dolly_classification,dolly_closed_qa.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gsm8k_data_path", default="openai/gsm8k")
    parser.add_argument("--gsm8k_config", default="main")
    parser.add_argument("--gsm8k_split", default="train")
    parser.add_argument(
        "--gsm8k_variants",
        default="question_answer,problem_result",
        help="Comma-separated prompt variants to materialize.",
    )
    parser.add_argument(
        "--mmlu_variants",
        default="question_answer,problem_result",
        help="Comma-separated MMLU evaluation prompt variants to materialize.",
    )
    parser.add_argument(
        "--mmlu_sft_samples",
        type=int,
        default=0,
        help="Number of random MMLU auxiliary_train examples to materialize for SFT.",
    )
    parser.add_argument("--mmlu_sft_seed", type=int, default=42)
    parser.add_argument(
        "--mmlu_sft_variant", choices=sorted(MMLU_PROMPT_FORMATS), default="question_answer"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {name.strip() for name in args.datasets.split(",") if name.strip()}

    if "gsm8k" in selected:
        variants = [value.strip() for value in args.gsm8k_variants.split(",") if value.strip()]
        prepare_gsm8k(
            output_dir=output_dir,
            cache_dir=args.cache_dir,
            max_samples=args.max_samples,
            data_path=args.gsm8k_data_path,
            dataset_config=args.gsm8k_config or None,
            split=args.gsm8k_split,
            variants=variants,
        )
    if "mmlu" in selected:
        mmlu_variants = [value.strip() for value in args.mmlu_variants.split(",") if value.strip()]
        for variant in mmlu_variants:
            prepare_mmlu(output_dir, args.cache_dir, args.max_samples, variant)
    if "mmlu_sft" in selected:
        if args.mmlu_sft_samples <= 0:
            raise ValueError("--mmlu_sft_samples must be positive when --datasets includes mmlu_sft")
        sample_path = prepare_mmlu_auxiliary_train_sft(
            output_dir, args.cache_dir, args.mmlu_sft_samples, args.mmlu_sft_seed, args.mmlu_sft_variant
        )
        print(f"Wrote MMLU auxiliary-train SFT data to {sample_path}")
    if "ifeval" in selected:
        prepare_ifeval(output_dir, args.cache_dir, args.max_samples)
    if {"dolly_classification", "dolly_closed_qa"} & selected:
        prepare_dolly(output_dir, args.cache_dir, args.max_samples)


if __name__ == "__main__":
    main()
