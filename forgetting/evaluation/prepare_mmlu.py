from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from .common import write_jsonl
from .prompts import format_mmlu_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the MMLU test split used by the forgetting experiments.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Debug only; omit for the paper setting.")
    args = parser.parse_args()

    dataset = load_dataset("cais/mmlu", "all", split="test", cache_dir=args.cache_dir)
    limit = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    rows = []
    for index in range(limit):
        raw = dict(dataset[index])
        prompt, target = format_mmlu_prompt(raw)
        subject = str(raw.get("subject") or raw.get("category") or "unknown")
        rows.append(
            {
                "dataset_name": "mmlu",
                "example_id": f"mmlu:{subject}/{index}",
                "prompt": prompt,
                "target": target,
                "metadata": {
                    "subject": subject,
                    "source_index": index,
                    "question": raw["question"],
                    "choices": list(raw["choices"]),
                },
            }
        )
    write_jsonl(Path(args.output_file), rows)


if __name__ == "__main__":
    main()
