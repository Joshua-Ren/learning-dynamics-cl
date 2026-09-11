from collections import defaultdict
from pathlib import Path
import sys

if __name__ == "__main__" and __package__ is None:
    repo_root = str(Path(__file__).resolve().parents[1])
    script_dir = str(Path(__file__).resolve().parent)
    sys.path = [repo_root] + [path for path in sys.path if path != script_dir]

import torch
from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer

from utils.gsm8k_tokenizer import GSM8KTokenizerProcessor


DEFAULT_MMLU_SUBJECTS = [
    "abstract_algebra",
    "college_mathematics",
    "high_school_physics",
    "computer_security",
    "international_law",
    "moral_disputes",
    "high_school_us_history",
    "clinical_knowledge",
]


def parse_csv_arg(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_hf_dataset(name, config_name, split, local_files_only):
    download_config = DownloadConfig(local_files_only=local_files_only)
    if config_name:
        return load_dataset(
            name,
            config_name,
            split=split,
            download_config=download_config,
        )
    return load_dataset(
        name,
        split=split,
        download_config=download_config,
    )


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _to_tensor(values):
    return torch.tensor(values, dtype=torch.long)


def _pad_to_max_length(input_ids, attention_mask, labels, max_length, pad_token_id):
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return _to_tensor(input_ids), _to_tensor(attention_mask), _to_tensor(labels)


def build_supervised_sample(
    tokenizer,
    prompt_text,
    target_text,
    max_length,
    metadata=None,
):
    full_text = prompt_text + target_text
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    attention_mask = [1] * len(full_ids)
    labels = [-100] * len(full_ids)
    target_start = len(prompt_ids)
    target_end = len(full_ids)
    for pos in range(target_start, target_end):
        if pos < len(labels):
            labels[pos] = full_ids[pos]

    input_ids, attention_mask, labels = _pad_to_max_length(
        input_ids=full_ids,
        attention_mask=attention_mask,
        labels=labels,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
    )

    sample = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_ids_rl": _to_tensor(prompt_ids[:max_length]),
        "attention_mask_rl": _to_tensor([1] * min(len(prompt_ids), max_length)),
        "prompt_text": prompt_text,
        "target_text": target_text,
    }
    if metadata:
        sample.update(metadata)
    return sample


class GSM8KHHAdapter:
    name = "gsm8k"

    def __init__(self, model_name_or_path, max_length, config, local_files_only):
        self.config = config
        self.processor = GSM8KTokenizerProcessor(
            model_name_or_path=model_name_or_path,
            max_length=max_length,
            local_files_only=local_files_only,
        )
        self.tokenizer = self.processor.tokenizer

    def load_dataset(self, split, local_files_only):
        dataset_config = self.config.get("dataset", {})
        return _load_hf_dataset(
            name=dataset_config.get("name", "openai/gsm8k"),
            config_name=dataset_config.get("config", "main"),
            split=split,
            local_files_only=local_files_only,
        )

    def process_dataset(self, dataset):
        processed = []
        for raw_row in self.iter_raw_examples(dataset):
            processed.append(self.process_raw_example(raw_row))
        return processed

    def iter_raw_examples(self, dataset):
        for source_idx, example in enumerate(dataset):
            yield {
                "raw_index": source_idx,
                "example_id": example.get("id", source_idx),
                "dataset": "gsm8k",
                "domain": "gsm8k",
                "example": example,
            }

    def process_raw_example(self, raw_row):
        item = self.processor.process_single_example(raw_row["example"])
        item["example_id"] = raw_row["example_id"]
        item["dataset"] = raw_row["dataset"]
        item["domain"] = raw_row["domain"]
        item["raw_index"] = raw_row["raw_index"]
        return item


class MMLUHHAdapter:
    name = "mmlu"

    def __init__(
        self,
        model_name_or_path,
        max_length,
        config,
        local_files_only,
        subjects=None,
    ):
        self.config = config
        self.max_length = max_length
        self.subjects = subjects or DEFAULT_MMLU_SUBJECTS
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="right",
            truncation_side="right",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        _ensure_pad_token(self.tokenizer)

    @staticmethod
    def format_prompt(question, choices):
        return (
            "The following is a multiple choice question.\n\n"
            f"Question: {question}\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n\n"
            "Answer:"
        )

    def load_dataset(self, split, local_files_only):
        dataset_config = self.config.get("hh_datasets", {}).get("mmlu", {})
        dataset_name = dataset_config.get("name", "cais/mmlu")
        by_subject = {}
        for subject in self.subjects:
            by_subject[subject] = _load_hf_dataset(
                name=dataset_name,
                config_name=subject,
                split=split,
                local_files_only=local_files_only,
            )
        return by_subject

    def process_dataset(self, dataset):
        processed = []
        for raw_row in self.iter_raw_examples(dataset):
            processed.append(self.process_raw_example(raw_row))
        return processed

    def iter_raw_examples(self, dataset):
        answer_letters = ["A", "B", "C", "D"]
        max_len = max((len(subject_dataset) for subject_dataset in dataset.values()), default=0)
        for source_idx in range(max_len):
            for subject in self.subjects:
                subject_dataset = dataset.get(subject)
                if subject_dataset is None or source_idx >= len(subject_dataset):
                    continue
                example = subject_dataset[source_idx]
                answer_idx = int(example["answer"])
                yield {
                    "raw_index": source_idx,
                    "example_id": f"{subject}_{source_idx}",
                    "dataset": "mmlu",
                    "domain": subject,
                    "answer_idx": answer_idx,
                    "answer_letter": answer_letters[answer_idx],
                    "example": example,
                }

    def process_raw_example(self, raw_row):
        example = raw_row["example"]
        prompt_text = self.format_prompt(example["question"], example["choices"])
        target_text = " " + raw_row["answer_letter"]
        return build_supervised_sample(
            tokenizer=self.tokenizer,
            prompt_text=prompt_text,
            target_text=target_text,
            max_length=self.max_length,
            metadata={
                "example_id": raw_row["example_id"],
                "dataset": raw_row["dataset"],
                "domain": raw_row["domain"],
                "raw_index": raw_row["raw_index"],
                "answer_idx": raw_row["answer_idx"],
                "answer_letter": raw_row["answer_letter"],
            },
        )


class DollyHHAdapter:
    name = "dolly"

    def __init__(self, model_name_or_path, max_length, config, local_files_only):
        self.config = config
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="right",
            truncation_side="right",
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        _ensure_pad_token(self.tokenizer)

    @staticmethod
    def format_prompt(instruction, context):
        instruction = (instruction or "").strip()
        context = (context or "").strip()
        if context:
            return f"Instruction:\n{instruction}\n\nInput:\n{context}\n\nResponse:\n"
        return f"Instruction:\n{instruction}\n\nResponse:\n"

    def load_dataset(self, split, local_files_only):
        dataset_config = self.config.get("hh_datasets", {}).get("dolly", {})
        return _load_hf_dataset(
            name=dataset_config.get("name", "databricks/databricks-dolly-15k"),
            config_name=dataset_config.get("config"),
            split=split,
            local_files_only=local_files_only,
        )

    def process_dataset(self, dataset):
        processed = []
        for raw_row in self.iter_raw_examples(dataset):
            processed.append(self.process_raw_example(raw_row))
        return processed

    def iter_raw_examples(self, dataset):
        for source_idx, example in enumerate(dataset):
            target_text = (example.get("response") or "").strip()
            if not target_text:
                continue
            yield {
                "raw_index": source_idx,
                "example_id": example.get("id", source_idx),
                "dataset": "dolly",
                "domain": example.get("category"),
                "example": example,
            }

    def process_raw_example(self, raw_row):
        example = raw_row["example"]
        target_text = (example.get("response") or "").strip()
        if not target_text:
            raise ValueError(f"Selected Dolly example has empty response: {raw_row['example_id']}")
        prompt_text = self.format_prompt(
            instruction=example.get("instruction"),
            context=example.get("context"),
        )
        return build_supervised_sample(
            tokenizer=self.tokenizer,
            prompt_text=prompt_text,
            target_text=target_text,
            max_length=self.max_length,
            metadata={
                "example_id": raw_row["example_id"],
                "dataset": raw_row["dataset"],
                "domain": raw_row["domain"],
                "raw_index": raw_row["raw_index"],
            },
        )


def make_hh_dataset_adapter(
    adapter_name,
    model_name_or_path,
    max_length,
    config,
    local_files_only,
    mmlu_subjects=None,
):
    adapter_name = adapter_name.lower()
    if adapter_name == "gsm8k":
        return GSM8KHHAdapter(model_name_or_path, max_length, config, local_files_only)
    if adapter_name == "mmlu":
        return MMLUHHAdapter(
            model_name_or_path,
            max_length,
            config,
            local_files_only,
            subjects=mmlu_subjects,
        )
    if adapter_name == "dolly":
        return DollyHHAdapter(model_name_or_path, max_length, config, local_files_only)
    raise ValueError(f"Unknown HH dataset adapter: {adapter_name}")


def _dry_test():
    class FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) % 251 + 1 for ch in text]

    tokenizer = FakeTokenizer()
    sample = build_supervised_sample(
        tokenizer=tokenizer,
        prompt_text="Question: 1+1?\nAnswer:",
        target_text=" 2",
        max_length=32,
        metadata={"example_id": "toy", "domain": "dry"},
    )
    valid = torch.where(sample["labels"] != -100)[0].tolist()
    prompt_len = len(tokenizer.encode(sample["prompt_text"], add_special_tokens=False))
    target_len = len(tokenizer.encode(sample["target_text"], add_special_tokens=False))
    assert valid == list(range(prompt_len, prompt_len + target_len))
    assert sample["labels"][valid[0]].item() == sample["input_ids"][valid[0]].item()
    assert sample["attention_mask"].sum().item() == len(sample["prompt_text"] + sample["target_text"])

    mmlu = MMLUHHAdapter.__new__(MMLUHHAdapter)
    mmlu.tokenizer = tokenizer
    mmlu.max_length = 128
    mmlu_sample = build_supervised_sample(
        tokenizer=tokenizer,
        prompt_text=MMLUHHAdapter.format_prompt("Q?", ["a", "b", "c", "d"]),
        target_text=" A",
        max_length=mmlu.max_length,
        metadata={"example_id": "mmlu_toy", "domain": "abstract_algebra"},
    )
    assert (mmlu_sample["labels"] != -100).sum().item() == 2

    grouped = defaultdict(int)
    grouped[sample["domain"]] += 1
    grouped[mmlu_sample["domain"]] += 1
    assert grouped["dry"] == 1
    assert grouped["abstract_algebra"] == 1

    print("hh_dataset_adapters dry test passed")


if __name__ == "__main__":
    _dry_test()
