from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Any

import torch

from .model_setup import ensure_repo_imports


@dataclass
class SupervisedSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_text: str
    response_text: str
    metadata: dict[str, Any]

    def as_model_sample(self) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }


@dataclass
class MMLUObservation:
    prompt_text: str
    prompt_ids: torch.Tensor
    attention_mask: torch.Tensor
    target: str
    metadata: dict[str, Any]


def _as_tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


def build_gsm8k_supervised_sample(
    repo_root: str | Path,
    raw_example: dict[str, Any],
    tokenizer: object,
    template: object,
    cutoff_len: int,
    template_name: str,
    dataset_name: str = "gsm8k_sft",
) -> SupervisedSample:
    """Build one GSM8K sample using this repo's LLaMA-Factory converter and SFT processor."""
    root = ensure_repo_imports(repo_root)

    from llamafactory.data.converter import AlpacaDatasetConverter
    from llamafactory.data.parser import get_dataset_list
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments

    data_args = DataArguments(
        dataset_dir=str(root / "data_eaft"),
        dataset=dataset_name,
        template=template_name,
        cutoff_len=cutoff_len,
    )
    dataset_attr = get_dataset_list([dataset_name], str(root / "data_eaft"))[0]
    converted = AlpacaDatasetConverter(dataset_attr=dataset_attr, data_args=data_args)(raw_example)
    processor = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer, processor=None, data_args=data_args)
    batch = {key: [value] for key, value in converted.items()}
    processed = processor.preprocess_dataset(batch)
    if not processed["input_ids"]:
        raise ValueError("LLaMA-Factory dropped the GSM8K example during preprocessing.")

    return SupervisedSample(
        input_ids=_as_tensor(processed["input_ids"][0]),
        attention_mask=_as_tensor(processed["attention_mask"][0]),
        labels=_as_tensor(processed["labels"][0]),
        prompt_text=raw_example["question"],
        response_text=raw_example["answer"],
        metadata={"dataset_name": dataset_name, "source": "openai/gsm8k", "columns": {"prompt": "question", "response": "answer"}},
    )


def _load_prepare_datasets_module(repo_root: str | Path):
    path = Path(repo_root) / "eval" / "prepare_datasets.py"
    spec = util.spec_from_file_location("eaft_prepare_datasets", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_mmlu_prompt_from_repo(repo_root: str | Path, raw_example: dict[str, Any]) -> tuple[str, str]:
    """Use eval/prepare_datasets.py as the MMLU prompt source of truth."""
    ensure_repo_imports(repo_root)
    module = _load_prepare_datasets_module(repo_root)
    return module._format_mmlu_prompt(raw_example)


def build_mmlu_observation(
    repo_root: str | Path,
    raw_example: dict[str, Any],
    tokenizer: object,
    template: object,
    subject: str,
    source_index: int,
) -> MMLUObservation:
    prompt_text, target = format_mmlu_prompt_from_repo(repo_root, raw_example)
    messages = template.mm_plugin.process_messages(
        [{"role": "user", "content": prompt_text}],
        [],
        [],
        [],
        None,
    )
    paired_messages = messages + [{"role": "assistant", "content": ""}]
    prompt_ids, _ = template.encode_oneturn(tokenizer, paired_messages, None, None)

    return MMLUObservation(
        prompt_text=prompt_text,
        prompt_ids=_as_tensor(prompt_ids),
        attention_mask=torch.ones(len(prompt_ids), dtype=torch.long),
        target=target,
        metadata={
            "dataset_name": "mmlu",
            "subject": subject,
            "source_index": source_index,
            "question": raw_example["question"],
            "choices": list(raw_example["choices"]),
        },
    )


def resolve_option_token_ids(
    tokenizer: object,
    template: object,
    prompt_text: str,
    candidates: tuple[str, ...] = ("A", "B", "C", "D"),
) -> dict[str, dict[str, Any]]:
    """Resolve exact one-token assistant continuations after the repo's chat-wrapped MMLU prompt."""
    resolved: dict[str, dict[str, Any]] = {}
    base_messages = template.mm_plugin.process_messages(
        [{"role": "user", "content": prompt_text}],
        [],
        [],
        [],
        None,
    )
    _, empty_response_ids = template.encode_oneturn(
        tokenizer,
        base_messages + [{"role": "assistant", "content": ""}],
        None,
        None,
    )

    for letter in candidates:
        chosen = None
        attempts = []
        for text in (letter, " " + letter):
            _, response_ids = template.encode_oneturn(
                tokenizer,
                base_messages + [{"role": "assistant", "content": text}],
                None,
                None,
            )
            content_ids = None
            if empty_response_ids and response_ids[-len(empty_response_ids):] == empty_response_ids:
                content_ids = response_ids[: -len(empty_response_ids)]
            else:
                content_ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(content_ids, skip_special_tokens=False)
            attempt = {"text": text, "ids": [int(x) for x in content_ids], "decoded": decoded, "n_tokens": len(content_ids)}
            attempts.append(attempt)
            if chosen is None and len(content_ids) == 1:
                chosen = attempt

        if chosen is None:
            raise ValueError(f"No one-token continuation found for option {letter}: {attempts}")
        resolved[letter] = {"token_id": int(chosen["ids"][0]), "chosen_text": chosen["text"], "attempts": attempts}

    return resolved


def load_hf_subset(dataset_name: str, subset: str | None, split: str, n: int | None, cache_dir: str | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"path": dataset_name, "split": split}
    if subset is not None:
        kwargs["name"] = subset
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    dataset = load_dataset(**kwargs)
    limit = len(dataset) if n is None else min(n, len(dataset))
    return [dict(dataset[i]) for i in range(limit)]
