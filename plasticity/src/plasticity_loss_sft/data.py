from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerBase


DATASET_ALIASES = {
    "gsm8k": {"path": "openai/gsm8k", "name": "main", "split": "train"},
    "gsm8k_sft": {"path": "openai/gsm8k", "name": "main", "split": "train"},
    "dolly": {"path": "databricks/databricks-dolly-15k", "name": None, "split": "train"},
    "mbpp": {"path": "google-research-datasets/mbpp", "name": None, "split": "train"},
    "lima": {"path": "GAIR/lima", "name": None, "split": "train"},
}


def load_instruction_dataset(
    dataset_name: str,
    split: str,
    limit: int | None,
    seed: int,
) -> Dataset:
    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    elif dataset_name in {"lima", "GAIR/lima"}:
        dataset = load_lima_jsonl(split)
    else:
        source = DATASET_ALIASES.get(dataset_name)
        if source is None:
            dataset = load_dataset(dataset_name, split=split)
        else:
            dataset_split = split or source["split"]
            if source["name"] is None:
                dataset = load_dataset(source["path"], split=dataset_split)
            else:
                dataset = load_dataset(source["path"], source["name"], split=dataset_split)

    if limit is not None:
        limit = min(limit, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(limit))
    return dataset.map(
        lambda example: _to_messages(example, dataset_name),
        remove_columns=dataset.column_names,
    )


def load_lima_jsonl(split: str) -> Dataset:
    split_name = split or "train"
    if split_name not in {"train", "test"}:
        raise ValueError(f"GAIR/lima only provides train/test jsonl files, got split={split_name!r}")
    jsonl_path = hf_hub_download(
        repo_id="GAIR/lima",
        repo_type="dataset",
        filename=f"{split_name}.jsonl",
    )
    return load_dataset("json", data_files=str(jsonl_path), split="train")


def to_prompt_completion_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    return dataset.map(
        lambda example: _messages_to_prompt_completion(example["messages"], tokenizer),
        remove_columns=dataset.column_names,
    )


def filter_prompt_completion_for_context(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    min_completion_tokens: int = 16,
) -> Dataset:
    def has_completion_after_truncation(example: Mapping[str, str]) -> bool:
        prompt_len = len(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"])
        completion_len = len(tokenizer(example["completion"], add_special_tokens=False)["input_ids"])
        return completion_len >= min_completion_tokens and prompt_len <= max_length - min_completion_tokens

    filtered = dataset.filter(has_completion_after_truncation)
    if len(filtered) == 0:
        raise RuntimeError(
            "No prompt/completion examples leave assistant tokens inside the context window. "
            "Increase max_seq_length or dataset_limit."
        )
    return filtered


def _to_messages(example: Mapping[str, Any], dataset_name: str | None = None) -> dict[str, list[dict[str, str]]]:
    if "messages" in example and example["messages"]:
        return {"messages": [_clean_message(message) for message in example["messages"]]}

    if dataset_name in {"gsm8k", "gsm8k_sft"}:
        return _gsm8k_to_messages(example)
    if dataset_name == "dolly":
        return _dolly_to_messages(example)
    if dataset_name == "mbpp":
        return _mbpp_to_messages(example)
    if dataset_name in {"lima", "GAIR/lima"}:
        return _lima_to_messages(example)

    prompt = _first_present(example, ("prompt", "instruction", "question", "input", "text"))
    response = _first_present(example, ("response", "completion", "answer", "output", "code"))
    if prompt is None or response is None:
        keys = ", ".join(sorted(example.keys()))
        raise ValueError(f"Cannot convert dataset row to chat messages. Available columns: {keys}")

    return {
        "messages": [
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(response)},
        ]
    }


def _gsm8k_to_messages(example: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {"role": "user", "content": str(example["question"])},
            {"role": "assistant", "content": str(example["answer"])},
        ]
    }


def _dolly_to_messages(example: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    prompt_parts = [f"Instruction: {example['instruction']}"]
    context = str(example.get("context") or "").strip()
    if context:
        prompt_parts.extend(["", f"Context: {context}"])
    return {
        "messages": [
            {"role": "user", "content": "\n".join(prompt_parts)},
            {"role": "assistant", "content": str(example["response"])},
        ]
    }


def _mbpp_to_messages(example: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    tests = example.get("test_list") or []
    prompt_parts = [
        "Write a Python function that solves the following programming task.",
        "",
        f"Task: {example['text']}",
    ]
    if tests:
        prompt_parts.extend(["", "The solution should satisfy these tests:"])
        prompt_parts.extend(str(test) for test in tests)

    setup = str(example.get("test_setup_code") or "").strip()
    if setup:
        prompt_parts.extend(["", "Test setup:", setup])

    return {
        "messages": [
            {"role": "user", "content": "\n".join(prompt_parts)},
            {"role": "assistant", "content": str(example["code"])},
        ]
    }


def _lima_to_messages(example: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    conversations = example.get("conversations")
    if not conversations:
        raise ValueError("Expected GAIR/lima row to contain a non-empty conversations field.")

    messages = []
    if all(isinstance(turn, str) for turn in conversations):
        for index, content in enumerate(conversations):
            role = "user" if index % 2 == 0 else "assistant"
            text = str(content).strip()
            if text:
                messages.append({"role": role, "content": text})
    else:
        for index, turn in enumerate(conversations):
            if not isinstance(turn, Mapping):
                raise ValueError(f"Unsupported LIMA conversation turn: {turn!r}")
            role = str(turn.get("role") or turn.get("from") or "").strip().lower()
            content = str(turn.get("content") or turn.get("value") or "").strip()
            if role in {"human", "user"}:
                role = "user"
            elif role in {"gpt", "assistant", "bot"}:
                role = "assistant"
            elif not role:
                role = "user" if index % 2 == 0 else "assistant"
            if content:
                messages.append({"role": role, "content": content})

    if not messages or not any(message["role"] == "assistant" for message in messages):
        raise ValueError("Expected LIMA conversation to include at least one assistant response.")
    return {"messages": messages}


def _messages_to_prompt_completion(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, str]:
    assistant_index = _last_assistant_index(messages)
    if tokenizer.chat_template is None:
        return _messages_to_plain_prompt_completion(messages, assistant_index)

    prompt_messages = messages[:assistant_index]
    full_messages = messages[: assistant_index + 1]

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt):
        raise RuntimeError("Chat-template prompt is not a prefix of the full rendered conversation.")

    completion = full_text[len(prompt) :]
    if not completion:
        raise RuntimeError("Rendered assistant completion is empty.")
    return {"prompt": prompt, "completion": completion}


def _messages_to_plain_prompt_completion(
    messages: list[dict[str, str]],
    assistant_index: int,
) -> dict[str, str]:
    prompt_parts = [_plain_message_block(message) for message in messages[:assistant_index]]
    assistant_message = messages[assistant_index]
    completion = str(assistant_message["content"]).strip()
    if not completion:
        raise RuntimeError("Rendered assistant completion is empty.")

    prompt = "\n\n".join(prompt_parts)
    if prompt:
        prompt += "\n\n"
    prompt += "Assistant:\n"
    return {"prompt": prompt, "completion": completion}


def _plain_message_block(message: Mapping[str, str]) -> str:
    role = str(message["role"]).strip().lower()
    content = str(message["content"]).strip()
    if role == "user":
        label = "User"
    elif role == "assistant":
        label = "Assistant"
    elif role == "system":
        label = "System"
    else:
        label = role.capitalize() or "Message"
    return f"{label}:\n{content}"


def _last_assistant_index(messages: list[dict[str, str]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "assistant":
            return index
    raise ValueError("Expected at least one assistant message.")


def _clean_message(message: Mapping[str, Any]) -> dict[str, str]:
    role = str(message.get("role", "")).strip()
    content = str(message.get("content", "")).strip()
    if not role or not content:
        raise ValueError(f"Invalid chat message: {message}")
    return {"role": role, "content": content}


def _first_present(example: Mapping[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        value = example.get(name)
        if value:
            return value
    return None
