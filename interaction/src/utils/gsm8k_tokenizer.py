import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List
from utils.utils import extract_solution
#from utils import extract_solution  # Only for debug,
# when run python utils/gsm8k_tokenizer.py


class GSM8KTokenizerProcessor:
    def __init__(self, model_name_or_path: str, max_length: int = 512, local_files_only: bool = False):
        """
        Args:
            model_name_or_path:
            max_length:
        """
        self.model_name_or_path = model_name_or_path
        self.model_name_lower = model_name_or_path.lower()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="right",
            truncation_side="right",
            trust_remote_code=True,
            local_files_only=local_files_only,
            # use_fast=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.answer_prefix = "Answer:\n"

    def _is_llama_family(self) -> bool:
        name = self.model_name_lower
        return ("llama" in name) or ("deepseek" in name)

    def _find_answer_start_token_idx(self, input_ids: List[int], prefix_tokens: List[int]) -> int:
        """
        Args:
            input_ids:
            prefix_tokens:
        Returns:
        """
        prefix_len = len(prefix_tokens)
        for i in range(len(input_ids) - prefix_len + 1):
            if input_ids[i:i + prefix_len] == prefix_tokens:
                return i + prefix_len

            current_token_seq = input_ids[i:i + prefix_len]
            current_text = self.tokenizer.decode(current_token_seq)
            if current_text == self.answer_prefix:
                return i + prefix_len

        return -1

    def _pad_to_max_length(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        seq_len = input_ids.shape[0]
        if seq_len > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
            return input_ids, attention_mask, labels

        pad_len = self.max_length - seq_len
        if pad_len == 0:
            return input_ids, attention_mask, labels

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.cat(
            [input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)],
            dim=0
        )
        attention_mask = torch.cat(
            [attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)],
            dim=0
        )
        labels = torch.cat(
            [labels, torch.full((pad_len,), -100, dtype=labels.dtype)],
            dim=0
        )
        return input_ids, attention_mask, labels

    def _process_llama_example(self, full_text: str, rl_text: str, answer: str) -> Dict:
        """
        LLaMA / DeepSeek family:
        Do NOT search answer_prefix token span in full_text.
        Instead, tokenize full_text and rl_text consistently with add_special_tokens=False,
        then use len(tokenize(rl_text)) as answer start.
        """
        full_enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        rl_enc = self.tokenizer(
            rl_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        input_ids_rl = rl_enc["input_ids"].squeeze(0)
        attention_mask_rl = rl_enc["attention_mask"].squeeze(0)

        labels = torch.full_like(input_ids, fill_value=-100)

        answer_start_idx = input_ids_rl.shape[0]
        seq_len = attention_mask.sum().item()

        if answer_start_idx < seq_len:
            labels[answer_start_idx:seq_len] = input_ids[answer_start_idx:seq_len]

        input_ids, attention_mask, labels = self._pad_to_max_length(
            input_ids, attention_mask, labels
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_ids_rl": input_ids_rl,
            "attention_mask_rl": attention_mask_rl,
            "ground_truth": extract_solution(answer),
        }

    def _process_default_example(self, full_text: str, rl_text: str, answer: str) -> Dict:
        """
        Original branch for Qwen / other models.
        Keep your original matching logic as much as possible.
        """
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        encoding_rl = self.tokenizer(
            rl_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids_rl = encoding_rl["input_ids"].squeeze(0)
        attention_mask_rl = encoding_rl["attention_mask"].squeeze(0)

        prefix_encoding = self.tokenizer(
            self.answer_prefix,
            add_special_tokens=True,
            return_tensors="pt"
        )
        prefix_tokens = prefix_encoding["input_ids"].squeeze(0).tolist()

        labels = torch.full_like(input_ids, fill_value=-100)

        input_ids_list = input_ids.tolist()
        answer_start_idx = self._find_answer_start_token_idx(input_ids_list, prefix_tokens)
        seq_len = attention_mask.sum().item()

        if answer_start_idx != -1 and answer_start_idx < seq_len:
            labels[answer_start_idx:seq_len] = input_ids[answer_start_idx:seq_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_ids_rl": input_ids_rl,
            "attention_mask_rl": attention_mask_rl,
            "ground_truth": extract_solution(answer),
        }

    def process_single_example(self, example: Dict) -> Dict:
        """
        Args:
            example:
        Returns:
        """
        question = example["question"].strip()
        answer = example["answer"].strip()
        instruction = 'Please answer the following question and give answer starting with "####".'

        full_text = f"{instruction}\nQuestion: {question}\n{self.answer_prefix}{answer}"
        rl_text = f"{instruction}\nQuestion: {question}\n{self.answer_prefix}"

        if self._is_llama_family():
            return self._process_llama_example(full_text, rl_text, answer)
        else:
            return self._process_default_example(full_text, rl_text, answer)

    def process_dataset(self, dataset) -> List[Dict]:
        processed_data = []
        for example in dataset:
            try:
                processed = self.process_single_example(example)
                processed_data.append(processed)
            except Exception as e:
                print(f"wrong case: {e}, example: {example}")
                continue
        return processed_data


if __name__ == "__main__":
    dataset = load_dataset("openai/gsm8k", "main", split="train[:10]")

    processor = GSM8KTokenizerProcessor(
        model_name_or_path="microsoft/Phi-3-mini-4k-instruct",
        max_length=512
    )

    processed_dataset = processor.process_dataset(dataset)

    sample = processed_dataset[0]
    print("===============")
    print(
        f"Example: \n{processor.tokenizer.decode(sample['input_ids'][:sample['attention_mask'].sum()], skip_special_tokens=True)}"
    )

    n_valid = (sample["labels"] != -100).sum().item()
    n_masked = (sample["labels"] == -100).sum().item()
    print(f"num_valid_labels: {n_valid}")
    print(f"num_masked_labels: {n_masked}")
    print(f"rl_len: {sample['input_ids_rl'].shape[0]}")
    print(f"seq_len: {sample['attention_mask'].sum().item()}")

    first_valid_idx = None
    for i in range(len(sample["labels"])):
        if sample["labels"][i].item() != -100:
            first_valid_idx = i
            break

    if first_valid_idx is not None:
        print("----- Around first valid label -----")
        left = max(0, first_valid_idx - 5)
        right = min(len(sample["input_ids"]), first_valid_idx + 5)
        for i in range(left, right):
            text = processor.tokenizer.decode([sample["input_ids"][i].item()])
            print(
                f"ID: {i} | TXT: {repr(text)} | "
                f"IDS: {sample['input_ids'][i].item()} | "
                f"LAB: {sample['labels'][i].item()}"
            )
    else:
        print("No valid labels found in this sample.")
