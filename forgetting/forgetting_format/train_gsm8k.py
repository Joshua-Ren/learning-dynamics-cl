"""Fine-tune one controlled GSM8K prompt-format condition.

The input JSONL is produced by prepare_datasets.py. Run this script once for
question_answer and once for problem_result with identical optimization settings.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from common import read_jsonl


DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune one paired GSM8K format condition.")
    parser.add_argument("--train_file", required=True, help="gsm8k_<variant>.jsonl from prepare_datasets.py")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--finetune_mode", choices=["lora", "full", "lm_head"], default="lora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_strategy", choices=["no", "steps", "epoch"], default="epoch")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chat_template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat_template_model", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


def expanded(values: list[str]) -> list[str]:
    return [part.strip() for value in values for part in value.split(",") if part.strip()]


def load_records(train_file: str) -> list[dict[str, object]]:
    records = list(read_jsonl(train_file))
    if not records:
        raise ValueError(f"No records in {train_file}")
    missing = {"prompt", "completion"} - set(records[0])
    if missing:
        raise KeyError(f"{train_file} lacks required fields: {sorted(missing)}")
    variants = {str(record.get("format_variant", "")) for record in records}
    if len(variants) != 1:
        raise ValueError(f"Expected one format_variant per training file, found {sorted(variants)}")
    return records


def render_training_prompt(tokenizer, prompt: str, chat_template: bool) -> str:
    if not chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_dataset(
    records: list[dict[str, object]],
    tokenizer,
    chat_template: bool,
    max_length: int,
) -> Dataset:
    """Create completion-only labels without importing TRL."""
    tokenized = []
    skipped_without_completion = 0
    left_truncated_prompt = 0
    for record in records:
        prompt_text = render_training_prompt(tokenizer, str(record["prompt"]), chat_template)
        completion = str(record["completion"])
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        input_ids = tokenizer(prompt_text + completion, add_special_tokens=False)["input_ids"]
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Prompt tokenization is not a prefix of prompt+completion tokenization. "
                "Use a completion that starts with whitespace."
            )
        if tokenizer.eos_token_id is not None and (
            not input_ids or input_ids[-1] != tokenizer.eos_token_id
        ):
            input_ids.append(tokenizer.eos_token_id)

        retained_prompt_length = len(prompt_ids)
        if len(input_ids) > max_length:
            # Preserve completion tokens: trim only the beginning of an
            # overlong prompt rather than silently removing its answer.
            left_trim = len(input_ids) - max_length
            input_ids = input_ids[left_trim:]
            retained_prompt_length = max(0, len(prompt_ids) - left_trim)
            left_truncated_prompt += 1
        labels = [-100] * min(retained_prompt_length, len(input_ids))
        labels.extend(input_ids[len(labels) :])
        if not any(label != -100 for label in labels):
            skipped_without_completion += 1
            continue
        tokenized.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )

    if not tokenized:
        raise ValueError(
            "No examples retain completion tokens after truncation. Increase --max_length."
        )
    if skipped_without_completion:
        print(f"Skipped {skipped_without_completion} examples with no completion tokens after truncation.")
    if left_truncated_prompt:
        print(f"Left-truncated {left_truncated_prompt} overlong prompts to preserve completions.")
    return Dataset.from_list(tokenized)


@dataclass
class CompletionOnlyDataCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            length = len(feature["input_ids"])
            padding = max_length - length
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def configure_lm_head_only(model) -> None:
    """Untie, when necessary, and train only the causal-LM output head."""
    output_head = model.get_output_embeddings()
    if output_head is None or not hasattr(output_head, "weight"):
        raise ValueError("Model does not expose a weight-bearing output embedding / lm_head module.")

    input_embeddings = model.get_input_embeddings()
    was_tied = bool(
        input_embeddings is not None
        and hasattr(input_embeddings, "weight")
        and output_head.weight.data_ptr() == input_embeddings.weight.data_ptr()
    )
    if was_tied:
        # All selected models tie input embeddings to lm_head by default. Clone
        # the output matrix so this intervention truly updates lm_head only.
        output_head.weight = torch.nn.Parameter(output_head.weight.detach().clone())
        model.config.tie_word_embeddings = False

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in output_head.parameters():
        parameter.requires_grad = True

    head_parameter_ids = {id(parameter) for parameter in output_head.parameters()}
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any(
        id(parameter) not in head_parameter_ids
        for _, parameter in model.named_parameters()
        if parameter.requires_grad
    ):
        raise RuntimeError("lm_head mode left parameters outside the output head trainable.")
    print(
        f"lm_head-only tuning: untied_from_input_embeddings={was_tied}; "
        f"trainable={', '.join(trainable)}"
    )


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if args.chat_template and tokenizer.chat_template is None:
        if not args.chat_template_model:
            raise ValueError(
                "This tokenizer has no chat template. Pass --chat_template_model or --no-chat_template."
            )
        template_tokenizer = AutoTokenizer.from_pretrained(
            args.chat_template_model,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        if template_tokenizer.chat_template is None:
            raise ValueError(f"{args.chat_template_model} has no chat template")
        tokenizer.chat_template = template_tokenizer.chat_template

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if args.finetune_mode == "lm_head":
        configure_lm_head_only(model)
        return model, tokenizer

    if args.finetune_mode == "full":
        return model, tokenizer

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=expanded(args.lora_target_modules),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model, tokenizer


def save_run_manifest(output_dir: Path, args: argparse.Namespace, records: list[dict[str, object]]) -> None:
    metadata = records[0].get("metadata", {})
    payload = {
        "train_file": str(Path(args.train_file).resolve()),
        "format_variant": records[0].get("format_variant"),
        "num_train_examples": len(records),
        "first_source_row": records[0].get("source_row"),
        "last_source_row": records[-1].get("source_row"),
        "response_label": metadata.get("response_label") if isinstance(metadata, dict) else None,
        "model_name": args.model_name,
        "finetune_mode": args.finetune_mode,
        "chat_template": args.chat_template,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
    }
    with (output_dir / "format_run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    records = load_records(args.train_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    train_dataset = tokenize_dataset(records, tokenizer, args.chat_template, args.max_length)
    save_run_manifest(output_dir, args, records)

    report_to = [] if args.report_to.lower() == "none" else [args.report_to]
    config = TrainingArguments(
        output_dir=str(output_dir),
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        data_collator=CompletionOnlyDataCollator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved {records[0].get('format_variant')} condition to {output_dir}")


if __name__ == "__main__":
    main()
