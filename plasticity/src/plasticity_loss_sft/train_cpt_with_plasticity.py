from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.optim import AdamW
from transformers import PreTrainedTokenizerBase, get_scheduler, set_seed

from plasticity_loss_sft.data import load_instruction_dataset, to_prompt_completion_dataset
from plasticity_loss_sft.modeling import (
    SUPPORTED_MODELS,
    get_lm_head_weight,
    inspect_model_access,
    load_model_and_tokenizer,
    supports_assistant_token_mask,
)
from plasticity_loss_sft.readout_geometry import compute_readout_geometry, load_token_support, validate_metrics
from plasticity_loss_sft.runtime import bf16_supported, gpu_report, package_versions


DEFAULT_EVAL_SPECS = (
    "gsm8k=data/prepared_subsets/gsm8k/train.jsonl",
    "mbpp=data/prepared_subsets/mbpp/train.jsonl",
    "dolly_qa=data/prepared_subsets/dolly_qa/train.jsonl",
    "bio=data/prepared_subsets/bio/train.jsonl",
)


@dataclass(frozen=True)
class EvalTask:
    name: str
    dataset_path: str


@dataclass(frozen=True)
class EncodedEvalExample:
    input_ids: list[int]
    response_mask: list[int]


class JsonlCptStream:
    def __init__(self, path: Path, max_training_tokens: int | None, sequence_length: int) -> None:
        self.path = path
        self.max_training_tokens = max_training_tokens
        self.sequence_length = sequence_length

    def __iter__(self):
        emitted_tokens = 0
        output_chunk_index = 0
        buffer: list[int] = []
        for row in self.iter_rows():
            buffer.extend(int(token_id) for token_id in row["input_ids"] if int(token_id) >= 0)
            while len(buffer) >= self.sequence_length:
                if (
                    self.max_training_tokens is not None
                    and emitted_tokens + self.sequence_length > self.max_training_tokens
                ):
                    return
                input_ids = buffer[: self.sequence_length]
                del buffer[: self.sequence_length]
                emitted_tokens += self.sequence_length
                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(self.sequence_length, dtype=torch.long),
                    "token_count": self.sequence_length,
                    "chunk_index": output_chunk_index,
                }
                output_chunk_index += 1

    def iter_rows(self):
        if self.path.is_dir():
            paths = sorted(self.path.glob("*.jsonl"))
            if not paths:
                raise FileNotFoundError(f"No JSONL shards found under CPT data directory: {self.path}")
        else:
            paths = [self.path]
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Biomedical CPT with supervised plasticity tracking.")
    parser.add_argument("--model_name", choices=SUPPORTED_MODELS, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--cpt_data_path", default="data/cpt_pubmed/cpt_train.jsonl")
    parser.add_argument("--eval_specs", nargs="+", default=list(DEFAULT_EVAL_SPECS))
    parser.add_argument("--output_dir", default="outputs/cpt_plasticity")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_training_tokens", type=int, default=100_000_000)
    parser.add_argument("--eval_every_tokens", type=int, default=500_000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--eval_limit", type=int, default=1000)
    parser.add_argument("--plasticity_block_size", type=int, default=8)
    parser.add_argument("--readout_geometry_support_files", nargs="+", default=[])
    parser.add_argument("--fixed_cpt_probe_specs", nargs="+", default=[], help="Fixed LM probe specs as name=jsonl_or_dir.")
    parser.add_argument("--fixed_cpt_probe_limit", type=int, default=64, help="Max packed sequences per fixed CPT-domain LM probe eval.")
    parser.add_argument("--checkpoint_sync_dir", default=None, help="Optional remote destination directory for per-checkpoint rsync.")
    parser.add_argument("--checkpoint_sync_host", default=None, help="SSH host used with --checkpoint_sync_dir.")
    parser.add_argument("--delete_synced_checkpoints", action="store_true", help="Delete local checkpoint dirs after successful rsync.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler_type", default="constant_after_warmup")
    parser.add_argument("--warmup_tokens", type=int, default=1_000_000)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wandb_project", default="plasticity-loss")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument(
        "--checkpoint_tokens",
        nargs="+",
        default=["5M", "10M", "20M", "40M", "100M"],
        help="Cumulative token thresholds for checkpoint saves. Supports suffixes K/M/B; use 'none' to disable.",
    )
    parser.add_argument("--save_model_at_end", action="store_true")
    parser.add_argument("--skip_initial_eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_thresholds = parse_token_thresholds(args.checkpoint_tokens, args.max_training_tokens)

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_run_name:
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)
    wandb_run = init_wandb(args)

    use_bf16 = bf16_supported() and not args.no_bf16
    print_json("package_versions", package_versions())
    print(f"bf16: {use_bf16}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, use_bf16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    print_json("model_access", inspect_model_access(model).__dict__)

    eval_tasks = parse_eval_specs(args.eval_specs)
    encoded_eval = load_eval_examples(
        eval_tasks=eval_tasks,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        limit=args.eval_limit,
        seed=args.seed,
    )
    print_json(
        "eval_examples",
        {task_name: len(examples) for task_name, examples in encoded_eval.items()},
    )
    readout_supports = load_readout_supports(args.readout_geometry_support_files)
    if readout_supports:
        print_json("readout_geometry_supports", {name: support.path for name, support in readout_supports.items()})
    fixed_cpt_probe_specs = parse_fixed_cpt_probe_specs(args.fixed_cpt_probe_specs)
    if fixed_cpt_probe_specs:
        print_json("fixed_cpt_probe_specs", {name: str(path) for name, path in fixed_cpt_probe_specs.items()})

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    tokens_per_optimizer_step = args.max_seq_length * args.per_device_train_batch_size * args.gradient_accumulation_steps
    estimated_training_steps = max(1, math.ceil(args.max_training_tokens / tokens_per_optimizer_step))
    warmup_steps = args.warmup_steps
    if warmup_steps <= 0 and args.warmup_tokens > 0:
        warmup_steps = math.ceil(args.warmup_tokens / tokens_per_optimizer_step)
    scheduler_name = normalize_scheduler_name(args.lr_scheduler_type)
    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=estimated_training_steps,
    )

    metrics_writer = LocalMetricsWriter(output_dir / "cpt_plasticity_metrics.csv", output_dir / "cpt_plasticity_metrics.jsonl")
    readout_writer = LocalMetricsWriter(output_dir / "cpt_readout_geometry_metrics.csv", output_dir / "cpt_readout_geometry_metrics.jsonl")
    cpt_stream = JsonlCptStream(Path(args.cpt_data_path), args.max_training_tokens, args.max_seq_length)

    cumulative_tokens = 0
    optimizer_step = 0
    micro_step = 0
    running_loss = 0.0
    running_loss_count = 0
    next_eval_tokens = args.eval_every_tokens
    next_checkpoint_index = 0
    saved_checkpoints: list[dict[str, Any]] = []

    if not args.skip_initial_eval:
        rows = evaluate_and_log(
            model=model,
            tokenizer=tokenizer,
            encoded_eval=encoded_eval,
            args=args,
            cumulative_tokens=0,
            optimizer_step=0,
            event="initial_eval",
            train_loss=None,
            metrics_writer=metrics_writer,
            readout_supports=readout_supports,
            readout_writer=readout_writer,
            fixed_cpt_probe_specs=fixed_cpt_probe_specs,
        )
        print(f"Initial eval rows: {len(rows)}")

    optimizer.zero_grad(set_to_none=True)
    batch_buffer: list[dict[str, Any]] = []
    for sample in cpt_stream:
        batch_buffer.append(sample)
        if len(batch_buffer) < args.per_device_train_batch_size:
            continue
        batch = collate_cpt_batch(batch_buffer, device)
        batch_buffer = []
        labels = batch["input_ids"].clone()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()

        raw_loss = float(outputs.loss.detach().float().item())
        running_loss += raw_loss
        running_loss_count += 1
        cumulative_tokens += int(batch["attention_mask"].sum().item())
        micro_step += 1

        if micro_step % args.gradient_accumulation_steps == 0:
            grad_norm = clip_grad_norm(model, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if optimizer_step % args.logging_steps == 0:
                train_loss = running_loss / max(1, running_loss_count)
                log_train_status(
                    cumulative_tokens=cumulative_tokens,
                    optimizer_step=optimizer_step,
                    train_loss=train_loss,
                    learning_rate=float(scheduler.get_last_lr()[0]),
                    grad_norm=grad_norm,
                )
                running_loss = 0.0
                running_loss_count = 0

        while (
            next_checkpoint_index < len(checkpoint_thresholds)
            and cumulative_tokens >= checkpoint_thresholds[next_checkpoint_index]
        ):
            checkpoint_report = save_cpt_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                threshold_tokens=checkpoint_thresholds[next_checkpoint_index],
                cumulative_tokens=cumulative_tokens,
                optimizer_step=optimizer_step,
            )
            checkpoint_report = sync_checkpoint_if_requested(checkpoint_report, args)
            saved_checkpoints.append(checkpoint_report)
            wandb_log(
                {
                    "cumulative_training_tokens": cumulative_tokens,
                    "cpt/checkpoint_saved": 1,
                    "cpt/checkpoint_threshold_tokens": checkpoint_thresholds[next_checkpoint_index],
                },
                step=cumulative_tokens,
            )
            next_checkpoint_index += 1

        if cumulative_tokens >= next_eval_tokens:
            train_loss = running_loss / max(1, running_loss_count) if running_loss_count else None
            evaluate_and_log(
                model=model,
                tokenizer=tokenizer,
                encoded_eval=encoded_eval,
                args=args,
                cumulative_tokens=cumulative_tokens,
                optimizer_step=optimizer_step,
                event="scheduled_eval",
                train_loss=train_loss,
                metrics_writer=metrics_writer,
                readout_supports=readout_supports,
                readout_writer=readout_writer,
                fixed_cpt_probe_specs=fixed_cpt_probe_specs,
            )
            while next_eval_tokens <= cumulative_tokens:
                next_eval_tokens += args.eval_every_tokens
            model.train()

        if cumulative_tokens >= args.max_training_tokens:
            break

    if batch_buffer:
        print(f"Dropped trailing partial CPT batch with {len(batch_buffer)} examples to avoid changing batch shape.")
    if micro_step > 0 and micro_step % args.gradient_accumulation_steps != 0:
        grad_norm = clip_grad_norm(model, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        print_json(
            "final_partial_optimizer_step",
            {
                "cumulative_training_tokens": cumulative_tokens,
                "optimizer_step": optimizer_step,
                "grad_norm": grad_norm,
            },
        )
        while (
            next_checkpoint_index < len(checkpoint_thresholds)
            and cumulative_tokens >= checkpoint_thresholds[next_checkpoint_index]
        ):
            checkpoint_report = save_cpt_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                threshold_tokens=checkpoint_thresholds[next_checkpoint_index],
                cumulative_tokens=cumulative_tokens,
                optimizer_step=optimizer_step,
            )
            checkpoint_report = sync_checkpoint_if_requested(checkpoint_report, args)
            saved_checkpoints.append(checkpoint_report)
            next_checkpoint_index += 1

    final_loss = running_loss / max(1, running_loss_count) if running_loss_count else None
    evaluate_and_log(
        model=model,
        tokenizer=tokenizer,
        encoded_eval=encoded_eval,
        args=args,
        cumulative_tokens=cumulative_tokens,
        optimizer_step=optimizer_step,
        event="final_eval",
        train_loss=final_loss,
        metrics_writer=metrics_writer,
        readout_supports=readout_supports,
        readout_writer=readout_writer,
        fixed_cpt_probe_specs=fixed_cpt_probe_specs,
    )

    report = {
        "model_name": args.model_name,
        "cpt_data_path": args.cpt_data_path,
        "output_dir": str(output_dir),
        "max_training_tokens": args.max_training_tokens,
        "actual_training_tokens": cumulative_tokens,
        "eval_every_tokens": args.eval_every_tokens,
        "optimizer_step": optimizer_step,
        "micro_step": micro_step,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "resolved_lr_scheduler_type": scheduler_name,
        "warmup_tokens": args.warmup_tokens,
        "warmup_steps": warmup_steps,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "eval_specs": args.eval_specs,
        "eval_limit": args.eval_limit,
        "readout_geometry_support_files": args.readout_geometry_support_files,
        "fixed_cpt_probe_specs": args.fixed_cpt_probe_specs,
        "fixed_cpt_probe_limit": args.fixed_cpt_probe_limit,
        "checkpoint_sync_dir": args.checkpoint_sync_dir,
        "delete_synced_checkpoints": args.delete_synced_checkpoints,
        "checkpoint_tokens": checkpoint_thresholds,
        "saved_checkpoints": saved_checkpoints,
        "gpu": gpu_report().__dict__,
    }
    write_json(output_dir / "cpt_plasticity_report.json", report)
    print_json("cpt_plasticity_report", report)

    if args.save_model_at_end:
        final_model_dir = output_dir / "final_model"
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
    if wandb_run is not None:
        wandb_run.finish()


def parse_token_thresholds(values: list[str], max_training_tokens: int) -> list[int]:
    if not values:
        return []
    if len(values) == 1 and values[0].strip().lower() in {"none", "0", "false", "off"}:
        return []
    thresholds = []
    for value in values:
        threshold = parse_token_count(value)
        if threshold <= 0:
            raise ValueError(f"Checkpoint token threshold must be positive, got {value!r}")
        if threshold <= max_training_tokens:
            thresholds.append(threshold)
    return sorted(set(thresholds))


def parse_token_count(value: str) -> int:
    text = str(value).strip().replace("_", "")
    if not text:
        raise ValueError("Empty token count")
    multiplier = 1
    suffix = text[-1].lower()
    if suffix in {"k", "m", "b"}:
        text = text[:-1]
        multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[suffix]
    return int(float(text) * multiplier)


def checkpoint_dir_name(threshold_tokens: int) -> str:
    if threshold_tokens % 1_000_000 == 0:
        return f"checkpoint_{threshold_tokens // 1_000_000}M_tokens"
    if threshold_tokens % 1_000 == 0:
        return f"checkpoint_{threshold_tokens // 1_000}K_tokens"
    return f"checkpoint_{threshold_tokens}_tokens"


def save_cpt_checkpoint(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    threshold_tokens: int,
    cumulative_tokens: int,
    optimizer_step: int,
) -> dict[str, Any]:
    checkpoint_dir = output_dir / checkpoint_dir_name(threshold_tokens)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    report = {
        "threshold_tokens": threshold_tokens,
        "cumulative_training_tokens": cumulative_tokens,
        "optimizer_step": optimizer_step,
        "checkpoint_dir": str(checkpoint_dir),
    }
    write_json(checkpoint_dir / "checkpoint_report.json", report)
    print_json("checkpoint_saved", report)
    return report


def validate_args(args: argparse.Namespace) -> None:
    cpt_path = Path(args.cpt_data_path)
    if not cpt_path.is_file() and not cpt_path.is_dir():
        raise FileNotFoundError(f"Missing CPT data file/directory: {args.cpt_data_path}")
    if args.checkpoint_sync_dir and not args.checkpoint_sync_host:
        raise ValueError("--checkpoint_sync_host is required with --checkpoint_sync_dir")
    if args.max_training_tokens <= 0:
        raise ValueError("--max_training_tokens must be positive")
    if args.eval_every_tokens <= 0:
        raise ValueError("--eval_every_tokens must be positive")
    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per_device_train_batch_size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient_accumulation_steps must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if args.plasticity_block_size <= 0:
        raise ValueError("--plasticity_block_size must be positive")
    for support_file in args.readout_geometry_support_files:
        if not Path(support_file).is_file():
            raise FileNotFoundError(f"Missing readout geometry support file: {support_file}")
    for _name, probe_path in parse_fixed_cpt_probe_specs(args.fixed_cpt_probe_specs).items():
        if not probe_path.is_file() and not probe_path.is_dir():
            raise FileNotFoundError(f"Missing fixed CPT probe path: {probe_path}")


def normalize_scheduler_name(name: str) -> str:
    if name == "constant_after_warmup":
        return "constant_with_warmup"
    return name


def init_wandb(args: argparse.Namespace) -> Any | None:
    try:
        import wandb
    except ImportError:
        print("wandb unavailable; continuing without WandB logging")
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    wandb.define_metric("cumulative_training_tokens")
    wandb.define_metric("cpt/*", step_metric="cumulative_training_tokens")
    wandb.define_metric("plasticity/*", step_metric="cumulative_training_tokens")
    wandb.define_metric("fixed_cpt_probe/*", step_metric="cumulative_training_tokens")
    wandb.define_metric("readout_geometry/*", step_metric="cumulative_training_tokens")
    return run


def load_readout_supports(support_files: list[str]) -> dict[str, Any]:
    supports = {}
    for support_file in support_files:
        path = Path(support_file)
        supports[path.stem] = load_token_support(path)
    return supports


def parse_eval_specs(values: list[str]) -> list[EvalTask]:
    tasks = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected eval spec task_name=jsonl_path, got {value!r}")
        name, dataset_path = value.split("=", 1)
        name = name.strip()
        dataset_path = dataset_path.strip()
        if not name or not dataset_path:
            raise ValueError(f"Invalid eval spec: {value!r}")
        if name in seen:
            raise ValueError(f"Duplicate eval task: {name}")
        if not Path(dataset_path).is_file():
            raise FileNotFoundError(f"Missing eval dataset for {name}: {dataset_path}")
        seen.add(name)
        tasks.append(EvalTask(name=name, dataset_path=dataset_path))
    if not tasks:
        raise ValueError("At least one eval task is required")
    return tasks


def load_eval_examples(
    eval_tasks: list[EvalTask],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    limit: int | None,
    seed: int,
) -> dict[str, list[EncodedEvalExample]]:
    use_assistant_mask = supports_assistant_token_mask(tokenizer)
    encoded: dict[str, list[EncodedEvalExample]] = {}
    for task in eval_tasks:
        dataset = load_instruction_dataset(task.dataset_path, "train", limit, seed)
        if use_assistant_mask:
            examples = [encode_messages(row["messages"], tokenizer, max_seq_length) for row in dataset]
        else:
            prompt_completion = to_prompt_completion_dataset(dataset, tokenizer)
            examples = [encode_prompt_completion(row, tokenizer, max_seq_length) for row in prompt_completion]
        examples = [example for example in examples if sum(example.response_mask) > 0]
        if not examples:
            raise RuntimeError(f"No eval examples with response tokens for task {task.name}")
        encoded[task.name] = examples
    return encoded


def encode_messages(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> EncodedEvalExample:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )
    input_ids = list(encoded["input_ids"])
    mask = encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
    if mask is None:
        raise RuntimeError("Tokenizer did not return assistant token masks for eval scoring.")
    return truncate_eval_example(EncodedEvalExample(input_ids=input_ids, response_mask=[int(x) for x in mask]), max_seq_length)


def encode_prompt_completion(
    row: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> EncodedEvalExample:
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    input_ids = list(prompt_ids) + list(completion_ids)
    response_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    return truncate_eval_example(EncodedEvalExample(input_ids=input_ids, response_mask=response_mask), max_seq_length)


def truncate_eval_example(example: EncodedEvalExample, max_seq_length: int) -> EncodedEvalExample:
    return EncodedEvalExample(
        input_ids=example.input_ids[:max_seq_length],
        response_mask=example.response_mask[:max_seq_length],
    )


def collate_cpt_batch(batch: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
    for row_idx, row in enumerate(batch):
        length = len(row["input_ids"])
        input_ids[row_idx, :length] = row["input_ids"].to(device)
        attention_mask[row_idx, :length] = row["attention_mask"].to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def clip_grad_norm(model: torch.nn.Module, max_grad_norm: float) -> float:
    if max_grad_norm <= 0:
        return math.nan
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    return float(grad_norm.detach().float().item())


def log_train_status(
    cumulative_tokens: int,
    optimizer_step: int,
    train_loss: float,
    learning_rate: float,
    grad_norm: float,
) -> None:
    payload = {
        "cumulative_training_tokens": cumulative_tokens,
        "cpt/optimizer_step": optimizer_step,
        "cpt/train_loss": train_loss,
        "cpt/learning_rate": learning_rate,
        "cpt/grad_norm": grad_norm,
    }
    print_json("train_status", payload)
    wandb_log(payload, step=cumulative_tokens)


def evaluate_and_log(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    encoded_eval: dict[str, list[EncodedEvalExample]],
    args: argparse.Namespace,
    cumulative_tokens: int,
    optimizer_step: int,
    event: str,
    train_loss: float | None,
    metrics_writer: "LocalMetricsWriter",
    readout_supports: dict[str, Any],
    readout_writer: "LocalMetricsWriter",
    fixed_cpt_probe_specs: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    rows = evaluate_plasticity(
        model=model,
        tokenizer=tokenizer,
        encoded_eval=encoded_eval,
        batch_size=args.eval_batch_size,
        block_size=args.plasticity_block_size,
    )
    payload: dict[str, int | float | str] = {
        "cumulative_training_tokens": cumulative_tokens,
        "cpt/optimizer_step": optimizer_step,
        "cpt/eval_event": event,
    }
    if train_loss is not None:
        payload["cpt/train_loss"] = train_loss
    output_rows = []
    for row in rows:
        task = row["task"]
        payload[f"plasticity/{task}/g2_mean"] = row["g2_mean"]
        payload[f"plasticity/{task}/wtg2_mean"] = row["wtg2_mean"]
        payload[f"plasticity/{task}/R_mean"] = row["R_mean"]
        payload[f"plasticity/{task}/R_p95"] = row["R_p95"]
        payload[f"plasticity/{task}/num_tokens"] = row["num_tokens"]
        output_rows.append(
            {
                "event": event,
                "cumulative_training_tokens": cumulative_tokens,
                "optimizer_step": optimizer_step,
                **row,
            }
        )
    metrics_writer.append(output_rows)

    readout_rows = evaluate_readout_geometry(
        model=model,
        readout_supports=readout_supports,
        cumulative_tokens=cumulative_tokens,
        optimizer_step=optimizer_step,
        event=event,
    )
    for row in readout_rows:
        prefix = f"readout_geometry/{sanitize_metric_name(str(row['support_name']))}/{sanitize_metric_name(str(row['task']))}"
        payload[f"{prefix}/trace"] = row["trace"]
        payload[f"{prefix}/effective_rank"] = row["effective_rank"]
        payload[f"{prefix}/hoyer_concentration"] = row["hoyer_concentration"]
        payload[f"{prefix}/max_eigenvalue"] = row["max_eigenvalue"]
        payload[f"{prefix}/token_count"] = row["token_count"]
    readout_writer.append(readout_rows)

    fixed_probe_rows = evaluate_fixed_cpt_probes(
        model=model,
        tokenizer=tokenizer,
        probe_specs=fixed_cpt_probe_specs or {},
        sequence_length=args.max_seq_length,
        limit=args.fixed_cpt_probe_limit,
        cumulative_tokens=cumulative_tokens,
        optimizer_step=optimizer_step,
        event=event,
        block_size=args.plasticity_block_size,
    )
    for row in fixed_probe_rows:
        probe = sanitize_metric_name(str(row["probe"]))
        payload[f"fixed_cpt_probe/{probe}/loss"] = row["loss"]
        payload[f"fixed_cpt_probe/{probe}/perplexity"] = row["perplexity"]
        payload[f"fixed_cpt_probe/{probe}/g2_mean"] = row["g2_mean"]
        payload[f"fixed_cpt_probe/{probe}/wtg2_mean"] = row["wtg2_mean"]
        payload[f"fixed_cpt_probe/{probe}/R_mean"] = row["R_mean"]
        payload[f"fixed_cpt_probe/{probe}/R_p95"] = row["R_p95"]
        payload[f"fixed_cpt_probe/{probe}/num_tokens"] = row["num_tokens"]
    if fixed_probe_rows:
        fixed_writer = LocalMetricsWriter(
            metrics_writer.csv_path.parent / "cpt_fixed_probe_metrics.csv",
            metrics_writer.csv_path.parent / "cpt_fixed_probe_metrics.jsonl",
        )
        fixed_writer.append(fixed_probe_rows)

    print_json("eval_metrics", payload)
    wandb_log(payload, step=cumulative_tokens)
    return output_rows


@torch.no_grad()
def evaluate_fixed_cpt_probes(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    probe_specs: dict[str, Path],
    sequence_length: int,
    limit: int,
    cumulative_tokens: int,
    optimizer_step: int,
    event: str,
    block_size: int = 8,
) -> list[dict[str, Any]]:
    if not probe_specs:
        return []
    was_training = bool(model.training)
    model.eval()
    rows: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    lm_head_weight = get_lm_head_weight(model).detach()
    try:
        for probe_name, probe_path in probe_specs.items():
            total_nll = 0.0
            total_tokens = 0
            num_sequences = 0
            g2_sum = 0.0
            wtg2_sum = 0.0
            r_values: list[float] = []
            stream = JsonlCptStream(probe_path, max_training_tokens=None, sequence_length=sequence_length)
            for sample in stream:
                if limit is not None and limit > 0 and num_sequences >= limit:
                    break
                input_ids = sample["input_ids"].unsqueeze(0).to(device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :].float()
                targets = input_ids[:, 1:]
                target_mask = attention_mask[:, 1:].bool()
                if not target_mask.any():
                    continue
                selected_logits = logits[target_mask]
                selected_targets = targets[target_mask]
                total_nll += float(
                    torch.nn.functional.cross_entropy(selected_logits, selected_targets, reduction="sum").item()
                )
                block_metrics = score_selected_tokens(
                    selected_logits=selected_logits,
                    selected_targets=selected_targets,
                    lm_head_weight=lm_head_weight,
                    block_size=block_size,
                )
                token_count = int(selected_targets.numel())
                total_tokens += token_count
                num_sequences += 1
                g2_sum += block_metrics["g2_sum"]
                wtg2_sum += block_metrics["wtg2_sum"]
                r_values.extend(block_metrics["r_values"])
            if total_tokens == 0:
                raise RuntimeError(f"Fixed CPT probe {probe_name} produced zero tokens from {probe_path}")
            r_sorted = sorted(r_values)
            loss = total_nll / total_tokens
            rows.append(
                {
                    "event": event,
                    "cumulative_training_tokens": cumulative_tokens,
                    "optimizer_step": optimizer_step,
                    "probe": probe_name,
                    "path": str(probe_path),
                    "num_sequences": num_sequences,
                    "num_tokens": total_tokens,
                    "loss": loss,
                    "perplexity": math.exp(min(loss, 50.0)),
                    "g2_mean": g2_sum / total_tokens,
                    "wtg2_mean": wtg2_sum / total_tokens,
                    "R_mean": sum(r_values) / len(r_values) if r_values else math.nan,
                    "R_p95": percentile(r_sorted, 0.95),
                }
            )
    finally:
        if was_training:
            model.train()
    return rows


def parse_fixed_cpt_probe_specs(values: list[str]) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected fixed CPT probe spec name=path, got {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid fixed CPT probe spec: {value!r}")
        if name in specs:
            raise ValueError(f"Duplicate fixed CPT probe name: {name}")
        specs[name] = Path(path)
    return specs


def sync_checkpoint_if_requested(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if not args.checkpoint_sync_dir:
        return report
    checkpoint_dir = Path(str(report["checkpoint_dir"]))
    destination_root = str(args.checkpoint_sync_dir).rstrip("/")
    destination = f"{destination_root}/{checkpoint_dir.name}/"
    mkdir_cmd = ["ssh", args.checkpoint_sync_host, "mkdir", "-p", destination_root]
    rsync_cmd = ["rsync", "-a", "--partial", f"{checkpoint_dir}/", f"{args.checkpoint_sync_host}:{destination}"]
    sync_report = {
        "local_checkpoint_dir": str(checkpoint_dir),
        "destination": f"{args.checkpoint_sync_host}:{destination}",
        "mkdir_returncode": None,
        "rsync_returncode": None,
        "synced": False,
        "local_deleted": False,
    }
    try:
        mkdir_result = subprocess.run(mkdir_cmd, check=False)
        sync_report["mkdir_returncode"] = mkdir_result.returncode
        if mkdir_result.returncode == 0:
            rsync_result = subprocess.run(rsync_cmd, check=False)
            sync_report["rsync_returncode"] = rsync_result.returncode
            sync_report["synced"] = rsync_result.returncode == 0
            if sync_report["synced"] and args.delete_synced_checkpoints:
                shutil.rmtree(checkpoint_dir)
                sync_report["local_deleted"] = True
    except Exception as exc:
        sync_report["error"] = repr(exc)
    report["sync_report"] = sync_report
    sync_path = checkpoint_dir / "checkpoint_sync_report.json"
    if checkpoint_dir.exists():
        write_json(sync_path, sync_report)
    print_json("checkpoint_sync", sync_report)
    return report


@torch.no_grad()
def evaluate_readout_geometry(
    model: torch.nn.Module,
    readout_supports: dict[str, Any],
    cumulative_tokens: int,
    optimizer_step: int,
    event: str,
) -> list[dict[str, Any]]:
    if not readout_supports:
        return []
    lm_head_weight = get_lm_head_weight(model).detach()
    rows: list[dict[str, Any]] = []
    for support_name, support in readout_supports.items():
        for task, token_ids in support.task_token_ids.items():
            metrics, _spectrum = compute_readout_geometry(
                lm_head_weight=lm_head_weight,
                token_ids=token_ids,
                support_name=support_name,
                support_type=support.support_type,
                task=task,
            )
            validate_metrics(metrics)
            row = metrics.to_dict()
            rows.append(
                {
                    "event": event,
                    "cumulative_training_tokens": cumulative_tokens,
                    "optimizer_step": optimizer_step,
                    **row,
                }
            )
    return rows


def sanitize_metric_name(value: str) -> str:
    return "_".join(value.replace("/", "_").replace(" ", "_").split()) or "unknown"


@torch.no_grad()
def evaluate_plasticity(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    encoded_eval: dict[str, list[EncodedEvalExample]],
    batch_size: int,
    block_size: int,
) -> list[dict[str, float | str]]:
    was_training = bool(model.training)
    model.eval()
    try:
        lm_head_weight = get_lm_head_weight(model).detach()
        rows = []
        for task_name, examples in encoded_eval.items():
            rows.append(
                evaluate_task_plasticity(
                    model=model,
                    tokenizer=tokenizer,
                    lm_head_weight=lm_head_weight,
                    task_name=task_name,
                    examples=examples,
                    batch_size=batch_size,
                    block_size=block_size,
                )
            )
        return rows
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def evaluate_task_plasticity(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    lm_head_weight: torch.Tensor,
    task_name: str,
    examples: list[EncodedEvalExample],
    batch_size: int,
    block_size: int,
) -> dict[str, float | str]:
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    total_tokens = 0
    total_examples = 0
    g2_sum = 0.0
    wtg2_sum = 0.0
    r_values: list[float] = []

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        max_len = max(len(example.input_ids) for example in batch)
        input_ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        response_mask = torch.zeros((len(batch), max_len), dtype=torch.bool, device=device)
        for row_index, example in enumerate(batch):
            length = len(example.input_ids)
            input_ids[row_index, :length] = torch.tensor(example.input_ids, dtype=torch.long, device=device)
            attention_mask[row_index, :length] = 1
            response_mask[row_index, :length] = torch.tensor(example.response_mask, dtype=torch.bool, device=device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        target_mask = response_mask[:, 1:] & attention_mask[:, 1:].bool()
        if not target_mask.any():
            continue
        selected_logits = logits[target_mask]
        selected_targets = targets[target_mask]
        block_metrics = score_selected_tokens(
            selected_logits=selected_logits,
            selected_targets=selected_targets,
            lm_head_weight=lm_head_weight,
            block_size=block_size,
        )
        token_count = int(selected_targets.numel())
        total_tokens += token_count
        total_examples += len(batch)
        g2_sum += block_metrics["g2_sum"]
        wtg2_sum += block_metrics["wtg2_sum"]
        r_values.extend(block_metrics["r_values"])

    if total_tokens == 0:
        raise RuntimeError(f"Evaluation task {task_name} produced zero scored response tokens")
    r_sorted = sorted(r_values)
    return {
        "task": task_name,
        "num_examples": float(total_examples),
        "num_tokens": float(total_tokens),
        "g2_mean": g2_sum / total_tokens,
        "wtg2_mean": wtg2_sum / total_tokens,
        "R_mean": sum(r_values) / len(r_values) if r_values else math.nan,
        "R_p95": percentile(r_sorted, 0.95),
    }


def score_selected_tokens(
    selected_logits: torch.Tensor,
    selected_targets: torch.Tensor,
    lm_head_weight: torch.Tensor,
    block_size: int,
) -> dict[str, Any]:
    g2_sum = 0.0
    wtg2_sum = 0.0
    r_values: list[float] = []
    for start in range(0, selected_logits.shape[0], block_size):
        end = start + block_size
        block_logits = selected_logits[start:end]
        block_targets = selected_targets[start:end]
        probs = torch.softmax(block_logits, dim=-1)
        p_y = probs.gather(1, block_targets[:, None]).squeeze(1)
        g2 = 1.0 - (2.0 * p_y) + torch.sum(probs * probs, dim=-1)
        expected_readout = probs.to(lm_head_weight.dtype) @ lm_head_weight
        target_readout = lm_head_weight.index_select(0, block_targets)
        wtg = target_readout - expected_readout
        wtg2 = torch.sum(wtg.float() * wtg.float(), dim=-1)
        r = wtg2 / g2.clamp_min(1e-12)
        g2_sum += float(g2.sum().item())
        wtg2_sum += float(wtg2.sum().item())
        r_values.extend(float(value) for value in r.detach().cpu().tolist())
    return {"g2_sum": g2_sum, "wtg2_sum": wtg2_sum, "r_values": r_values}


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


class LocalMetricsWriter:
    def __init__(self, csv_path: Path, jsonl_path: Path) -> None:
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def wandb_log(payload: dict[str, int | float | str], step: int) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log(payload, step=step)


def print_json(label: str, value: object) -> None:
    print(f"{label}: {json.dumps(value, indent=2, sort_keys=True)}")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
