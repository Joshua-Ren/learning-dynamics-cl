from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, get_scheduler, set_seed

from plasticity_loss_sft.data import (
    filter_prompt_completion_for_context,
    load_instruction_dataset,
    to_prompt_completion_dataset,
)
from plasticity_loss_sft.modeling import get_lm_head_weight
from plasticity_loss_sft.runtime import bf16_supported, gpu_report, package_versions

TASK_PATHS = {
    "gsm8k": {
        "train": "data/prepared_subsets/gsm8k/train.jsonl",
        "probe": "data/prepared_subsets/gsm8k/probe.jsonl",
    },
    "mbpp": {
        "train": "data/prepared_subsets/mbpp/train.jsonl",
        "probe": "data/prepared_subsets/mbpp/probe.jsonl",
    },
    "dolly_qa": {
        "train": "data/prepared_subsets/dolly_qa/train.jsonl",
        "probe": "data/prepared_subsets/dolly_qa/probe.jsonl",
    },
}

DEFAULT_DISCOVERY_ROOTS = (
    "results",
    "outputs/checkpoints",
    "outputs/checkpoints",
)


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    response_mask: list[int]


@dataclass(frozen=True)
class ReadoutInterventionReport:
    enabled: bool
    strategy: str
    base_model_name: str | None
    embeddings_tied_before: bool
    embeddings_tied_after: bool
    input_weight_shape: tuple[int, ...] | None
    output_weight_shape: tuple[int, ...] | None
    input_weight_trainable: bool | None
    output_weight_trainable: bool
    untied_output_embeddings: bool
    reset_last_l_layers: int | None
    reset_transformer_layers: list[int]


class EncodedSftDataset(Dataset[EncodedExample]):
    def __init__(self, examples: list[EncodedExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step6 SFT plasticity sweep from Bio-CPT checkpoints.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--discover", action="store_true", help="Discover trajectories and write a job manifest.")
    mode.add_argument("--job_manifest", default=None, help="Run one job from this manifest JSONL.")
    parser.add_argument("--job_index", type=int, default=None)
    parser.add_argument("--discovery_roots", nargs="+", default=list(DEFAULT_DISCOVERY_ROOTS))
    parser.add_argument("--manifest_path", default="analysis/step6_sft_sweep/job_manifest.jsonl")
    parser.add_argument("--discovery_report_path", default="analysis/step6_sft_sweep/discovery_report.json")
    parser.add_argument("--output_root", default="outputs/step6_sft_sweep")
    parser.add_argument("--shared_output_root", default="outputs/step6_sft_sweep")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "mbpp", "dolly_qa"])
    parser.add_argument("--trajectory_filter", default=None)
    parser.add_argument("--model_filter", default=None)
    parser.add_argument("--max_checkpoints_per_trajectory", type=int, default=None)
    parser.add_argument("--dry_run_discovery", action="store_true")

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--plasticity_block_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wandb_project", default="plasticity-step6")
    parser.add_argument("--wandb_mode", default=None)
    parser.add_argument("--rewind_readout_to_base", action="store_true")
    parser.add_argument("--reset_embedding_and_readout_to_base_tied", action="store_true")
    parser.add_argument("--reset_embedding_readout_and_last_layers_to_base_tied", action="store_true")
    parser.add_argument("--reset_last_l_layers", type=int, default=1)
    parser.add_argument("--intervention_name", default=None)
    parser.add_argument("--no_bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_intervention_args(args)
    if args.discover:
        discover_and_write(args)
        return
    if args.job_index is None:
        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_index is None:
            raise ValueError("--job_index is required unless SLURM_ARRAY_TASK_ID is set")
        args.job_index = int(env_index)
    run_manifest_job(args)


def discover_and_write(args: argparse.Namespace) -> None:
    tasks = validate_tasks(args.tasks)
    trajectories = discover_trajectories(
        roots=[Path(root) for root in args.discovery_roots],
        trajectory_filter=args.trajectory_filter,
        model_filter=args.model_filter,
        max_checkpoints_per_trajectory=args.max_checkpoints_per_trajectory,
    )
    print_discovery(trajectories)
    jobs = build_jobs(trajectories, tasks, Path(args.output_root), Path(args.shared_output_root))
    report = {
        "discovery_roots": args.discovery_roots,
        "tasks": tasks,
        "num_trajectories": len(trajectories),
        "num_jobs": len(jobs),
        "trajectories": trajectories,
        "manifest_path": args.manifest_path,
    }
    report_path = Path(args.discovery_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    if not args.dry_run_discovery:
        manifest_path = Path(args.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(manifest_path, jobs)
        print(f"Wrote job manifest: {manifest_path} ({len(jobs)} jobs)")
    else:
        print("Dry run: manifest not written")


def validate_tasks(tasks: list[str]) -> list[str]:
    cleaned = []
    for task in tasks:
        if task not in TASK_PATHS:
            raise ValueError(f"Unknown task {task!r}. Valid tasks: {sorted(TASK_PATHS)}")
        for split, path in TASK_PATHS[task].items():
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing {task}/{split}: {path}")
        cleaned.append(task)
    return cleaned


def discover_trajectories(
    roots: list[Path],
    trajectory_filter: str | None,
    model_filter: str | None,
    max_checkpoints_per_trajectory: int | None,
) -> list[dict[str, Any]]:
    trajectories = []
    seen_reports = set()
    for root in roots:
        if not root.exists():
            continue
        for report_path in root.rglob("cpt_plasticity_report.json"):
            resolved = str(report_path.resolve())
            if resolved in seen_reports:
                continue
            seen_reports.add(resolved)
            run_dir = report_path.parent
            trajectory_id = f"{run_dir.parent.name}__{run_dir.name}"
            if trajectory_filter and trajectory_filter not in trajectory_id:
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            model_name = infer_model_name(report, run_dir)
            if model_filter and model_filter not in model_name and model_filter not in trajectory_id:
                continue
            checkpoints = discover_checkpoints(run_dir)
            if not checkpoints:
                continue
            if max_checkpoints_per_trajectory is not None:
                checkpoints = checkpoints[:max_checkpoints_per_trajectory]
            trajectory = {
                "trajectory_id": sanitize_id(trajectory_id),
                "run_name": run_dir.parent.name,
                "run_dir": str(run_dir),
                "report_path": str(report_path),
                "model_name": model_name,
                "cpt_hparams": extract_cpt_hparams(report),
                "checkpoints": [
                    {"cpt_tokens": 0, "init_checkpoint": None, "label": "base"},
                    *checkpoints,
                ],
            }
            trajectories.append(trajectory)
    trajectories.sort(key=lambda row: (row["model_name"], row["trajectory_id"]))
    return trajectories


def infer_model_name(report: Mapping[str, Any], run_dir: Path) -> str:
    model_name = report.get("model_name")
    if model_name:
        return str(model_name)
    config_path = run_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("_name_or_path"):
            return str(config["_name_or_path"])
        if config.get("architectures"):
            return str(config["architectures"][0])
    raise RuntimeError(f"Cannot infer model_name for {run_dir}")


def extract_cpt_hparams(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "learning_rate",
        "weight_decay",
        "lr_scheduler_type",
        "resolved_lr_scheduler_type",
        "warmup_tokens",
        "warmup_steps",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "max_training_tokens",
        "actual_training_tokens",
        "cpt_data_path",
    )
    return {key: report.get(key) for key in keys if key in report}


def discover_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    checkpoints = []
    for checkpoint_report in run_dir.glob("checkpoint_*_tokens/checkpoint_report.json"):
        checkpoint_dir = checkpoint_report.parent
        if not (checkpoint_dir / "config.json").is_file():
            continue
        report = json.loads(checkpoint_report.read_text(encoding="utf-8"))
        cpt_tokens = int(report.get("threshold_tokens") or report.get("cumulative_training_tokens") or parse_tokens_from_name(checkpoint_dir.name))
        checkpoints.append(
            {
                "cpt_tokens": cpt_tokens,
                "init_checkpoint": str(checkpoint_dir),
                "label": checkpoint_dir.name,
                "cumulative_training_tokens": int(report.get("cumulative_training_tokens", cpt_tokens)),
                "optimizer_step": int(report.get("optimizer_step", 0)),
            }
        )
    checkpoints.sort(key=lambda row: row["cpt_tokens"])
    return checkpoints


def parse_tokens_from_name(name: str) -> int:
    match = re.search(r"checkpoint_(\d+)([KMB]?)_tokens", name)
    if not match:
        raise ValueError(f"Cannot parse token count from checkpoint dir name {name!r}")
    value = int(match.group(1))
    suffix = match.group(2)
    return value * {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[suffix]


def build_jobs(
    trajectories: list[dict[str, Any]],
    tasks: list[str],
    output_root: Path,
    shared_output_root: Path,
) -> list[dict[str, Any]]:
    jobs = []
    for trajectory in trajectories:
        for checkpoint in trajectory["checkpoints"]:
            for task in tasks:
                label = checkpoint["label"]
                output_dir = output_root / trajectory["trajectory_id"] / task / label
                shared_output_dir = shared_output_root / trajectory["trajectory_id"] / task / label
                jobs.append(
                    {
                        "job_id": len(jobs),
                        "trajectory_id": trajectory["trajectory_id"],
                        "run_name": trajectory["run_name"],
                        "model_name": trajectory["model_name"],
                        "cpt_hparams": trajectory["cpt_hparams"],
                        "init_checkpoint": checkpoint["init_checkpoint"],
                        "cpt_tokens": checkpoint["cpt_tokens"],
                        "checkpoint_label": label,
                        "task": task,
                        "train_path": TASK_PATHS[task]["train"],
                        "probe_path": TASK_PATHS[task]["probe"],
                        "output_dir": str(output_dir),
                        "shared_output_dir": str(shared_output_dir),
                        "wandb_group": f"step6/{trajectory['trajectory_id']}/{task}",
                        "wandb_run_name": f"{task}_{label}",
                    }
                )
    return jobs


def print_discovery(trajectories: list[dict[str, Any]]) -> None:
    print("Discovered CPT trajectories:")
    for trajectory in trajectories:
        tokens = [row["cpt_tokens"] for row in trajectory["checkpoints"]]
        print(f"- {trajectory['trajectory_id']}")
        print(f"  model: {trajectory['model_name']}")
        print(f"  run_dir: {trajectory['run_dir']}")
        print(f"  checkpoints: {tokens}")


def run_manifest_job(args: argparse.Namespace) -> None:
    manifest_path = Path(args.job_manifest)
    jobs = read_jsonl(manifest_path)
    if args.job_index < 0 or args.job_index >= len(jobs):
        raise IndexError(f"job_index {args.job_index} out of range for {len(jobs)} jobs")
    job = jobs[args.job_index]
    if intervention_enabled(args):
        job = with_step7_job_names(job, intervention_name(args))
        job["intervention_strategy"] = intervention_strategy(args)
        job["reset_embedding_and_readout_to_base_tied"] = bool(args.reset_embedding_and_readout_to_base_tied)
        job["reset_embedding_readout_and_last_layers_to_base_tied"] = bool(args.reset_embedding_readout_and_last_layers_to_base_tied)
        job["reset_last_l_layers"] = int(args.reset_last_l_layers) if args.reset_embedding_readout_and_last_layers_to_base_tied else None
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    wandb_run = init_wandb(args, job)
    use_bf16 = bf16_supported() and not args.no_bf16
    print_json("package_versions", package_versions())
    print_json("job_spec", job)
    print(f"bf16: {use_bf16}")

    model, tokenizer = load_model_for_job(job, use_bf16)
    intervention_report = apply_readout_intervention_if_requested(model, job, args, use_bf16)
    print_json("readout_intervention", intervention_report.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_examples = load_encoded_split(job["train_path"], tokenizer, args.max_seq_length, args.seed)
    probe_examples = load_encoded_split(job["probe_path"], tokenizer, args.max_seq_length, args.seed)
    train_dataset = EncodedSftDataset(train_examples)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: collate_sft_batch(batch, tokenizer, device),
        drop_last=False,
    )

    total_update_steps = max(1, math.ceil(len(train_loader) * args.num_train_epochs / args.gradient_accumulation_steps))
    warmup_steps = math.ceil(args.warmup_ratio * total_update_steps)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    writer = LocalMetricsWriter(output_dir / "step6_sft_metrics.csv", output_dir / "step6_sft_metrics.jsonl")

    state = {
        "sft_global_step": 0,
        "sft_tokens_seen": 0,
        "epoch": 0.0,
        "last_train_loss": None,
    }
    evaluate_and_log(
        model=model,
        tokenizer=tokenizer,
        job=job,
        args=args,
        train_examples=train_examples,
        probe_examples=probe_examples,
        state=state,
        event="step0_eval",
        writer=writer,
    )

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    target_micro_steps = max(1, math.ceil(args.num_train_epochs * len(train_loader)))
    completed_micro_steps = 0
    epoch_index = 0
    while completed_micro_steps < target_micro_steps:
        for batch in train_loader:
            if completed_micro_steps >= target_micro_steps:
                break
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            raw_loss = float(outputs.loss.detach().float().item())
            response_tokens = int((batch["labels"] != -100).sum().item())
            state["sft_tokens_seen"] += response_tokens
            completed_micro_steps += 1
            micro_step += 1
            state["epoch"] = epoch_index + ((completed_micro_steps % max(1, len(train_loader))) / max(1, len(train_loader)))
            state["last_train_loss"] = raw_loss

            if micro_step % args.gradient_accumulation_steps == 0:
                grad_norm = clip_grad_norm(model, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state["sft_global_step"] += 1
                log_train_update(job, state, raw_loss, float(scheduler.get_last_lr()[0]), grad_norm)
                if state["sft_global_step"] % args.eval_steps == 0:
                    evaluate_and_log(
                        model=model,
                        tokenizer=tokenizer,
                        job=job,
                        args=args,
                        train_examples=train_examples,
                        probe_examples=probe_examples,
                        state=state,
                        event="scheduled_eval",
                        writer=writer,
                    )
        epoch_index += 1

    if micro_step > 0 and micro_step % args.gradient_accumulation_steps != 0:
        grad_norm = clip_grad_norm(model, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        state["sft_global_step"] += 1
        log_train_update(job, state, float(state["last_train_loss"] or math.nan), float(scheduler.get_last_lr()[0]), grad_norm)

    evaluate_and_log(
        model=model,
        tokenizer=tokenizer,
        job=job,
        args=args,
        train_examples=train_examples,
        probe_examples=probe_examples,
        state=state,
        event="final_eval",
        writer=writer,
    )
    report = {
        "job": job,
        "sft_hparams": sft_hparams(args, total_update_steps, warmup_steps),
        "readout_intervention": intervention_report.__dict__,
        "final_state": state,
        "gpu": gpu_report().__dict__,
    }
    write_json(output_dir / "step6_sft_report.json", report)
    shared_output_dir = Path(job["shared_output_dir"])
    shared_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(shared_output_dir / "step6_sft_report.json", report)
    for filename in ("step6_sft_metrics.csv", "step6_sft_metrics.jsonl"):
        src = output_dir / filename
        if src.is_file():
            (shared_output_dir / filename).write_bytes(src.read_bytes())
    if wandb_run is not None:
        wandb_run.finish()


def load_model_for_job(job: Mapping[str, Any], use_bf16: bool):
    source = job.get("init_checkpoint") or job["model_name"]
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    return model, tokenizer


def validate_intervention_args(args: argparse.Namespace) -> None:
    enabled = [
        args.rewind_readout_to_base,
        args.reset_embedding_and_readout_to_base_tied,
        args.reset_embedding_readout_and_last_layers_to_base_tied,
    ]
    if sum(bool(value) for value in enabled) > 1:
        raise ValueError(
            "Choose only one intervention: --rewind_readout_to_base, "
            "--reset_embedding_and_readout_to_base_tied, or "
            "--reset_embedding_readout_and_last_layers_to_base_tied"
        )
    if args.reset_last_l_layers < 1:
        raise ValueError("--reset_last_l_layers must be >= 1; L=1 means embedding/readout reset only")


def intervention_enabled(args: argparse.Namespace) -> bool:
    return bool(
        args.rewind_readout_to_base
        or args.reset_embedding_and_readout_to_base_tied
        or args.reset_embedding_readout_and_last_layers_to_base_tied
    )


def intervention_name(args: argparse.Namespace) -> str:
    if args.intervention_name:
        return args.intervention_name
    if args.reset_embedding_readout_and_last_layers_to_base_tied:
        return f"base_embedding_readout_tied_last{args.reset_last_l_layers}_trainable"
    if args.reset_embedding_and_readout_to_base_tied:
        return "base_embedding_readout_tied_trainable"
    return "base_readout_trainable"


def intervention_strategy(args: argparse.Namespace) -> str:
    if args.reset_embedding_readout_and_last_layers_to_base_tied:
        return "embedding_readout_tied_last_layers"
    if args.reset_embedding_and_readout_to_base_tied:
        return "embedding_readout_tied"
    if args.rewind_readout_to_base:
        return "readout_only"
    return "none"


def with_step7_job_names(job: Mapping[str, Any], intervention_name_value: str) -> dict[str, Any]:
    step7_job = dict(job)
    suffix = sanitize_id(intervention_name_value)
    step7_job["output_dir"] = f"{job['output_dir']}__{suffix}"
    step7_job["shared_output_dir"] = f"{job['shared_output_dir']}__{suffix}"
    step7_job["wandb_group"] = step7_wandb_group(job, suffix)
    step7_job["wandb_run_name"] = f"{job['wandb_run_name']}_{suffix}"
    step7_job["intervention_name"] = suffix
    step7_job["rewind_readout_to_base"] = True
    step7_job["intervention_strategy"] = "step7"
    return step7_job


def step7_wandb_group(job: Mapping[str, Any], suffix: str) -> str:
    model_tag = sanitize_id(str(job["model_name"]).split("/")[-1])
    digest = hashlib.sha1(str(job["trajectory_id"]).encode("utf-8")).hexdigest()[:8]
    group = f"step7/{job['task']}/{suffix}/{model_tag}-{digest}"
    if len(group) > 128:
        group = f"step7/{job['task']}/{suffix[:40]}/{model_tag[:32]}-{digest}"
    return group


def apply_readout_intervention_if_requested(
    model: torch.nn.Module,
    job: Mapping[str, Any],
    args: argparse.Namespace,
    use_bf16: bool,
) -> ReadoutInterventionReport:
    output_weight = get_lm_head_weight(model)
    input_weight = get_input_embedding_weight(model)
    if not intervention_enabled(args):
        tied = embeddings_are_tied(model)
        return ReadoutInterventionReport(
            enabled=False,
            strategy="none",
            base_model_name=None,
            embeddings_tied_before=tied,
            embeddings_tied_after=tied,
            input_weight_shape=tuple(input_weight.shape) if input_weight is not None else None,
            output_weight_shape=tuple(output_weight.shape),
            input_weight_trainable=input_weight.requires_grad if input_weight is not None else None,
            output_weight_trainable=output_weight.requires_grad,
            untied_output_embeddings=False,
            reset_last_l_layers=None,
            reset_transformer_layers=[],
        )

    base_model_name = str(job["model_name"])
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, attn_implementation="sdpa")
    try:
        before_tied = embeddings_are_tied(model)
        reset_transformer_layers: list[int] = []
        if args.reset_embedding_readout_and_last_layers_to_base_tied:
            untied_output_embeddings = apply_embedding_readout_tied_reset(model, base_model, output_weight.dtype)
            reset_transformer_layers = apply_top_transformer_layer_reset(model, base_model, args.reset_last_l_layers - 1)
        elif args.reset_embedding_and_readout_to_base_tied:
            untied_output_embeddings = apply_embedding_readout_tied_reset(model, base_model, output_weight.dtype)
        else:
            untied_output_embeddings = apply_readout_only_reset(model, base_model, output_weight.dtype)

        output_weight = get_lm_head_weight(model)
        input_weight = get_input_embedding_weight(model)
        return ReadoutInterventionReport(
            enabled=True,
            strategy=intervention_strategy(args),
            base_model_name=base_model_name,
            embeddings_tied_before=before_tied,
            embeddings_tied_after=embeddings_are_tied(model),
            input_weight_shape=tuple(input_weight.shape) if input_weight is not None else None,
            output_weight_shape=tuple(output_weight.shape),
            input_weight_trainable=input_weight.requires_grad if input_weight is not None else None,
            output_weight_trainable=output_weight.requires_grad,
            untied_output_embeddings=untied_output_embeddings,
            reset_last_l_layers=int(args.reset_last_l_layers) if args.reset_embedding_readout_and_last_layers_to_base_tied else None,
            reset_transformer_layers=reset_transformer_layers,
        )
    finally:
        del base_model


def apply_readout_only_reset(model: torch.nn.Module, base_model: torch.nn.Module, dtype: torch.dtype) -> bool:
    base_readout = get_lm_head_weight(base_model).detach().to(dtype=dtype, device="cpu")
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise AttributeError("Expected model.get_output_embeddings().weight for readout intervention.")
    output_weight = get_lm_head_weight(model)
    if tuple(output_weight.shape) != tuple(base_readout.shape):
        raise ValueError(f"Base readout shape {tuple(base_readout.shape)} does not match target readout shape {tuple(output_weight.shape)}")

    if embeddings_are_tied(model):
        replacement = build_output_head(output_embeddings, base_readout)
        model.set_output_embeddings(replacement)
        if hasattr(model, "config"):
            model.config.tie_word_embeddings = False
        return True

    output_weight.data.copy_(base_readout.to(device=output_weight.device, dtype=output_weight.dtype))
    output_weight.requires_grad_(True)
    bias = getattr(output_embeddings, "bias", None)
    if bias is not None:
        bias.requires_grad_(True)
    return False


def apply_embedding_readout_tied_reset(model: torch.nn.Module, base_model: torch.nn.Module, dtype: torch.dtype) -> bool:
    input_weight = get_input_embedding_weight(model)
    base_input_weight = get_input_embedding_weight(base_model)
    if input_weight is None or base_input_weight is None:
        raise AttributeError("Expected model input embeddings for tied embedding/readout reset.")
    if tuple(input_weight.shape) != tuple(base_input_weight.shape):
        raise ValueError(
            f"Base input embedding shape {tuple(base_input_weight.shape)} does not match target shape {tuple(input_weight.shape)}"
        )

    base_input_weight = base_input_weight.detach().to(device=input_weight.device, dtype=input_weight.dtype)
    input_weight.data.copy_(base_input_weight)
    input_weight.requires_grad_(True)

    if hasattr(model, "config"):
        model.config.tie_word_embeddings = True
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    if not embeddings_are_tied(model):
        tie_output_to_input_embeddings(model)
    if not embeddings_are_tied(model):
        raise RuntimeError("Failed to keep input embeddings and output readout tied after reset.")

    output_weight = get_lm_head_weight(model)
    output_weight.requires_grad_(True)
    return False


def apply_top_transformer_layer_reset(model: torch.nn.Module, base_model: torch.nn.Module, num_top_layers: int) -> list[int]:
    if num_top_layers <= 0:
        return []
    layers = get_transformer_layers(model)
    base_layers = get_transformer_layers(base_model)
    if len(layers) != len(base_layers):
        raise ValueError(f"Target has {len(layers)} layers but base has {len(base_layers)} layers")
    if num_top_layers > len(layers):
        raise ValueError(f"Cannot reset {num_top_layers} top transformer layers; model only has {len(layers)} layers")

    reset_indices = list(range(len(layers) - num_top_layers, len(layers)))
    for index in reset_indices:
        layers[index].load_state_dict(base_layers[index].state_dict())
        for parameter in layers[index].parameters():
            parameter.requires_grad_(True)
    return reset_indices


def get_transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList | list[torch.nn.Module]:
    candidates = (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("decoder", "layers"),
    )
    for path in candidates:
        value: Any = model
        for attr in path:
            value = getattr(value, attr, None)
            if value is None:
                break
        if value is not None and isinstance(value, (nn.ModuleList, list, tuple)) and len(value) > 0:
            return value
    raise AttributeError(
        "Could not locate transformer blocks. Tried: "
        + ", ".join(".".join(path) for path in candidates)
    )


def get_input_embedding_weight(model: torch.nn.Module) -> torch.Tensor | None:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        return None
    return input_embeddings.weight


def tie_output_to_input_embeddings(model: torch.nn.Module) -> None:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise AttributeError("Expected both input and output embeddings to tie weights.")
    if not hasattr(input_embeddings, "weight") or not hasattr(output_embeddings, "weight"):
        raise AttributeError("Expected input/output embeddings with .weight to tie weights.")
    output_embeddings.weight = input_embeddings.weight


def embeddings_are_tied(model: torch.nn.Module) -> bool:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    return bool(
        input_embeddings is not None
        and output_embeddings is not None
        and hasattr(input_embeddings, "weight")
        and hasattr(output_embeddings, "weight")
        and input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr()
    )


def build_output_head(output_embeddings: torch.nn.Module, base_readout: torch.Tensor) -> nn.Linear:
    bias = getattr(output_embeddings, "bias", None)
    replacement = nn.Linear(
        in_features=base_readout.shape[1],
        out_features=base_readout.shape[0],
        bias=bias is not None,
        dtype=base_readout.dtype,
    )
    replacement.weight.data.copy_(base_readout)
    replacement.weight.requires_grad_(True)
    if bias is not None and replacement.bias is not None:
        replacement.bias.data.copy_(bias.detach().to(dtype=replacement.bias.dtype, device=replacement.bias.device))
        replacement.bias.requires_grad_(True)
    return replacement


def load_encoded_split(path: str, tokenizer: PreTrainedTokenizerBase, max_seq_length: int, seed: int) -> list[EncodedExample]:
    dataset = load_instruction_dataset(path, "train", None, seed)
    prompt_completion = to_prompt_completion_dataset(dataset, tokenizer)
    prompt_completion = filter_prompt_completion_for_context(prompt_completion, tokenizer, max_seq_length)
    examples = []
    for row in prompt_completion:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        input_ids = (list(prompt_ids) + list(completion_ids))[:max_seq_length]
        response_mask = ([0] * len(prompt_ids) + [1] * len(completion_ids))[:max_seq_length]
        if sum(response_mask) > 0:
            examples.append(EncodedExample(input_ids=input_ids, response_mask=response_mask))
    if not examples:
        raise RuntimeError(f"No usable examples in {path}")
    return examples


def collate_sft_batch(batch: list[EncodedExample], tokenizer: PreTrainedTokenizerBase, device: torch.device) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    max_len = max(len(example.input_ids) for example in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long, device=device)
    response_mask = torch.zeros((len(batch), max_len), dtype=torch.bool, device=device)
    for row, example in enumerate(batch):
        length = len(example.input_ids)
        ids = torch.tensor(example.input_ids, dtype=torch.long, device=device)
        mask = torch.tensor(example.response_mask, dtype=torch.bool, device=device)
        input_ids[row, :length] = ids
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.where(mask, ids, torch.full_like(ids, -100))
        response_mask[row, :length] = mask
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "response_mask": response_mask}


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[EncodedExample],
    batch_size: int,
    block_size: int,
) -> dict[str, float]:
    was_training = bool(model.training)
    model.eval()
    try:
        device = next(model.parameters()).device
        lm_head_weight = get_lm_head_weight(model).detach()
        total_nll = 0.0
        total_tokens = 0
        total_examples = 0
        g2_sum = 0.0
        wtg2_sum = 0.0
        r_values: list[float] = []
        for start in range(0, len(examples), batch_size):
            batch = collate_sft_batch(examples[start : start + batch_size], tokenizer, device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits[:, :-1, :].float()
            targets = batch["input_ids"][:, 1:]
            target_mask = batch["response_mask"][:, 1:] & batch["attention_mask"][:, 1:].bool()
            if not target_mask.any():
                continue
            selected_logits = logits[target_mask]
            selected_targets = targets[target_mask]
            log_probs = torch.log_softmax(selected_logits, dim=-1)
            nll = -log_probs.gather(1, selected_targets[:, None]).squeeze(1)
            metrics = score_selected_tokens(selected_logits, selected_targets, lm_head_weight, block_size)
            token_count = int(selected_targets.numel())
            total_nll += float(nll.sum().item())
            total_tokens += token_count
            total_examples += len(examples[start : start + batch_size])
            g2_sum += metrics["g2_sum"]
            wtg2_sum += metrics["wtg2_sum"]
            r_values.extend(metrics["r_values"])
        if total_tokens == 0:
            raise RuntimeError("Evaluation produced zero supervised tokens")
        r_sorted = sorted(r_values)
        return {
            "loss": total_nll / total_tokens,
            "g2_mean": g2_sum / total_tokens,
            "wtg2_mean": wtg2_sum / total_tokens,
            "R_mean": sum(r_values) / len(r_values) if r_values else math.nan,
            "R_p95": percentile(r_sorted, 0.95),
            "num_tokens": float(total_tokens),
            "num_examples": float(total_examples),
        }
    finally:
        if was_training:
            model.train()


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
        block_logits = selected_logits[start : start + block_size]
        block_targets = selected_targets[start : start + block_size]
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


def evaluate_and_log(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    job: Mapping[str, Any],
    args: argparse.Namespace,
    train_examples: list[EncodedExample],
    probe_examples: list[EncodedExample],
    state: Mapping[str, Any],
    event: str,
    writer: "LocalMetricsWriter",
) -> None:
    rows = []
    base_payload = common_payload(job, state, event)
    wandb_payload: dict[str, Any] = dict(base_payload)
    for split, examples in (("train", train_examples), ("probe", probe_examples)):
        metrics = evaluate_split(model, tokenizer, examples, args.eval_batch_size, args.plasticity_block_size)
        rows.append({**base_payload, "split": split, **metrics})
        for key, value in metrics.items():
            wandb_payload[f"{split}/{key}"] = value
    writer.append(rows)
    print_json("eval_metrics", wandb_payload)
    wandb_log(wandb_payload, int(state["sft_global_step"]))


def log_train_update(job: Mapping[str, Any], state: Mapping[str, Any], loss: float, learning_rate: float, grad_norm: float) -> None:
    payload = common_payload(job, state, "train_update")
    payload.update({"train/update_loss": loss, "train/learning_rate": learning_rate, "train/grad_norm": grad_norm})
    if int(state["sft_global_step"]) % 1 == 0:
        wandb_log(payload, int(state["sft_global_step"]))


def common_payload(job: Mapping[str, Any], state: Mapping[str, Any], event: str) -> dict[str, Any]:
    return {
        "event": event,
        "trajectory_id": job["trajectory_id"],
        "run_name": job["run_name"],
        "model_name": job["model_name"],
        "task": job["task"],
        "cpt_tokens": int(job["cpt_tokens"]),
        "checkpoint_label": job["checkpoint_label"],
        "init_checkpoint": job.get("init_checkpoint") or "base",
        "rewind_readout_to_base": bool(job.get("rewind_readout_to_base", False)),
        "reset_embedding_and_readout_to_base_tied": bool(job.get("reset_embedding_and_readout_to_base_tied", False)),
        "reset_embedding_readout_and_last_layers_to_base_tied": bool(job.get("reset_embedding_readout_and_last_layers_to_base_tied", False)),
        "reset_last_l_layers": int(job["reset_last_l_layers"]) if job.get("reset_last_l_layers") is not None else 0,
        "intervention_name": job.get("intervention_name") or "none",
        "intervention_strategy": job.get("intervention_strategy") or "none",
        "sft_global_step": int(state["sft_global_step"]),
        "sft_tokens_seen": int(state["sft_tokens_seen"]),
        "epoch": float(state["epoch"]),
    }


def clip_grad_norm(model: torch.nn.Module, max_grad_norm: float) -> float:
    if max_grad_norm <= 0:
        return math.nan
    value = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    return float(value.detach().float().item())


def init_wandb(args: argparse.Namespace, job: Mapping[str, Any]) -> Any | None:
    try:
        import wandb
    except ImportError:
        print("wandb unavailable; continuing without WandB logging")
        return None
    run = wandb.init(
        project=args.wandb_project,
        group=job["wandb_group"],
        name=job["wandb_run_name"],
        config={"job": dict(job), "sft_hparams": sft_hparams(args)},
    )
    wandb.define_metric("sft_global_step")
    wandb.define_metric("train/*", step_metric="sft_global_step")
    wandb.define_metric("probe/*", step_metric="sft_global_step")
    return run


def sft_hparams(args: argparse.Namespace, total_update_steps: int | None = None, warmup_steps: int | None = None) -> dict[str, Any]:
    return {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_batch_size": args.eval_batch_size,
        "eval_steps": args.eval_steps,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "total_update_steps": total_update_steps,
        "rewind_readout_to_base": args.rewind_readout_to_base,
        "reset_embedding_and_readout_to_base_tied": args.reset_embedding_and_readout_to_base_tied,
        "reset_embedding_readout_and_last_layers_to_base_tied": args.reset_embedding_readout_and_last_layers_to_base_tied,
        "reset_last_l_layers": args.reset_last_l_layers if args.reset_embedding_readout_and_last_layers_to_base_tied else None,
        "intervention_name": intervention_name(args) if intervention_enabled(args) else None,
        "intervention_strategy": intervention_strategy(args),
    }


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


def sanitize_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "unknown"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def print_json(label: str, value: object) -> None:
    print(f"{label}: {json.dumps(value, indent=2, sort_keys=True)}")


def wandb_log(payload: dict[str, Any], step: int) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log(payload, step=step)


if __name__ == "__main__":
    main()
