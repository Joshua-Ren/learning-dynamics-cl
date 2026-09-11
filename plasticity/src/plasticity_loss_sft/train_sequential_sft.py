from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments, set_seed
from trl import SFTConfig, SFTTrainer

from plasticity_loss_sft.callbacks import CustomSaveCallback, LossHistoryCallback
from plasticity_loss_sft.data import (
    filter_prompt_completion_for_context,
    load_instruction_dataset,
    to_prompt_completion_dataset,
)
from plasticity_loss_sft.modeling import (
    SUPPORTED_MODELS,
    assert_assistant_mask_supported,
    inspect_model_access,
    load_model_and_tokenizer,
    supports_assistant_token_mask,
)
from plasticity_loss_sft.probe_confidence_callback import ProbeConfidenceWandbCallback, parse_probe_specs
from plasticity_loss_sft.readout_geometry_callback import ReadoutGeometryWandbCallback
from plasticity_loss_sft.runtime import bf16_supported, gpu_report, package_versions, reset_peak_memory


DEFAULT_TASK_SEQUENCE = (
    ("gsm8k", "data/prepared_subsets/gsm8k/train.jsonl"),
    ("mbpp", "data/prepared_subsets/mbpp/train.jsonl"),
    ("dolly_qa", "data/prepared_subsets/dolly_qa/train.jsonl"),
)


@dataclass(frozen=True)
class SequentialTask:
    name: str
    dataset_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long sequential full-parameter SFT entry point.")
    parser.add_argument("--model_name", choices=SUPPORTED_MODELS, default=SUPPORTED_MODELS[0])
    parser.add_argument(
        "--task_sequence",
        nargs="+",
        default=[f"{name}={path}" for name, path in DEFAULT_TASK_SEQUENCE],
        help="Ordered task specs as task_name=dataset_jsonl.",
    )
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--output_dir", default="outputs/sequential_sft")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--prepare_dataset_name", default="GAIR/lima")
    parser.add_argument("--num_prepare_train_epochs", type=float, default=5.0)
    parser.add_argument("--prepare_dataset_limit", type=int, default=None)
    parser.add_argument("--skip_prepare_stage", action="store_true")
    parser.add_argument("--num_train_epochs_per_task", type=float, default=100.0)
    parser.add_argument("--num_task_rounds", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--custom_save_steps", type=int, default=0)
    parser.add_argument("--no_save_checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wandb_project", default="plasticity-loss")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--skip_assistant_mask_check", action="store_true")
    parser.add_argument("--readout_geometry_support_files", nargs="+", default=[])
    parser.add_argument("--readout_geometry_logging_steps", type=int, default=0)
    parser.add_argument("--probe_eval_specs", nargs="+", default=[])
    parser.add_argument("--probe_eval_steps", type=int, default=0)
    parser.add_argument("--probe_eval_batch_size", type=int, default=2)
    parser.add_argument("--probe_eval_limit", type=int, default=None)
    parser.add_argument("--probe_plasticity_block_size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = parse_task_sequence(args.task_sequence)
    if args.num_task_rounds <= 0:
        raise ValueError("--num_task_rounds must be positive.")
    task_segments = build_task_segments(tasks, args.num_task_rounds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_run_name:
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)
    set_seed(args.seed)
    wandb_run = init_wandb(args, tasks, task_segments)

    use_bf16 = bf16_supported() and not args.no_bf16
    print_json("package_versions", package_versions())
    print(f"bf16: {use_bf16}")

    model, tokenizer = load_model_and_tokenizer(args.model_name, use_bf16)
    has_assistant_mask = supports_assistant_token_mask(tokenizer)
    if not args.skip_assistant_mask_check and has_assistant_mask:
        assert_assistant_mask_supported(tokenizer)
    if has_assistant_mask:
        assistant_only_loss = True
        completion_only_loss = None
        loss_masking_mode = "assistant_only_loss"
    else:
        assistant_only_loss = False
        completion_only_loss = True
        loss_masking_mode = "native_prompt_completion_loss"
    print(f"loss_masking_mode: {loss_masking_mode}")
    print_json("model_access", inspect_model_access(model).__dict__)
    write_base_reference(output_dir, args.model_name)
    reset_peak_memory()

    prepare_report = None
    if not args.skip_prepare_stage and args.num_prepare_train_epochs > 0:
        prepare_result = run_prepare_stage(
            args=args,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            use_bf16=use_bf16,
            assistant_only_loss=assistant_only_loss,
            completion_only_loss=completion_only_loss,
            has_assistant_mask=has_assistant_mask,
        )
        model = prepare_result.pop("model")
        prepare_report = prepare_result

    readout_geometry_callback = None
    if args.readout_geometry_support_files:
        readout_geometry_callback = ReadoutGeometryWandbCallback(
            support_files=args.readout_geometry_support_files,
            logging_steps=args.readout_geometry_logging_steps,
        )
    probe_confidence_callback = None
    if args.probe_eval_specs:
        probe_confidence_callback = ProbeConfidenceWandbCallback(
            probe_tasks=parse_probe_specs(args.probe_eval_specs),
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            eval_steps=args.probe_eval_steps,
            batch_size=args.probe_eval_batch_size,
            limit=args.probe_eval_limit,
            seed=args.seed,
            output_dir=args.output_dir,
            plasticity_block_size=args.probe_plasticity_block_size,
        )

    sequential_global_step_offset = 0
    wandb_step_offset = int(prepare_report["prepare_global_steps"]) if prepare_report is not None else 0
    task_reports = []
    task_chain: list[str] = []

    for segment_index, task in enumerate(task_segments, start=1):
        task_chain.append(task.name)
        task_output_dir = output_dir / f"{segment_index:02d}_{'_then_'.join(task_chain)}"
        task_output_dir.mkdir(parents=True, exist_ok=True)
        round_index = ((segment_index - 1) // len(tasks)) + 1
        task_index_in_round = ((segment_index - 1) % len(tasks)) + 1
        print(
            f"Starting segment {segment_index}/{len(task_segments)}: "
            f"round {round_index}/{args.num_task_rounds}, "
            f"task {task_index_in_round}/{len(tasks)}: {task.name}"
        )
        print(f"Task output dir: {task_output_dir}")

        dataset = build_training_dataset(
            dataset_name=task.dataset_path,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
            limit=args.dataset_limit,
            has_assistant_mask=has_assistant_mask,
        )
        print(f"{task.name}: train_examples={len(dataset)}")

        loss_history = LossHistoryCallback()
        training_args = SFTConfig(
            output_dir=str(task_output_dir),
            max_length=args.max_seq_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs_per_task,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            optim="adamw_torch",
            seed=args.seed,
            data_seed=args.seed,
            bf16=use_bf16,
            fp16=False,
            assistant_only_loss=assistant_only_loss,
            completion_only_loss=completion_only_loss,
            packing=False,
            logging_steps=args.logging_steps,
            save_strategy="no",
            save_total_limit=args.save_total_limit,
            report_to=[],
            run_name=args.wandb_run_name,
            remove_unused_columns=False,
        )
        task_callback = SequentialTaskLoggingCallback(
            task_name=task.name,
            task_index=task_index_in_round,
            round_index=round_index,
            segment_index=segment_index,
            sequential_global_step_offset=sequential_global_step_offset,
            wandb_step_offset=wandb_step_offset,
            epochs_per_task=args.num_train_epochs_per_task,
        )
        callbacks: list[TrainerCallback] = [
            task_callback,
            loss_history,
        ]
        if not args.no_save_checkpoints:
            callbacks.append(CustomSaveCallback(str(task_output_dir), args.custom_save_steps))
        if readout_geometry_callback is not None:
            readout_geometry_callback.set_context(
                continuous_epoch_offset=(segment_index - 1) * args.num_train_epochs_per_task,
                sequential_global_step_offset=sequential_global_step_offset,
                wandb_step_offset=wandb_step_offset,
                task_name=task.name,
                task_index=task_index_in_round,
                task_round=round_index,
                task_segment_index=segment_index,
            )
            callbacks.append(readout_geometry_callback)
        if probe_confidence_callback is not None:
            probe_confidence_callback.set_context(
                continuous_epoch_offset=(segment_index - 1) * args.num_train_epochs_per_task,
                sequential_global_step_offset=sequential_global_step_offset,
                wandb_step_offset=wandb_step_offset,
                task_name=task.name,
                task_index=task_index_in_round,
                task_round=round_index,
                task_segment_index=segment_index,
            )
            callbacks.append(probe_confidence_callback)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        train_output = trainer.train()
        if not args.no_save_checkpoints:
            trainer.save_model(str(task_output_dir))
            trainer.save_state()

        sequential_global_step_offset += int(trainer.state.global_step)
        task_report = {
            "segment_index": segment_index,
            "round_index": round_index,
            "task_index_in_round": task_index_in_round,
            "task": task.name,
            "dataset_path": task.dataset_path,
            "train_examples": len(dataset),
            "output_dir": str(task_output_dir),
            "task_global_steps": int(trainer.state.global_step),
            "sequential_global_step_end": sequential_global_step_offset,
            "train_metrics": train_output.metrics,
            "loss_history": loss_history.losses,
        }
        task_reports.append(task_report)
        write_json(task_output_dir / "sequential_task_report.json", task_report)
        print_json(f"{task.name}_train_metrics", train_output.metrics)

        model = trainer.model
        del trainer

    final_report = {
        "model_name": args.model_name,
        "output_dir": str(output_dir),
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "loss_masking_mode": loss_masking_mode,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs_per_task": args.num_train_epochs_per_task,
        "num_task_rounds": args.num_task_rounds,
        "save_checkpoints": not args.no_save_checkpoints,
        "prepare_stage": prepare_report,
        "tasks": task_reports,
        "final_checkpoint_dir": task_reports[-1]["output_dir"] if task_reports and not args.no_save_checkpoints else None,
        "gpu": gpu_report().__dict__,
    }
    write_json(output_dir / "sequential_sft_report.json", final_report)
    print_json("gpu", final_report["gpu"])
    print(f"Sequential SFT report: {output_dir / 'sequential_sft_report.json'}")
    if wandb_run is not None:
        wandb_run.finish()


def init_wandb(
    args: argparse.Namespace,
    tasks: list[SequentialTask],
    task_segments: list[SequentialTask],
) -> Any | None:
    try:
        import wandb
    except ImportError:
        print("wandb unavailable; continuing without WandB logging")
        return None

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_name": args.model_name,
            "task_sequence": [task.__dict__ for task in tasks],
            "expanded_task_sequence": [task.name for task in task_segments],
            "num_task_rounds": args.num_task_rounds,
            "max_seq_length": args.max_seq_length,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "weight_decay": args.weight_decay,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "prepare_dataset_name": args.prepare_dataset_name,
            "num_prepare_train_epochs": args.num_prepare_train_epochs,
            "prepare_dataset_limit": args.prepare_dataset_limit,
            "skip_prepare_stage": args.skip_prepare_stage,
            "num_train_epochs_per_task": args.num_train_epochs_per_task,
            "seed": args.seed,
            "readout_geometry_support_files": args.readout_geometry_support_files,
            "readout_geometry_logging_steps": args.readout_geometry_logging_steps,
            "probe_eval_specs": args.probe_eval_specs,
            "probe_eval_steps": args.probe_eval_steps,
            "probe_eval_batch_size": args.probe_eval_batch_size,
            "probe_eval_limit": args.probe_eval_limit,
            "probe_plasticity_block_size": args.probe_plasticity_block_size,
        },
    )
    wandb.define_metric("sequential_global_step")
    wandb.define_metric("continuous_epoch")
    wandb.define_metric("prepare_global_step")
    wandb.define_metric("prepare/*", step_metric="prepare_global_step")
    wandb.define_metric("prepare_train_loss", step_metric="prepare_global_step")
    wandb.define_metric("train/*", step_metric="continuous_epoch")
    wandb.define_metric("train_loss", step_metric="continuous_epoch")
    wandb.define_metric("loss", step_metric="continuous_epoch")
    wandb.define_metric("learning_rate", step_metric="continuous_epoch")
    wandb.define_metric("grad_norm", step_metric="continuous_epoch")
    wandb.define_metric("task/*", step_metric="sequential_global_step")
    wandb.define_metric("readout_geometry/*", step_metric="continuous_epoch")
    wandb.define_metric("probe/*", step_metric="continuous_epoch")
    wandb.define_metric("plasticity/*", step_metric="continuous_epoch")
    return run


def build_training_dataset(
    dataset_name: str,
    tokenizer: Any,
    max_seq_length: int,
    seed: int,
    limit: int | None,
    has_assistant_mask: bool,
) -> Any:
    dataset = load_instruction_dataset(dataset_name, "train", limit, seed)
    if not has_assistant_mask:
        dataset = to_prompt_completion_dataset(dataset, tokenizer)
        dataset = filter_prompt_completion_for_context(dataset, tokenizer, max_seq_length)
    return dataset


def run_prepare_stage(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    use_bf16: bool,
    assistant_only_loss: bool,
    completion_only_loss: bool | None,
    has_assistant_mask: bool,
) -> dict[str, Any]:
    prepare_output_dir = output_dir / "00_prepare_lima"
    prepare_output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting prepare SFT stage on {args.prepare_dataset_name} "
        f"for {args.num_prepare_train_epochs} epochs"
    )
    print(f"Prepare output dir: {prepare_output_dir}")

    dataset = build_training_dataset(
        dataset_name=args.prepare_dataset_name,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        limit=args.prepare_dataset_limit,
        has_assistant_mask=has_assistant_mask,
    )
    print(f"prepare_sft: train_examples={len(dataset)}")

    loss_history = LossHistoryCallback()
    training_args = SFTConfig(
        output_dir=str(prepare_output_dir),
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_prepare_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        bf16=use_bf16,
        fp16=False,
        assistant_only_loss=assistant_only_loss,
        completion_only_loss=completion_only_loss,
        packing=False,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
        run_name=args.wandb_run_name,
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[PrepareStageLoggingCallback(), loss_history],
    )
    train_output = trainer.train()
    trainer.save_state()

    report: dict[str, Any] = {
        "stage": "prepare_sft",
        "dataset_name": args.prepare_dataset_name,
        "train_examples": len(dataset),
        "num_train_epochs": args.num_prepare_train_epochs,
        "output_dir": str(prepare_output_dir),
        "prepare_global_steps": int(trainer.state.global_step),
        "train_metrics": train_output.metrics,
        "loss_history": loss_history.losses,
        "model": trainer.model,
    }
    report_for_disk = {key: value for key, value in report.items() if key != "model"}
    write_json(prepare_output_dir / "prepare_stage_report.json", report_for_disk)
    print_json("prepare_train_metrics", train_output.metrics)
    del trainer
    return report


class PrepareStageLoggingCallback(TrainerCallback):
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        self._wandb_log(
            {
                "prepare_global_step": 0,
                "prepare/epoch": 0.0,
                "prepare/stage": "prepare_sft",
            }
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        if logs is None:
            return
        prepare_step = int(state.global_step)
        wandb_logs: dict[str, int | float | str] = {
            "prepare_global_step": prepare_step,
            "prepare/epoch": float(logs.get("epoch", 0.0)),
        }
        if "loss" in logs:
            loss = float(logs["loss"])
            wandb_logs["prepare/loss"] = loss
            wandb_logs["prepare_train_loss"] = loss
        if "train_loss" in logs:
            loss = float(logs["train_loss"])
            wandb_logs["prepare/loss"] = loss
            wandb_logs["prepare_train_loss"] = loss
        if "learning_rate" in logs:
            wandb_logs["prepare/learning_rate"] = float(logs["learning_rate"])
        if "grad_norm" in logs:
            wandb_logs["prepare/grad_norm"] = float(logs["grad_norm"])
        self._wandb_log(wandb_logs)

    def _wandb_log(self, values: dict[str, int | float | str]) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            step = values.get("prepare_global_step")
            wandb.log(values, step=int(step) if isinstance(step, int | float) else None)


class SequentialTaskLoggingCallback(TrainerCallback):
    def __init__(
        self,
        task_name: str,
        task_index: int,
        round_index: int,
        segment_index: int,
        sequential_global_step_offset: int,
        wandb_step_offset: int,
        epochs_per_task: float,
    ) -> None:
        self.task_name = task_name
        self.task_index = task_index
        self.round_index = round_index
        self.segment_index = segment_index
        self.sequential_global_step_offset = sequential_global_step_offset
        self.wandb_step_offset = wandb_step_offset
        self.epochs_per_task = epochs_per_task

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        self._wandb_log(
            {
                "sequential_global_step": self.sequential_global_step_offset,
                "continuous_epoch": (self.segment_index - 1) * self.epochs_per_task,
                "task/index": self.task_index,
                "task/round": self.round_index,
                "task/segment_index": self.segment_index,
                "task/name": self.task_name,
                "task/transition": self.segment_index,
            }
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        if logs is None:
            return
        task_epoch = float(logs.get("epoch", 0.0))
        sequential_step = self.sequential_global_step_offset + state.global_step
        continuous_epoch = (self.segment_index - 1) * self.epochs_per_task + task_epoch

        logs["current_task_index"] = self.task_index
        logs["current_round_index"] = self.round_index
        logs["current_segment_index"] = self.segment_index
        logs["current_task_name"] = self.task_name
        logs["current_task_epoch"] = task_epoch
        logs["sequential_global_step"] = sequential_step
        logs["continuous_epoch"] = continuous_epoch

        wandb_logs: dict[str, int | float | str] = {
            "sequential_global_step": sequential_step,
            "continuous_epoch": continuous_epoch,
            "task/index": self.task_index,
            "task/round": self.round_index,
            "task/segment_index": self.segment_index,
            "task/name": self.task_name,
            "task/epoch": task_epoch,
        }
        field_map = {
            "loss": "train/loss",
            "train_loss": "train/loss",
            "learning_rate": "train/learning_rate",
            "grad_norm": "train/grad_norm",
            "mean_token_accuracy": "train/mean_token_accuracy",
            "entropy": "train/entropy",
            "num_tokens": "train/num_tokens",
            "train_runtime": "train/runtime",
            "train_samples_per_second": "train/samples_per_second",
            "train_steps_per_second": "train/steps_per_second",
        }
        for source, target in field_map.items():
            if source in logs:
                wandb_logs[target] = float(logs[source])

        if "loss" in logs:
            wandb_logs["loss"] = float(logs["loss"])
            wandb_logs["train_loss"] = float(logs["loss"])
        if "train_loss" in logs:
            wandb_logs["train_loss"] = float(logs["train_loss"])
        if "learning_rate" in logs:
            wandb_logs["learning_rate"] = float(logs["learning_rate"])
        if "grad_norm" in logs:
            wandb_logs["grad_norm"] = float(logs["grad_norm"])
        self._wandb_log(wandb_logs)

    def _wandb_log(self, values: dict[str, int | float | str]) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            step = values.get("sequential_global_step")
            if isinstance(step, int | float):
                wandb.log(values, step=self.wandb_step_offset + int(step))
            else:
                wandb.log(values)


def build_task_segments(tasks: list[SequentialTask], num_rounds: int) -> list[SequentialTask]:
    return [task for _round in range(num_rounds) for task in tasks]


def parse_task_sequence(values: list[str]) -> list[SequentialTask]:
    tasks = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected task spec task_name=dataset_jsonl, got {value!r}")
        name, dataset_path = value.split("=", 1)
        name = name.strip()
        dataset_path = dataset_path.strip()
        if not name or not dataset_path:
            raise ValueError(f"Invalid empty task spec component in {value!r}")
        if name in seen:
            raise ValueError(f"Duplicate task name: {name}")
        seen.add(name)
        tasks.append(SequentialTask(name=name, dataset_path=dataset_path))
    if not tasks:
        raise ValueError("At least one task is required.")
    return tasks


def write_base_reference(output_dir: Path, model_name: str) -> None:
    write_json(
        output_dir / "00_base_model_reference.json",
        {
            "model_name": model_name,
            "note": "Base model is loaded from the upstream model name; task checkpoints are saved after each sequential SFT stage.",
        },
    )


def print_json(label: str, value: object) -> None:
    print(f"{label}: {json.dumps(value, indent=2, sort_keys=True)}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
