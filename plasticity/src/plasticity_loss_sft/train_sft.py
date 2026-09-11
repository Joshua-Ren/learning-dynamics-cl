from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal full-parameter SFT entry point.")
    parser.add_argument("--model_name", choices=SUPPORTED_MODELS, default=SUPPORTED_MODELS[0])
    parser.add_argument("--dataset_name", default="trl-lib/Capybara")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_limit", type=int, default=32)
    parser.add_argument("--output_dir", default="outputs/sft_baseline")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--save_strategy", choices=("no", "steps", "epoch"), default="steps")
    parser.add_argument("--save_model_at_end", action="store_true")
    parser.add_argument("--custom_save_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--wandb_project", default="plasticity-loss")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--report_to_wandb", action="store_true")
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--task_index", type=int, default=1)
    parser.add_argument("--task_round", type=int, default=1)
    parser.add_argument("--task_segment_index", type=int, default=1)
    parser.add_argument("--continuous_epoch_offset", type=float, default=0.0)
    parser.add_argument("--sequential_global_step_offset", type=int, default=0)
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
    if args.report_to_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_run_name:
            os.environ.setdefault("WANDB_NAME", args.wandb_run_name)
        init_wandb(args)
    set_seed(args.seed)

    use_bf16 = bf16_supported() and not args.no_bf16
    print_json("package_versions", package_versions())
    print(f"bf16: {use_bf16}")

    dataset_limit = None if args.dataset_limit is not None and args.dataset_limit < 0 else args.dataset_limit
    dataset = load_instruction_dataset(args.dataset_name, args.split, dataset_limit, args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model_name, use_bf16)
    has_assistant_mask = supports_assistant_token_mask(tokenizer)
    if not args.skip_assistant_mask_check and has_assistant_mask:
        assert_assistant_mask_supported(tokenizer)

    if has_assistant_mask:
        loss_masking_mode = "assistant_only_loss"
        assistant_only_loss = True
        completion_only_loss = None
    else:
        loss_masking_mode = "native_prompt_completion_loss"
        dataset = to_prompt_completion_dataset(dataset, tokenizer)
        dataset = filter_prompt_completion_for_context(dataset, tokenizer, args.max_seq_length)
        print(f"filtered_train_examples: {len(dataset)}")
        assistant_only_loss = False
        completion_only_loss = True
    print(f"loss_masking_mode: {loss_masking_mode}")

    access_report = inspect_model_access(model)
    print_json("model_access", access_report.__dict__)
    reset_peak_memory()

    loss_history = LossHistoryCallback()
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
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
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        report_to=["wandb"] if args.report_to_wandb else [],
        run_name=args.wandb_run_name,
        remove_unused_columns=False,
    )

    callbacks: list[TrainerCallback] = [
        SingleTaskLoggingCallback(
            task_name=args.task_name,
            task_index=args.task_index,
            task_round=args.task_round,
            task_segment_index=args.task_segment_index,
            continuous_epoch_offset=args.continuous_epoch_offset,
            sequential_global_step_offset=args.sequential_global_step_offset,
        ),
        loss_history,
        CustomSaveCallback(args.output_dir, args.custom_save_steps),
    ]
    if args.readout_geometry_support_files:
        callbacks.append(
            ReadoutGeometryWandbCallback(
                support_files=args.readout_geometry_support_files,
                logging_steps=args.readout_geometry_logging_steps,
                continuous_epoch_offset=args.continuous_epoch_offset,
                sequential_global_step_offset=args.sequential_global_step_offset,
                task_name=args.task_name,
                task_index=args.task_index,
                task_round=args.task_round,
                task_segment_index=args.task_segment_index,
            )
        )
    if args.probe_eval_specs:
        callbacks.append(
            ProbeConfidenceWandbCallback(
                probe_tasks=parse_probe_specs(args.probe_eval_specs),
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                eval_steps=args.probe_eval_steps,
                batch_size=args.probe_eval_batch_size,
                limit=args.probe_eval_limit,
                seed=args.seed,
                output_dir=args.output_dir,
                plasticity_block_size=args.probe_plasticity_block_size,
                continuous_epoch_offset=args.continuous_epoch_offset,
                sequential_global_step_offset=args.sequential_global_step_offset,
                task_name=args.task_name,
                task_index=args.task_index,
                task_round=args.task_round,
                task_segment_index=args.task_segment_index,
            )
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if args.save_model_at_end:
        trainer.save_model()
    trainer.save_state()

    gpu = gpu_report()
    print_json("loss_history", loss_history.losses)
    print_json("gpu", gpu.__dict__)
    print(f"trainer_state: {Path(args.output_dir, 'trainer_state.json')}")
    finish_wandb()


def init_wandb(args: argparse.Namespace) -> None:
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is not None:
        return
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "dataset_limit": None if args.dataset_limit is not None and args.dataset_limit < 0 else args.dataset_limit,
            "max_seq_length": args.max_seq_length,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "weight_decay": args.weight_decay,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "task_name": args.task_name,
            "task_index": args.task_index,
            "task_round": args.task_round,
            "task_segment_index": args.task_segment_index,
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
    wandb.define_metric("train/*", step_metric="continuous_epoch")
    wandb.define_metric("train_loss", step_metric="continuous_epoch")
    wandb.define_metric("loss", step_metric="continuous_epoch")
    wandb.define_metric("learning_rate", step_metric="continuous_epoch")
    wandb.define_metric("grad_norm", step_metric="continuous_epoch")
    wandb.define_metric("task/*", step_metric="sequential_global_step")
    wandb.define_metric("readout_geometry/*", step_metric="continuous_epoch")
    wandb.define_metric("probe/*", step_metric="continuous_epoch")
    wandb.define_metric("plasticity/*", step_metric="continuous_epoch")


def finish_wandb() -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()


class SingleTaskLoggingCallback(TrainerCallback):
    def __init__(
        self,
        task_name: str | None,
        task_index: int,
        task_round: int,
        task_segment_index: int,
        continuous_epoch_offset: float,
        sequential_global_step_offset: int,
    ) -> None:
        self.task_name = task_name
        self.task_index = task_index
        self.task_round = task_round
        self.task_segment_index = task_segment_index
        self.continuous_epoch_offset = continuous_epoch_offset
        self.sequential_global_step_offset = sequential_global_step_offset

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if self.task_name is None:
            return
        self._wandb_log(
            {
                "sequential_global_step": self.sequential_global_step_offset,
                "continuous_epoch": self.continuous_epoch_offset,
                "task/index": self.task_index,
                "task/round": self.task_round,
                "task/segment_index": self.task_segment_index,
                "task/name": self.task_name,
                "task/transition": self.task_segment_index,
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
        if logs is None or self.task_name is None:
            return
        task_epoch = float(logs.get("epoch", 0.0))
        sequential_step = self.sequential_global_step_offset + state.global_step
        continuous_epoch = self.continuous_epoch_offset + task_epoch

        logs["current_task_index"] = self.task_index
        logs["current_round_index"] = self.task_round
        logs["current_segment_index"] = self.task_segment_index
        logs["current_task_name"] = self.task_name
        logs["current_task_epoch"] = task_epoch
        logs["sequential_global_step"] = sequential_step
        logs["continuous_epoch"] = continuous_epoch

        wandb_logs: dict[str, int | float | str] = {
            "sequential_global_step": sequential_step,
            "continuous_epoch": continuous_epoch,
            "task/index": self.task_index,
            "task/round": self.task_round,
            "task/segment_index": self.task_segment_index,
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
            wandb.log(values, step=int(step) if isinstance(step, int | float) else None)


def print_json(label: str, value: object) -> None:
    print(f"{label}: {json.dumps(value, indent=2, sort_keys=True)}")


if __name__ == "__main__":
    main()
