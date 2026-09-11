from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from plasticity_loss_sft.modeling import get_lm_head_weight
from plasticity_loss_sft.readout_geometry import compute_readout_geometry, load_token_support, validate_metrics


METRIC_FIELDS = ("trace", "effective_rank", "hoyer_concentration", "max_eigenvalue")


class ReadoutGeometryWandbCallback(TrainerCallback):
    def __init__(
        self,
        support_files: list[str],
        logging_steps: int,
        continuous_epoch_offset: float = 0.0,
        sequential_global_step_offset: int = 0,
        wandb_step_offset: int = 0,
        task_name: str | None = None,
        task_index: int | None = None,
        task_round: int | None = None,
        task_segment_index: int | None = None,
    ) -> None:
        if logging_steps < 0:
            raise ValueError("readout geometry logging_steps must be non-negative")
        self.support_files = [str(path) for path in support_files]
        self.logging_steps = logging_steps
        self.continuous_epoch_offset = continuous_epoch_offset
        self.sequential_global_step_offset = sequential_global_step_offset
        self.wandb_step_offset = wandb_step_offset
        self.task_name = task_name
        self.task_index = task_index
        self.task_round = task_round
        self.task_segment_index = task_segment_index
        self.supports = {Path(path).stem: load_token_support(Path(path)) for path in self.support_files}
        self.initial_metrics: dict[tuple[str, str], dict[str, float]] = {}
        self.previous_metrics: dict[tuple[str, str], dict[str, float]] = {}
        self.last_logged_step: int | None = None

    def set_context(
        self,
        continuous_epoch_offset: float,
        sequential_global_step_offset: int,
        wandb_step_offset: int = 0,
        task_name: str | None = None,
        task_index: int | None = None,
        task_round: int | None = None,
        task_segment_index: int | None = None,
    ) -> None:
        self.continuous_epoch_offset = continuous_epoch_offset
        self.sequential_global_step_offset = sequential_global_step_offset
        self.wandb_step_offset = wandb_step_offset
        self.task_name = task_name
        self.task_index = task_index
        self.task_round = task_round
        self.task_segment_index = task_segment_index

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        self._log(state=state, model=kwargs.get("model"), event="train_begin", force=True)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if self.logging_steps == 0:
            return
        if state.global_step <= 0 or state.global_step % self.logging_steps != 0:
            return
        self._log(state=state, model=kwargs.get("model"), event="step", force=False)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        self._log(state=state, model=kwargs.get("model"), event="train_end", force=True)

    def _log(self, state: TrainerState, model: Any, event: str, force: bool) -> None:
        if model is None:
            return
        if not force and self.last_logged_step == state.global_step:
            return
        if force and self.last_logged_step == state.global_step and self.initial_metrics:
            return

        metrics = self._compute_metrics(model)
        sequential_step = self.sequential_global_step_offset + int(state.global_step)
        continuous_epoch = self.continuous_epoch_offset + float(state.epoch or 0.0)
        payload: dict[str, int | float | str] = {
            "sequential_global_step": sequential_step,
            "continuous_epoch": continuous_epoch,
            "readout_geometry/event": event,
        }
        if self.task_name is not None:
            payload["task/name"] = self.task_name
        if self.task_index is not None:
            payload["task/index"] = self.task_index
        if self.task_round is not None:
            payload["task/round"] = self.task_round
        if self.task_segment_index is not None:
            payload["task/segment_index"] = self.task_segment_index

        for key, values in metrics.items():
            support_name, task = key
            if key not in self.initial_metrics:
                self.initial_metrics[key] = values
            initial = self.initial_metrics[key]
            previous = self.previous_metrics.get(key, values)
            prefix = f"readout_geometry/{sanitize_metric_name(support_name)}/{sanitize_metric_name(task)}"
            for field in METRIC_FIELDS:
                value = values[field]
                payload[f"{prefix}/{field}"] = value
                payload[f"{prefix}/rel_initial_{field}"] = relative_change(value, initial[field])
                payload[f"{prefix}/rel_previous_{field}"] = relative_change(value, previous[field])
            payload[f"{prefix}/token_count"] = values["token_count"]
            payload[f"{prefix}/hidden_dim"] = values["hidden_dim"]

        self._wandb_log(payload, self.wandb_step_offset + sequential_step)
        self.previous_metrics = metrics
        self.last_logged_step = int(state.global_step)

    def _compute_metrics(self, model: Any) -> dict[tuple[str, str], dict[str, float]]:
        lm_head_weight = get_lm_head_weight(model).detach()
        results: dict[tuple[str, str], dict[str, float]] = {}
        for support_name, support in self.supports.items():
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
                results[(support_name, task)] = {
                    "trace": float(row["trace"]),
                    "effective_rank": float(row["effective_rank"]),
                    "hoyer_concentration": float(row["hoyer_concentration"]),
                    "max_eigenvalue": float(row["max_eigenvalue"]),
                    "token_count": float(row["token_count"]),
                    "hidden_dim": float(row["hidden_dim"]),
                }
        return results

    def _wandb_log(self, values: dict[str, int | float | str], step: int) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is not None:
            wandb.log(values, step=step)


def relative_change(value: float, reference_value: float) -> float:
    if reference_value == 0.0:
        return math.nan
    return (value - reference_value) / reference_value


def sanitize_metric_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "unknown"
