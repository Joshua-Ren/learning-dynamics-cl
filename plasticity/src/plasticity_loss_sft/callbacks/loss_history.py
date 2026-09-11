from __future__ import annotations

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class LossHistoryCallback(TrainerCallback):
    def __init__(self) -> None:
        self.losses: list[tuple[int, float]] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        if logs and "loss" in logs:
            self.losses.append((state.global_step, float(logs["loss"])))
