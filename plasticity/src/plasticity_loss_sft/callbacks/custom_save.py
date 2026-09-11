from __future__ import annotations

from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class CustomSaveCallback(TrainerCallback):
    """Placeholder for future project-specific periodic saves."""

    def __init__(self, output_dir: str, save_steps: int | None) -> None:
        self.output_dir = Path(output_dir)
        self.save_steps = save_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if self.save_steps is None or self.save_steps <= 0:
            return
        if state.global_step <= 0 or state.global_step % self.save_steps != 0:
            return
        custom_save_placeholder(self.output_dir, state.global_step)


def custom_save_placeholder(output_dir: Path, step: int) -> None:
    marker_dir = output_dir / "custom_saves"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"step-{step}.txt"
    marker_path.write_text(
        "custom save placeholder; no model/tokenizer/checkpoint artifact is saved yet\n",
        encoding="utf-8",
    )
