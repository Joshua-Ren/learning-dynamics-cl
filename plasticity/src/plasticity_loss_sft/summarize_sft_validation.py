from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a validation SFT run.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_global_step", type=int, required=True)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--preflight_json", default=None)
    parser.add_argument("--summary_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    trainer_state_path = output_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Missing trainer state: {trainer_state_path}")

    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    log_history: list[dict[str, Any]] = state.get("log_history", [])
    loss_logs = [entry for entry in log_history if "loss" in entry]
    losses = [float(entry["loss"]) for entry in loss_logs]
    learning_rate_logged = any("learning_rate" in entry for entry in log_history)
    grad_norm_logged = any("grad_norm" in entry for entry in log_history)
    finite_losses = all(math.isfinite(loss) for loss in losses)

    if losses:
        representative_losses = [losses[0], losses[len(losses) // 2], losses[-1]]
    else:
        representative_losses = []

    global_step = int(state.get("global_step", -1))
    checkpoints = sorted(path.name for path in output_dir.glob("checkpoint-*"))
    resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    resume_state_exists = (
        (resume_checkpoint / "trainer_state.json").exists() if resume_checkpoint is not None else None
    )
    optimizer_state_exists = (
        (resume_checkpoint / "optimizer.pt").exists() if resume_checkpoint is not None else None
    )
    scheduler_state_exists = (
        (resume_checkpoint / "scheduler.pt").exists() if resume_checkpoint is not None else None
    )

    preflight = None
    if args.preflight_json:
        preflight_path = Path(args.preflight_json)
        if preflight_path.exists():
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))

    summary = {
        "output_dir": str(output_dir),
        "global_step": global_step,
        "expected_global_step": args.expected_global_step,
        "global_step_matches_expected": global_step == args.expected_global_step,
        "checkpoint_dirs": checkpoints,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "resume_checkpoint_state_exists": resume_state_exists,
        "resume_optimizer_state_exists": optimizer_state_exists,
        "resume_scheduler_state_exists": scheduler_state_exists,
        "loss_count": len(losses),
        "losses_finite": finite_losses,
        "representative_losses": representative_losses,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "loss_decreased_first_to_last": bool(losses and losses[-1] < losses[0]),
        "learning_rate_logged": learning_rate_logged,
        "grad_norm_logged": grad_norm_logged,
        "preflight": preflight,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    required_ok = [
        summary["global_step_matches_expected"],
        summary["losses_finite"],
        summary["learning_rate_logged"],
        summary["grad_norm_logged"],
    ]
    if resume_checkpoint is not None:
        required_ok.extend(
            [
                summary["resume_checkpoint_state_exists"],
                summary["resume_optimizer_state_exists"],
                summary["resume_scheduler_state_exists"],
            ]
        )
    if preflight is not None:
        required_ok.extend(
            [
                preflight["chat_template_present"],
                preflight["prompt_labels_all_ignored"],
                preflight["assistant_labels_present"],
                preflight["hidden_state_count"] > 0,
                preflight["lm_head_grad_present"],
            ]
        )
    if not all(required_ok):
        raise SystemExit("One or more validation checks failed; inspect the summary JSON above.")


if __name__ == "__main__":
    main()
