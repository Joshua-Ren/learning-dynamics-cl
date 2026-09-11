from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify checkpoint and final trainer state files.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_global_step", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    checkpoints = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint trainer_state.json files found in {output_dir}")

    final_state_path = output_dir / "trainer_state.json"
    if not final_state_path.exists():
        raise FileNotFoundError(f"Missing final state file: {final_state_path}")

    state = json.loads(final_state_path.read_text(encoding="utf-8"))
    global_step = int(state["global_step"])
    if args.expected_global_step is not None and global_step != args.expected_global_step:
        raise RuntimeError(f"Expected global_step={args.expected_global_step}, got {global_step}")

    print(
        json.dumps(
            {
                "checkpoint_count": len(checkpoints),
                "latest_checkpoint": str(checkpoints[-1].parent),
                "final_global_step": global_step,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
