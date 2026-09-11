from __future__ import annotations

from analysis.sequence_score.model_helpers import supervised_positions


def iter_supervised_positions(sample: dict, max_positions: int | None = None):
    positions = supervised_positions(sample["labels"], sample.get("attention_mask"))
    if max_positions is not None:
        positions = positions[:max_positions]
    yield from positions
