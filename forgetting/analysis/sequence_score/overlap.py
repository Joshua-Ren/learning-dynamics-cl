from collections import Counter

import torch


def _filtered(ids: list[int], excluded: set[int]) -> list[int]:
    return [int(x) for x in ids if int(x) not in excluded]


def multiset_overlap_count(left_ids: list[int], right_ids: list[int], excluded_token_ids: set[int] | None = None) -> int:
    excluded = excluded_token_ids or set()
    left_counts = Counter(_filtered(left_ids, excluded))
    right_counts = Counter(_filtered(right_ids, excluded))
    return sum(count * right_counts[token_id] for token_id, count in left_counts.items())


def prefix_overlap_counts(
    update_input_ids: torch.Tensor,
    label_positions: torch.Tensor,
    observation_prompt_ids: torch.Tensor,
    mode: str = "prefix",
    excluded_token_ids: set[int] | None = None,
) -> torch.Tensor:
    update_ids = [int(x) for x in update_input_ids.detach().cpu().tolist()]
    obs_ids = [int(x) for x in observation_prompt_ids.detach().cpu().tolist()]
    positions = [int(x) for x in label_positions.detach().cpu().tolist()]

    if mode == "full_sequence_scalar":
        value = multiset_overlap_count(update_ids, obs_ids, excluded_token_ids)
        return torch.full((len(positions),), float(value), dtype=torch.float32)
    if mode != "prefix":
        raise ValueError(f"Unsupported kembd mode: {mode}")

    values = []
    for pos in positions:
        if pos < 0 or pos > len(update_ids):
            raise ValueError(f"label position {pos} is out of range for update sequence length {len(update_ids)}.")
        values.append(float(multiset_overlap_count(update_ids[:pos], obs_ids, excluded_token_ids)))
    return torch.tensor(values, dtype=torch.float32)
