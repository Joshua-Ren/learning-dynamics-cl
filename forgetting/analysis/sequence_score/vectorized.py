from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

import torch

from .components import score_from_components
from .config import SequenceScoreConfig
from .features import ObservationFeatures, UpdateFeatures
from .overlap import prefix_overlap_counts


def _stack(items: list[torch.Tensor], name: str) -> torch.Tensor:
    if not items:
        raise ValueError(f"Cannot stack empty {name}.")
    return torch.stack([item.float() for item in items], dim=0)


def _safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def _last_hidden_baselines(update_hidden: torch.Tensor, observation_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return simple last-layer hidden-similarity baselines for one update vs many observations.

    update_hidden has shape [T, L, d]; observation_hidden has shape [O, L, d].
    The observation side is the final MMLU prompt position already cached in ObservationFeatures.
    """
    update_last = update_hidden[:, -1, :].float()
    obs_last = observation_hidden[:, -1, :].float()

    dot = torch.einsum("td,od->to", update_last, obs_last)
    cosine = torch.einsum("td,od->to", _safe_normalize(update_last), _safe_normalize(obs_last))

    update_centered = update_last - update_last.mean(dim=-1, keepdim=True)
    obs_centered = obs_last - obs_last.mean(dim=-1, keepdim=True)
    centered_cosine = torch.einsum(
        "td,od->to",
        _safe_normalize(update_centered),
        _safe_normalize(obs_centered),
    )

    return {
        "baseline_last_hidden_dot_mean": dot.mean(dim=0),
        "baseline_last_hidden_dot_sum": dot.sum(dim=0),
        "baseline_last_hidden_dot_max": dot.max(dim=0).values,
        "baseline_last_hidden_cosine_mean": cosine.mean(dim=0),
        "baseline_last_hidden_cosine_sum": cosine.sum(dim=0),
        "baseline_last_hidden_cosine_max": cosine.max(dim=0).values,
        "baseline_last_hidden_centered_cosine_mean": centered_cosine.mean(dim=0),
        "baseline_last_hidden_centered_cosine_sum": centered_cosine.sum(dim=0),
        "baseline_last_hidden_centered_cosine_max": centered_cosine.max(dim=0).values,
    }


def score_update_against_observations(
    update: UpdateFeatures,
    observations: list[ObservationFeatures],
    config: SequenceScoreConfig,
    observation_chunk_size: int = 16,
) -> list[dict]:
    """Score one GSM8K update against cached MMLU observations.

    This keeps the Phase 1 equations but vectorizes over a chunk of observations.
    No model-parameter gradients are computed.
    """
    if observation_chunk_size <= 0:
        raise ValueError("observation_chunk_size must be positive.")
    if update.hidden.dim() != 3:
        raise ValueError("Update hidden must have shape [T, L, d].")

    rows = []
    for start in range(0, len(observations), observation_chunk_size):
        chunk = observations[start : start + observation_chunk_size]
        obs_option_grad = _stack([obs.option_gradients for obs in chunk], "option gradients")
        obs_projected_option = _stack([obs.projected_option_gradients for obs in chunk], "projected option gradients")
        obs_ifmass_grad = _stack([obs.ifmass_gradient for obs in chunk], "ifmass gradients")
        obs_projected_ifmass = _stack([obs.projected_ifmass_gradient for obs in chunk], "projected ifmass gradients")
        obs_hidden = _stack([obs.hidden for obs in chunk], "observation hidden")

        uniform_gg = torch.einsum("tv,oav->toa", update.gradients.float(), obs_option_grad)
        uniform_gwwg = torch.einsum("td,oad->toa", update.projected_gradients.float(), obs_projected_option)
        ifmass_gg = torch.einsum("tv,oav->toa", update.gradients.float(), obs_ifmass_grad)
        ifmass_gwwg = torch.einsum("td,oad->toa", update.projected_gradients.float(), obs_projected_ifmass)
        hh_all = torch.einsum("tld,old->tol", update.hidden.float(), obs_hidden)
        last_hidden_baselines = _last_hidden_baselines(update.hidden, obs_hidden)

        for local_idx, obs in enumerate(chunk):
            kembd = prefix_overlap_counts(
                update_input_ids=update.input_ids,
                label_positions=update.label_positions,
                observation_prompt_ids=obs.prompt_ids,
                mode=config.kembd_mode,
                excluded_token_ids=set(),
            )
            uniform = score_from_components(
                gg=uniform_gg[:, local_idx, :],
                hh_all=hh_all[:, local_idx, :],
                gwwg=uniform_gwwg[:, local_idx, :],
                kembd=kembd,
                objective="uniform_options",
            )
            ifmass = score_from_components(
                gg=ifmass_gg[:, local_idx, :],
                hh_all=hh_all[:, local_idx, :],
                gwwg=ifmass_gwwg[:, local_idx, :],
                kembd=kembd,
                objective="ifmass",
            )
            rows.append(
                {
                    "observation_local_index": start + local_idx,
                    "mmlu_subject": obs.metadata.get("subject"),
                    "mmlu_index": obs.metadata.get("source_index"),
                    "mmlu_target": obs.metadata.get("target"),
                    "mmlu_observation_mode": obs.metadata.get("observation_mode", "sampled"),
                    "num_source_examples": obs.metadata.get("num_source_examples", 1),
                    "representative_source_index": obs.metadata.get("representative_source_index", ""),
                    "num_update_tokens": int(update.label_positions.numel()),
                    **{key: float(value[local_idx].item()) for key, value in last_hidden_baselines.items()},
                    **{f"uniform_{key}": value for key, value in asdict(uniform).items() if key != "objective"},
                    **{f"ifmass_{key}": value for key, value in asdict(ifmass).items() if key != "objective"},
                }
            )

    return rows


def mean_or_none(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_pair_rows(pair_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return MMLU-example rows and subject-level rows."""
    by_example = defaultdict(list)
    by_subject = defaultdict(list)
    score_cols = [
        "uniform_ch1_mean",
        "uniform_ch2_mean",
        "uniform_ch2_hidden_mean",
        "uniform_ch2_embd_mean",
        "uniform_total_mean",
        "ifmass_ch1_mean",
        "ifmass_ch2_mean",
        "ifmass_ch2_hidden_mean",
        "ifmass_ch2_embd_mean",
        "ifmass_total_mean",
        "uniform_ch1_sum",
        "uniform_ch2_sum",
        "uniform_total_sum",
        "ifmass_ch1_sum",
        "ifmass_ch2_sum",
        "ifmass_total_sum",
        "baseline_last_hidden_dot_mean",
        "baseline_last_hidden_dot_sum",
        "baseline_last_hidden_dot_max",
        "baseline_last_hidden_cosine_mean",
        "baseline_last_hidden_cosine_sum",
        "baseline_last_hidden_cosine_max",
        "baseline_last_hidden_centered_cosine_mean",
        "baseline_last_hidden_centered_cosine_sum",
        "baseline_last_hidden_centered_cosine_max",
    ]

    for row in pair_rows:
        key = (row["mmlu_subject"], row["mmlu_index"])
        by_example[key].append(row)
        by_subject[row["mmlu_subject"]].append(row)

    example_rows = []
    for (subject, mmlu_index), rows in sorted(by_example.items()):
        out = {"subject": subject, "mmlu_subject": subject, "mmlu_index": mmlu_index, "num_pairs": len(rows)}
        for col in score_cols:
            if col not in rows[0]:
                continue
            values = [float(row[col]) for row in rows]
            out[col] = mean_or_none(values)
            out[f"{col}_abs_mean"] = mean_or_none(abs(value) for value in values)
        example_rows.append(out)

    subject_rows = []
    for subject_order, (subject, rows) in enumerate(sorted(by_subject.items())):
        if any(row.get("mmlu_observation_mode") == "subject_mean" for row in rows):
            num_mmlu_examples = max(int(row.get("num_source_examples") or 1) for row in rows)
            observation_mode = "subject_mean"
        else:
            num_mmlu_examples = len({row["mmlu_index"] for row in rows})
            observation_mode = "sampled"
        out = {
            "subject_order": subject_order,
            "subject": subject,
            "mmlu_subject": subject,
            "num_pairs": len(rows),
            "num_mmlu_examples": num_mmlu_examples,
            "num_gsm8k_examples": len({row["gsm8k_index"] for row in rows}),
            "mmlu_observation_mode": observation_mode,
        }
        for col in score_cols:
            if col not in rows[0]:
                continue
            values = torch.tensor([float(row[col]) for row in rows], dtype=torch.float32)
            abs_values = values.abs()
            out[col] = float(values.mean().item())
            out[f"{col}_std"] = float(values.std(unbiased=True).item()) if values.numel() > 1 else 0.0
            out[f"{col}_stderr"] = (
                float(values.std(unbiased=True).item() / (values.numel() ** 0.5)) if values.numel() > 1 else 0.0
            )
            out[f"{col}_abs_mean"] = float(abs_values.mean().item())
            out[f"{col}_abs_mean_std"] = float(abs_values.std(unbiased=True).item()) if abs_values.numel() > 1 else 0.0
            out[f"{col}_abs_mean_stderr"] = (
                float(abs_values.std(unbiased=True).item() / (abs_values.numel() ** 0.5)) if abs_values.numel() > 1 else 0.0
            )
        subject_rows.append(out)

    return example_rows, subject_rows
