from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class TokenSupport:
    path: str
    support_type: str
    tokenizer_name: str
    top_k: int
    filtering_rule: str
    task_token_ids: dict[str, list[int]]


@dataclass(frozen=True)
class ReadoutGeometryMetrics:
    support_name: str
    support_type: str
    task: str
    token_count: int
    hidden_dim: int
    w_shape: tuple[int, int]
    gram_shape: tuple[int, int]
    trace: float
    effective_rank: float
    hoyer_concentration: float
    min_eigenvalue: float
    max_eigenvalue: float
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_token_support(path: Path) -> TokenSupport:
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data["metadata"]
    task_token_ids = {
        task: [int(row["token_id"]) for row in rows] for task, rows in data["tasks"].items()
    }
    return TokenSupport(
        path=str(path),
        support_type=str(metadata["support_type"]),
        tokenizer_name=str(metadata["tokenizer_name"]),
        top_k=int(metadata["top_k"]),
        filtering_rule=str(metadata["filtering_rule"]),
        task_token_ids=task_token_ids,
    )


def task_readout_rows(lm_head_weight: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    if lm_head_weight.ndim != 2:
        raise ValueError(f"Expected a 2D LM head matrix, got shape {tuple(lm_head_weight.shape)}")
    if not token_ids:
        raise ValueError("Token support is empty.")
    vocab_size = lm_head_weight.shape[0]
    invalid = [token_id for token_id in token_ids if token_id < 0 or token_id >= vocab_size]
    if invalid:
        raise ValueError(f"Token IDs outside LM head vocab range 0..{vocab_size - 1}: {invalid[:10]}")
    index = torch.tensor(token_ids, dtype=torch.long, device=lm_head_weight.device)
    return lm_head_weight.index_select(0, index).float()


def gram_matrix(readout_rows: torch.Tensor) -> torch.Tensor:
    rows = readout_rows.float()
    return rows @ rows.T


def eigenvalue_spectrum(gram: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (gram.float() + gram.float().T)
    return torch.linalg.eigvalsh(symmetric).clamp_min(0.0)


def effective_rank(eigenvalues: torch.Tensor, eps: float = 1e-12) -> float:
    values = eigenvalues.float().clamp_min(0.0)
    total = values.sum()
    if total <= eps:
        return 0.0
    probabilities = values / total
    nonzero = probabilities[probabilities > eps]
    entropy = -(nonzero * torch.log(nonzero)).sum()
    return float(torch.exp(entropy).item())


def hoyer_concentration(eigenvalues: torch.Tensor, eps: float = 1e-12) -> float:
    values = eigenvalues.float().clamp_min(0.0)
    n = values.numel()
    if n <= 1:
        return 0.0
    l1 = values.sum()
    l2 = torch.linalg.vector_norm(values, ord=2)
    if l2 <= eps:
        return 0.0
    score = (math.sqrt(n) - float((l1 / l2).item())) / (math.sqrt(n) - 1.0)
    return float(max(0.0, min(1.0, score)))


def compute_readout_geometry(
    lm_head_weight: torch.Tensor,
    token_ids: list[int],
    support_name: str,
    support_type: str,
    task: str,
) -> tuple[ReadoutGeometryMetrics, list[float]]:
    rows = task_readout_rows(lm_head_weight, token_ids)
    gram = gram_matrix(rows)
    eigenvalues = eigenvalue_spectrum(gram)
    trace = float(torch.trace(gram).item())
    spectrum = [float(value) for value in eigenvalues.tolist()]
    metrics = ReadoutGeometryMetrics(
        support_name=support_name,
        support_type=support_type,
        task=task,
        token_count=len(token_ids),
        hidden_dim=int(rows.shape[1]),
        w_shape=(int(rows.shape[0]), int(rows.shape[1])),
        gram_shape=(int(gram.shape[0]), int(gram.shape[1])),
        trace=trace,
        effective_rank=effective_rank(eigenvalues),
        hoyer_concentration=hoyer_concentration(eigenvalues),
        min_eigenvalue=spectrum[0] if spectrum else math.nan,
        max_eigenvalue=spectrum[-1] if spectrum else math.nan,
        finite=all(
            math.isfinite(value)
            for value in (
                trace,
                effective_rank(eigenvalues),
                hoyer_concentration(eigenvalues),
                spectrum[0] if spectrum else math.nan,
                spectrum[-1] if spectrum else math.nan,
            )
        ),
    )
    return metrics, spectrum


def validate_metrics(metrics: ReadoutGeometryMetrics) -> None:
    if metrics.w_shape != (metrics.token_count, metrics.hidden_dim):
        raise RuntimeError(f"Unexpected W_S shape for {metrics.support_name}/{metrics.task}: {metrics.w_shape}")
    if metrics.gram_shape != (metrics.token_count, metrics.token_count):
        raise RuntimeError(
            f"Unexpected Gram shape for {metrics.support_name}/{metrics.task}: {metrics.gram_shape}"
        )
    if not metrics.finite:
        raise RuntimeError(f"Non-finite metric for {metrics.support_name}/{metrics.task}")
    if not 0.0 <= metrics.effective_rank <= metrics.token_count + 1e-5:
        raise RuntimeError(
            f"Effective rank out of range for {metrics.support_name}/{metrics.task}: "
            f"{metrics.effective_rank}"
        )
    if not 0.0 <= metrics.hoyer_concentration <= 1.0:
        raise RuntimeError(
            f"Hoyer concentration out of range for {metrics.support_name}/{metrics.task}: "
            f"{metrics.hoyer_concentration}"
        )
