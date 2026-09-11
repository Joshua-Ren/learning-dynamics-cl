from __future__ import annotations

from analysis.sequence_score.model_helpers import (
    block_input_norm as _block_input_norm,
    get_readout_weight,
    get_transformer_layers as _get_transformer_layers,
    rms_normalize,
    rmsnorm_epsilon as _get_rmsnorm_eps,
)


def _rms_normalize_hidden(hidden, eps=1e-6):
    return rms_normalize(hidden, epsilon=eps)


__all__ = [
    "_block_input_norm",
    "_get_rmsnorm_eps",
    "_get_transformer_layers",
    "_rms_normalize_hidden",
    "get_readout_weight",
]
