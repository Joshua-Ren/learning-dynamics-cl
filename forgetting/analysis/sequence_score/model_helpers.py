from __future__ import annotations

import torch


def get_readout_weight(model: object) -> torch.Tensor:
    for attribute in ("lm_head", "embed_out", "output_projection"):
        module = getattr(model, attribute, None)
        weight = getattr(module, "weight", None)
        if weight is not None:
            return weight
    if hasattr(model, "get_output_embeddings"):
        module = model.get_output_embeddings()
        weight = getattr(module, "weight", None)
        if weight is not None:
            return weight
    raise AttributeError("Could not locate the language-model readout weight.")


def get_transformer_layers(model: object):
    candidates = (
        getattr(model, "model", None),
        getattr(model, "transformer", None),
        getattr(model, "gpt_neox", None),
        model,
    )
    for container in candidates:
        if container is None:
            continue
        for attribute in ("layers", "h", "blocks"):
            layers = getattr(container, attribute, None)
            if layers is not None:
                return layers
    return None


def block_input_norm(model: object, layer_index: int):
    layers = get_transformer_layers(model)
    if layers is None or not 0 <= layer_index < len(layers):
        return None
    block = layers[layer_index]
    for attribute in ("input_layernorm", "ln_1", "input_norm"):
        norm = getattr(block, attribute, None)
        if norm is not None:
            return norm
    return None


def rmsnorm_epsilon(model: object) -> float:
    config = getattr(model, "config", None)
    for attribute in ("rms_norm_eps", "layer_norm_epsilon", "norm_eps"):
        value = getattr(config, attribute, None)
        if value is not None:
            return float(value)
    return 1e-6


def rms_normalize(hidden: torch.Tensor, epsilon: float) -> torch.Tensor:
    return hidden * torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + epsilon)


def supervised_positions(labels: torch.Tensor, attention_mask: torch.Tensor | None = None) -> list[int]:
    if labels.dim() != 1:
        raise ValueError("labels must have shape [sequence_length].")
    if attention_mask is not None and attention_mask.dim() != 1:
        raise ValueError("attention_mask must have shape [sequence_length].")
    return [
        position
        for position in range(1, labels.shape[0])
        if int(labels[position].item()) != -100
        and (attention_mask is None or int(attention_mask[position].item()) != 0)
    ]
