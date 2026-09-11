from dataclasses import dataclass
from typing import Any

import torch

from utils.ch1_ch2_metrics import get_readout_weight
from utils.one_step_forgetting import iter_supervised_positions

from .config import SequenceScoreConfig
from .data import MMLUObservation, SupervisedSample, resolve_option_token_ids
from .hidden_extractor import NormalizedHiddenExtractor
from .output_gradients import (
    allowed_token_mass_gradient,
    logprob_gradient,
    nll_gradient,
    option_gradients,
    project_output_gradients,
)


@dataclass
class UpdateFeatures:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label_positions: torch.Tensor
    logit_positions: torch.Tensor
    target_ids: torch.Tensor
    gradients: torch.Tensor
    projected_gradients: torch.Tensor
    hidden: torch.Tensor
    metadata: dict[str, Any]


@dataclass
class ObservationFeatures:
    prompt_ids: torch.Tensor
    attention_mask: torch.Tensor
    option_token_ids: list[int]
    option_token_info: dict[str, Any]
    option_gradients: torch.Tensor
    projected_option_gradients: torch.Tensor
    ifmass_gradient: torch.Tensor
    projected_ifmass_gradient: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor
    metadata: dict[str, Any]


def _model_device(model: object) -> torch.device:
    return next(model.parameters()).device


def _decode_token(tokenizer: object, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _select_first_response_position(
    tokenizer: object,
    labels: torch.Tensor,
    label_positions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if label_positions.numel() == 0:
        raise ValueError("Cannot select first response token from an empty supervised sequence.")

    selected_position = label_positions[:1]
    selected_token_id = int(labels[selected_position[0]].detach().cpu().item())
    selected_text = _decode_token(tokenizer, selected_token_id)
    return selected_position, {
        "update_token_mode": "first_response_token",
        "selected_supervised_offset": 0,
        "selected_label_position": int(selected_position.item()),
        "selected_token_id": selected_token_id,
        "selected_token_text": selected_text,
        "num_original_supervised_tokens": int(label_positions.numel()),
    }


def _select_first_final_answer_position(
    tokenizer: object,
    labels: torch.Tensor,
    label_positions: torch.Tensor,
    markers: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, Any]]:
    target_ids = labels[label_positions].detach().cpu().long().tolist()
    if not target_ids:
        raise ValueError("Cannot select final-answer token from an empty supervised sequence.")

    token_texts = [_decode_token(tokenizer, token_id) for token_id in target_ids]
    decoded = "".join(token_texts)

    marker_hits = [(decoded.rfind(marker), marker) for marker in markers]
    marker_start, marker = max(marker_hits, key=lambda item: item[0])
    if marker_start < 0:
        preview = decoded[-500:] if len(decoded) > 500 else decoded
        raise ValueError(f"Could not find any final-answer marker {markers} in supervised GSM8K response: {preview!r}")

    answer_start = marker_start + len(marker)
    while answer_start < len(decoded) and decoded[answer_start].isspace():
        answer_start += 1

    cursor = 0
    selected_offset = None
    selected_span = None
    for offset, text in enumerate(token_texts):
        start = cursor
        end = cursor + len(text)
        cursor = end
        if end > answer_start:
            selected_offset = offset
            selected_span = (start, end)
            break

    if selected_offset is None:
        raise ValueError("Could not map final-answer marker to a supervised token position.")

    selected_position = label_positions[selected_offset : selected_offset + 1]
    selected_token_id = int(target_ids[selected_offset])
    selected_text = token_texts[selected_offset]
    return selected_position, {
        "update_token_mode": "first_final_answer_token",
        "final_answer_marker": marker,
        "final_answer_marker_char_start": marker_start,
        "final_answer_content_char_start": answer_start,
        "selected_supervised_offset": int(selected_offset),
        "selected_label_position": int(selected_position.item()),
        "selected_token_id": selected_token_id,
        "selected_token_text": selected_text,
        "selected_token_span": selected_span,
        "num_original_supervised_tokens": len(target_ids),
    }


def _select_update_label_positions(
    sample: SupervisedSample,
    tokenizer: object | None,
    config: SequenceScoreConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    label_positions = torch.tensor(list(iter_supervised_positions(sample.as_model_sample())), dtype=torch.long)
    if label_positions.numel() == 0:
        raise ValueError("GSM8K sample has no supervised tokens.")
    if config.update_token_mode == "all_supervised":
        return label_positions, {
            "update_token_mode": "all_supervised",
            "num_original_supervised_tokens": int(label_positions.numel()),
            "num_selected_supervised_tokens": int(label_positions.numel()),
        }
    if config.update_token_mode == "first_response_token":
        if tokenizer is None:
            raise ValueError("tokenizer is required for update_token_mode=first_response_token.")
        selected, info = _select_first_response_position(tokenizer, sample.labels, label_positions)
        info["num_selected_supervised_tokens"] = int(selected.numel())
        return selected, info
    if config.update_token_mode == "first_final_answer_token":
        if tokenizer is None:
            raise ValueError("tokenizer is required for update_token_mode=first_final_answer_token.")
        selected, info = _select_first_final_answer_position(tokenizer, sample.labels, label_positions, config.final_answer_markers)
        info["num_selected_supervised_tokens"] = int(selected.numel())
        return selected, info
    raise ValueError(f"Unsupported update_token_mode: {config.update_token_mode}")


@torch.no_grad()
def extract_update_features(
    model: object,
    sample: SupervisedSample,
    config: SequenceScoreConfig,
    tokenizer: object | None = None,
) -> UpdateFeatures:
    model.eval()
    device = _model_device(model)
    input_ids = sample.input_ids.to(device)
    attention_mask = sample.attention_mask.to(device)
    labels = sample.labels.to(device)

    label_positions, token_selection_info = _select_update_label_positions(sample, tokenizer, config)
    logit_positions = label_positions - 1
    target_ids = labels[label_positions.to(device)].long()

    outputs = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        output_hidden_states=False,
        output_attentions=False,
        use_cache=False,
    )
    logits = outputs.logits[0, logit_positions.to(device)]
    if config.gradient_convention == "logprob":
        gradients = logprob_gradient(logits, target_ids)
    elif config.gradient_convention == "nll":
        gradients = nll_gradient(logits, target_ids)
    else:
        raise ValueError(f"Unsupported gradient convention: {config.gradient_convention}")

    readout_weight = get_readout_weight(model).detach()
    projected = project_output_gradients(gradients, readout_weight)
    hidden = NormalizedHiddenExtractor(model).extract(input_ids, attention_mask, logit_positions).hidden

    return UpdateFeatures(
        input_ids=sample.input_ids.detach().cpu(),
        attention_mask=sample.attention_mask.detach().cpu(),
        label_positions=label_positions.detach().cpu(),
        logit_positions=logit_positions.detach().cpu(),
        target_ids=target_ids.detach().cpu(),
        gradients=gradients.detach().cpu(),
        projected_gradients=projected.detach().cpu(),
        hidden=hidden,
        metadata={**sample.metadata, **token_selection_info},
    )


@torch.no_grad()
def extract_observation_features(
    model: object,
    tokenizer: object,
    template: object,
    observation: MMLUObservation,
    config: SequenceScoreConfig,
) -> ObservationFeatures:
    model.eval()
    device = _model_device(model)
    prompt_ids = observation.prompt_ids.to(device)
    attention_mask = observation.attention_mask.to(device)

    outputs = model(
        input_ids=prompt_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        output_hidden_states=False,
        output_attentions=False,
        use_cache=False,
    )
    logits = outputs.logits[0, -1]

    token_info = resolve_option_token_ids(tokenizer, template, observation.prompt_text)
    option_ids = [int(token_info[letter]["token_id"]) for letter in ("A", "B", "C", "D")]
    option_grads = option_gradients(logits, option_ids, config.gradient_convention)
    ifmass_grad = allowed_token_mass_gradient(logits, option_ids, config.gradient_convention)

    readout_weight = get_readout_weight(model).detach()
    projected_options = project_output_gradients(option_grads, readout_weight)
    projected_ifmass = project_output_gradients(ifmass_grad, readout_weight)

    final_logit_pos = torch.tensor([prompt_ids.numel() - 1], dtype=torch.long)
    hidden = NormalizedHiddenExtractor(model).extract(prompt_ids, attention_mask, final_logit_pos).hidden[0]

    return ObservationFeatures(
        prompt_ids=observation.prompt_ids.detach().cpu(),
        attention_mask=observation.attention_mask.detach().cpu(),
        option_token_ids=option_ids,
        option_token_info=token_info,
        option_gradients=option_grads.detach().cpu(),
        projected_option_gradients=projected_options.detach().cpu(),
        ifmass_gradient=ifmass_grad.detach().cpu(),
        projected_ifmass_gradient=projected_ifmass.detach().cpu(),
        hidden=hidden,
        logits=logits.detach().cpu(),
        metadata=observation.metadata,
    )
