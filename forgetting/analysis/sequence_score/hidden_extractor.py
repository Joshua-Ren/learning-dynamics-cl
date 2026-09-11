from dataclasses import dataclass

import torch

from utils.ch1_ch2_metrics import _block_input_norm, _get_rmsnorm_eps, _get_transformer_layers, _rms_normalize_hidden


@dataclass
class HiddenExtractionResult:
    hidden: torch.Tensor
    hooked_module_names: list[str]
    selected_layers: list[int]


class NormalizedHiddenExtractor:
    """Extract one RMS-normalized attention-input residual stream per transformer block."""

    def __init__(self, model: object):
        self.model = model
        self.layers = _get_transformer_layers(model)
        if self.layers is None:
            raise ValueError("Could not locate transformer layers on model.")

    def module_names(self) -> list[str]:
        names = dict(self.model.named_modules()) if hasattr(self.model, "named_modules") else {}
        reverse = {id(module): name for name, module in names.items()}
        out = []
        for idx in range(len(self.layers)):
            norm = _block_input_norm(self.model, idx)
            out.append(reverse.get(id(norm), f"layers.{idx}.input_layernorm") if norm is not None else f"layers.{idx}.manual_rmsnorm")
        return out

    @torch.no_grad()
    def extract(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, logit_positions: torch.Tensor) -> HiddenExtractionResult:
        device = next(self.model.parameters()).device
        input_ids = input_ids.reshape(1, -1).to(device)
        attention = attention_mask.reshape(1, -1).to(device) if attention_mask is not None else None
        positions = logit_positions.reshape(-1).long().to(device)
        if positions.numel() == 0:
            raise ValueError("No logit positions requested.")
        if int(positions.min().item()) < 0 or int(positions.max().item()) >= input_ids.shape[1]:
            raise ValueError("Requested logit position is outside the input sequence.")

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        num_layers = len(self.layers)
        if len(hidden_states) < num_layers:
            raise ValueError("Model returned fewer hidden-state tensors than transformer layers.")

        eps = _get_rmsnorm_eps(self.model)
        per_layer = []
        selected_layers = list(range(num_layers))
        for layer_idx in selected_layers:
            raw = hidden_states[layer_idx][0, positions]
            norm = _block_input_norm(self.model, layer_idx)
            if norm is not None:
                normalized = norm(raw)
            else:
                normalized = _rms_normalize_hidden(raw.float(), eps=eps)
            per_layer.append(normalized.detach().float().cpu())

        stacked = torch.stack(per_layer, dim=1)
        return HiddenExtractionResult(
            hidden=stacked,
            hooked_module_names=self.module_names(),
            selected_layers=selected_layers,
        )
