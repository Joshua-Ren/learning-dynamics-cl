import sys

import torch
import torch.nn.functional as F

from utils.hidden_records import parse_hidden_layers
from utils.one_step_forgetting import iter_supervised_positions




def _safe_compute_dtype(dtype):
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def _model_compute_dtype(model):
    for param in model.parameters():
        if param.is_floating_point():
            return _safe_compute_dtype(param.dtype)
    return torch.float32


def _tensor_compute_dtype(*tensors):
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.dtype == torch.float64:
            return torch.float64
    return torch.float32

def _model_device(model):
    return next(model.parameters()).device


def _as_1d_tensor(value, device, name):
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.tensor(value, dtype=torch.long)
    if tensor.dim() != 1:
        raise ValueError(f"sample['{name}'] must have shape [seq_len].")
    return tensor.to(device)


def _normalize_label_positions(sample, label_positions):
    if label_positions is None:
        return list(iter_supervised_positions(sample))
    positions = [int(pos) for pos in label_positions]
    if any(pos <= 0 for pos in positions):
        raise ValueError("label positions must be positive because logits[pos - 1] scores labels[pos].")
    return positions


def _sample_id(sample, default=None):
    for key in ("probe_id", "example_id", "sample_id", "raw_index"):
        if key in sample:
            return sample[key]
    return default


@torch.no_grad()
def extract_ch_factors(model, sample, layer=-1, label_positions=None):
    """
    Extract channel-1 factors for supervised causal-LM target positions.

    Position convention:
        label_pos is the target token position.
        logit_pos = hidden_pos = label_pos - 1.

    Returns tensors on CPU:
        label_positions: [N]
        logit_positions: [N]
        target_ids: [N]
        hidden: [N, d]
        residual: [N, V], where residual = one_hot(target) - softmax(logits)
        log_probs: [N]
    """
    if "input_ids" not in sample or "labels" not in sample:
        raise KeyError("sample must contain input_ids and labels.")

    device = _model_device(model)
    input_ids = _as_1d_tensor(sample["input_ids"], device, "input_ids").unsqueeze(0)
    labels = _as_1d_tensor(sample["labels"], device, "labels")
    attention_mask = sample.get("attention_mask")
    if attention_mask is not None:
        attention_mask = _as_1d_tensor(attention_mask, device, "attention_mask")

    positions = _normalize_label_positions(
        {
            "labels": labels.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu() if attention_mask is not None else None,
        },
        label_positions,
    )
    seq_len = labels.shape[0]
    for pos in positions:
        if pos >= seq_len:
            raise IndexError(f"label_pos={pos} is out of range for seq_len={seq_len}.")
        if int(labels[pos].item()) == -100:
            raise ValueError(f"label_pos={pos} is ignored by labels == -100.")
        if attention_mask is not None:
            if int(attention_mask[pos].item()) != 1 or int(attention_mask[pos - 1].item()) != 1:
                raise ValueError(f"label_pos={pos} or logit_pos={pos - 1} is masked out.")

    model.eval()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask.unsqueeze(0) if attention_mask is not None else None,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states
    selected_layers = parse_hidden_layers(str(layer), len(hidden_states))
    if len(selected_layers) != 1:
        raise ValueError("extract_ch_factors expects exactly one layer.")
    selected_layer = selected_layers[0]

    if not positions:
        vocab_size = outputs.logits.shape[-1]
        hidden_dim = hidden_states[selected_layer].shape[-1]
        return {
            "layer": selected_layer,
            "label_positions": torch.empty(0, dtype=torch.long),
            "logit_positions": torch.empty(0, dtype=torch.long),
            "target_ids": torch.empty(0, dtype=torch.long),
            "hidden": torch.empty(0, hidden_dim, dtype=_model_compute_dtype(model)),
            "residual": torch.empty(0, vocab_size, dtype=_model_compute_dtype(model)),
            "log_probs": torch.empty(0, dtype=_model_compute_dtype(model)),
        }

    label_pos = torch.tensor(positions, dtype=torch.long, device=device)
    logit_pos = label_pos - 1
    target_ids = labels[label_pos].long()

    logits_device = outputs.logits.device
    logits_logit_pos = logit_pos.to(logits_device)
    logits_target_ids = target_ids.to(logits_device)
    compute_dtype = _model_compute_dtype(model)
    selected_logits = outputs.logits[0, logits_logit_pos].to(dtype=compute_dtype)
    pi = F.softmax(selected_logits, dim=-1)
    residual = -pi.clone()
    row_index = torch.arange(logits_target_ids.numel(), device=logits_device)
    residual[row_index, logits_target_ids] += 1.0
    log_probs = F.log_softmax(selected_logits, dim=-1)[
        row_index,
        logits_target_ids,
    ]
    selected_hidden = hidden_states[selected_layer]
    hidden_logit_pos = logit_pos.to(selected_hidden.device)
    hidden = selected_hidden[0, hidden_logit_pos].to(dtype=compute_dtype)

    return {
        "layer": selected_layer,
        "label_positions": label_pos.detach().cpu(),
        "logit_positions": logit_pos.detach().cpu(),
        "target_ids": target_ids.detach().cpu(),
        "hidden": hidden.detach().cpu(),
        "residual": residual.detach().cpu(),
        "logits": selected_logits.detach().cpu(),
        "log_probs": log_probs.detach().cpu(),
    }


def compute_ch1_from_factors(obs_factors, update_factors, return_parts=False):
    """
    Compute ch1_l(o, u) = <g_o, g_u> * <h_o^l, h_u^l>.
    """
    compute_dtype = _tensor_compute_dtype(
        obs_factors["hidden"],
        update_factors["hidden"],
        obs_factors["residual"],
        update_factors["residual"],
    )
    obs_hidden = obs_factors["hidden"].to(dtype=compute_dtype)
    update_hidden = update_factors["hidden"].to(dtype=compute_dtype)
    obs_residual = obs_factors["residual"].to(dtype=compute_dtype)
    update_residual = update_factors["residual"].to(dtype=compute_dtype)

    if obs_hidden.dim() != 2 or update_hidden.dim() != 2:
        raise ValueError("hidden factors must have shape [N, d].")
    if obs_residual.dim() != 2 or update_residual.dim() != 2:
        raise ValueError("residual factors must have shape [N, vocab_size].")
    if obs_hidden.shape[1] != update_hidden.shape[1]:
        raise ValueError("hidden dimensions do not match.")
    if obs_residual.shape[1] != update_residual.shape[1]:
        raise ValueError("residual vocabulary dimensions do not match.")

    gg = obs_residual @ update_residual.T
    hh = obs_hidden @ update_hidden.T
    ch1 = gg * hh
    if return_parts:
        return ch1, gg, hh
    return ch1


def get_readout_weight(model):
    for attr_name in ("lm_head", "embed_out", "output_projection"):
        readout = getattr(model, attr_name, None)
        weight = getattr(readout, "weight", None)
        if weight is not None:
            return weight
    raise AttributeError("Could not find lm_head/embed_out/output_projection weight.")


def _get_transformer_layers(model):
    """Return the model's repeated transformer blocks when exposed by common HF layouts."""
    candidates = [
        getattr(model, "model", None),
        getattr(model, "transformer", None),
        getattr(model, "gpt_neox", None),
        model,
    ]
    for container in candidates:
        if container is None:
            continue
        for attr_name in ("layers", "h", "blocks"):
            layers = getattr(container, attr_name, None)
            if layers is not None:
                return layers
    return None


def _rms_normalize_hidden(hidden, eps=1e-6):
    """Fallback RMSNorm without learned scale: x / sqrt(mean(x^2) + eps)."""
    return hidden * torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + eps)


def _get_rmsnorm_eps(model):
    config = getattr(model, "config", None)
    for attr_name in ("rms_norm_eps", "layer_norm_epsilon", "norm_eps"):
        value = getattr(config, attr_name, None)
        if value is not None:
            return float(value)
    return 1e-6


def _append_if_residual_dim(streams, tensor, residual_dim):
    """Append [N, d] stream tensors that match the residual-flow hidden size."""
    if tensor is not None and tensor.shape[-1] == residual_dim:
        streams.append(tensor.detach().to(dtype=_safe_compute_dtype(tensor.dtype), device="cpu"))


def _select_positions_from_stream(stream, logit_pos):
    """
    Select token positions from a hidden/residual stream without assuming shape.

    Supports [batch, seq_len, hidden_dim] and [seq_len, hidden_dim]. If the
    captured stream is shorter than the requested positions or has an unexpected
    rank, return None so callers can skip that stream instead of triggering a
    CUDA index assert.
    """
    if stream is None or stream.dim() not in (2, 3):
        return None

    pos_cpu = logit_pos.detach().cpu()
    if pos_cpu.numel() == 0:
        return stream.new_empty((0, stream.shape[-1]))

    if stream.dim() == 3:
        if stream.shape[0] < 1:
            return None
        seq_len = stream.shape[1]
    else:
        seq_len = stream.shape[0]

    min_pos = int(pos_cpu.min().item())
    max_pos = int(pos_cpu.max().item())
    if min_pos < 0 or max_pos >= seq_len:
        return None

    stream_pos = logit_pos.to(stream.device)
    if stream.dim() == 3:
        return stream[0, stream_pos]
    return stream[stream_pos]


def _select_positions_from_hook_stream(stream, logit_pos):
    """
    CPU-safe positional selection for tensors captured by forward hooks.

    Some model implementations expose hooked MLP norm streams with sequence
    layouts that can differ from outputs.hidden_states. We validate bounds on
    CPU and also perform the actual indexing on CPU, avoiding CUDA device-side
    asserts from bad advanced indexing. Returned tensor is [N, hidden_dim] on
    CPU, or None when the stream cannot match requested logit positions.
    """
    if stream is None or stream.dim() not in (2, 3):
        return None

    pos_cpu = logit_pos.detach().cpu()
    if pos_cpu.numel() == 0:
        return torch.empty(0, stream.shape[-1], dtype=_safe_compute_dtype(stream.dtype))

    if stream.dim() == 3:
        if stream.shape[0] < 1:
            return None
        seq_len = stream.shape[1]
    else:
        seq_len = stream.shape[0]

    min_pos = int(pos_cpu.min().item())
    max_pos = int(pos_cpu.max().item())
    if min_pos < 0 or max_pos >= seq_len:
        return None

    stream_cpu = stream.detach().to(dtype=_safe_compute_dtype(stream.dtype), device="cpu")
    if stream.dim() == 3:
        return stream_cpu[0, pos_cpu]
    return stream_cpu[pos_cpu]


def _block_input_norm(model, layer_index):
    layers = _get_transformer_layers(model)
    if layers is None or layer_index < 0 or layer_index >= len(layers):
        return None
    block = layers[layer_index]
    for attr_name in ("input_layernorm", "ln_1", "input_norm"):
        norm = getattr(block, attr_name, None)
        if norm is not None:
            return norm
    return None


def _block_mlp_input_norm(model, layer_index):
    layers = _get_transformer_layers(model)
    if layers is None or layer_index < 0 or layer_index >= len(layers):
        return None
    block = layers[layer_index]
    for attr_name in (
        "post_attention_layernorm",
        "post_attn_layernorm",
        "post_attention_norm",
        "ln_2",
        "mlp_norm",
    ):
        norm = getattr(block, attr_name, None)
        if norm is not None:
            return norm
    return None


def _resolve_overlap_layers(layer, num_hidden_states):
    """
    Resolve block-input layers used in sum_ell <h_tilde_ell,o, h_tilde_ell,u>.

    outputs.hidden_states has length num_blocks + 1. hidden_states[ell] is the
    residual stream entering block ell for ell in [0, num_blocks - 1].
    layer=-1 keeps the Eq. (9) default and sums over all block-input layers.
    Other layer specs reuse parse_hidden_layers and keep valid block-input ids.
    """
    num_blocks = max(num_hidden_states - 1, 0)
    if str(layer).strip() == "-1":
        return list(range(num_blocks))
    selected = parse_hidden_layers(str(layer), num_hidden_states)
    return [idx for idx in selected if 0 <= idx < num_blocks]


def _extract_normalized_block_inputs(model, sample, label_positions, layer=-1, include_mlp=True):
    """
    Extract normalized residual streams used in the ch2_approx layer factor.

    For each supervised target label_pos, the relevant context position is
    logit_pos = label_pos - 1. For every selected transformer block ell, this
    returns h_tilde_attn_ell and, when include_mlp=True and available,
    h_tilde_mlp_ell.

    Returns: [num_streams, N, hidden_dim] on CPU.
    """
    device = _model_device(model)
    input_ids = _as_1d_tensor(sample["input_ids"], device, "input_ids").unsqueeze(0)
    attention_mask = sample.get("attention_mask")
    if attention_mask is not None:
        attention_mask = _as_1d_tensor(attention_mask, device, "attention_mask")
    positions = _normalize_label_positions(
        {
            "labels": _as_1d_tensor(sample["labels"], device, "labels").detach().cpu(),
            "attention_mask": attention_mask.detach().cpu() if attention_mask is not None else None,
        },
        label_positions,
    )
    logit_pos = torch.tensor(positions, dtype=torch.long, device=device) - 1

    layers = _get_transformer_layers(model)
    num_hidden_states_hint = len(layers) + 1 if layers is not None else 0
    preselected_layers = _resolve_overlap_layers(layer, num_hidden_states_hint) if num_hidden_states_hint else []

    mlp_norm_outputs = {}
    hook_handles = []

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            # h_tilde_mlp_ell: output of the norm applied to the residual stream
            # after self-attention and before the MLP sublayer.
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            mlp_norm_outputs[layer_idx] = tensor.detach()
        return hook

    if include_mlp and layers is not None:
        for layer_idx in preselected_layers:
            norm = _block_mlp_input_norm(model, layer_idx)
            if norm is not None:
                hook_handles.append(norm.register_forward_hook(make_hook(layer_idx)))

    model.eval()
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask.unsqueeze(0) if attention_mask is not None else None,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
            )
    finally:
        for handle in hook_handles:
            handle.remove()

    hidden_states = outputs.hidden_states
    overlap_layers = preselected_layers or _resolve_overlap_layers(layer, len(hidden_states))
    if not overlap_layers:
        hidden_dim = hidden_states[0].shape[-1]
        return torch.empty(0, len(positions), hidden_dim, dtype=_model_compute_dtype(model))

    eps = _get_rmsnorm_eps(model)
    residual_dim = hidden_states[overlap_layers[0]].shape[-1]
    normalized = []
    for layer_idx in overlap_layers:
        # h_tilde_attn_ell: RMS-normalized residual entering self-attention.
        layer_hidden = hidden_states[layer_idx]
        attn_input = _select_positions_from_stream(layer_hidden, logit_pos)
        if attn_input is not None:
            attn_norm = _block_input_norm(model, layer_idx)
            if attn_norm is not None:
                # Keep native dtype for learned RMSNorm weights, then cast to fp32.
                attn_input = attn_norm(attn_input)
            else:
                attn_input = _rms_normalize_hidden(attn_input.to(dtype=_safe_compute_dtype(attn_input.dtype)), eps=eps)
            _append_if_residual_dim(normalized, attn_input, residual_dim)

        if include_mlp:
            # h_tilde_mlp_ell: RMS-normalized residual entering the MLP.
            mlp_input = _select_positions_from_hook_stream(mlp_norm_outputs.get(layer_idx), logit_pos)
            _append_if_residual_dim(normalized, mlp_input, residual_dim)

    if not normalized:
        return torch.empty(0, len(positions), residual_dim, dtype=_model_compute_dtype(model))
    return torch.stack(normalized, dim=0)


def extract_normalized_block_inputs(model, sample, label_positions, layer=-1):
    """
    Extract RMS-normalized self-attention and MLP input streams.

    Returns: [num_streams, N, hidden_dim] on CPU, with streams ordered as
    [attn_0, mlp_0, attn_1, mlp_1, ...] when MLP input norms are available.
    """
    return _extract_normalized_block_inputs(
        model,
        sample,
        label_positions=label_positions,
        layer=layer,
        include_mlp=True,
    )


def extract_normalized_attn_inputs(model, sample, label_positions, layer=-1):
    """
    Extract h_tilde_attn_ell only: the RMS-normalized residual stream entering
    each selected block's self-attention sublayer.

    Returns: [num_layers, N, hidden_dim] on CPU.
    """
    return _extract_normalized_block_inputs(
        model,
        sample,
        label_positions=label_positions,
        layer=layer,
        include_mlp=False,
    )


def extract_block_inputs_wo_rms(model, sample, label_positions, layer=-1):
    """
    Extract raw pre-RMS residual streams for the ch2_approx_wo_rms variant.

    For each supervised target label_pos, logit_pos = label_pos - 1. For every
    selected transformer block ell, this returns h_attn_ell, the raw residual
    entering self-attention, and h_mlp_ell, the raw residual entering the MLP
    input RMSNorm when that norm is exposed.

    Returns: [num_streams, N, hidden_dim] on CPU.
    """
    device = _model_device(model)
    input_ids = _as_1d_tensor(sample["input_ids"], device, "input_ids").unsqueeze(0)
    attention_mask = sample.get("attention_mask")
    if attention_mask is not None:
        attention_mask = _as_1d_tensor(attention_mask, device, "attention_mask")
    positions = _normalize_label_positions(
        {
            "labels": _as_1d_tensor(sample["labels"], device, "labels").detach().cpu(),
            "attention_mask": attention_mask.detach().cpu() if attention_mask is not None else None,
        },
        label_positions,
    )
    logit_pos = torch.tensor(positions, dtype=torch.long, device=device) - 1

    layers = _get_transformer_layers(model)
    num_hidden_states_hint = len(layers) + 1 if layers is not None else 0
    preselected_layers = _resolve_overlap_layers(layer, num_hidden_states_hint) if num_hidden_states_hint else []

    mlp_norm_inputs = {}
    hook_handles = []

    def make_pre_hook(layer_idx):
        def hook(_module, inputs):
            # h_mlp_ell: raw residual stream before the MLP input RMSNorm.
            if inputs:
                mlp_norm_inputs[layer_idx] = inputs[0].detach()
        return hook

    if layers is not None:
        for layer_idx in preselected_layers:
            norm = _block_mlp_input_norm(model, layer_idx)
            if norm is not None:
                hook_handles.append(norm.register_forward_pre_hook(make_pre_hook(layer_idx)))

    model.eval()
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask.unsqueeze(0) if attention_mask is not None else None,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
            )
    finally:
        for handle in hook_handles:
            handle.remove()

    hidden_states = outputs.hidden_states
    overlap_layers = preselected_layers or _resolve_overlap_layers(layer, len(hidden_states))
    if not overlap_layers:
        hidden_dim = hidden_states[0].shape[-1]
        return torch.empty(0, len(positions), hidden_dim, dtype=_model_compute_dtype(model))

    residual_dim = hidden_states[overlap_layers[0]].shape[-1]
    raw_inputs = []
    for layer_idx in overlap_layers:
        # h_attn_ell: raw residual stream entering the self-attention sublayer.
        layer_hidden = hidden_states[layer_idx]
        attn_input = _select_positions_from_stream(layer_hidden, logit_pos)
        _append_if_residual_dim(raw_inputs, attn_input, residual_dim)

        # h_mlp_ell: raw residual stream entering the MLP input RMSNorm.
        mlp_input = _select_positions_from_hook_stream(mlp_norm_inputs.get(layer_idx), logit_pos)
        _append_if_residual_dim(raw_inputs, mlp_input, residual_dim)

    if not raw_inputs:
        return torch.empty(0, len(positions), residual_dim, dtype=_model_compute_dtype(model))
    return torch.stack(raw_inputs, dim=0)


def compute_layer_overlap_from_inputs(obs_inputs, update_inputs):
    """
    Compute sum_s <h_{s,o}, h_{s,u}> over stacked block streams.

    obs_inputs: [num_streams, N_obs, d]
    update_inputs: [num_streams, N_update, d]
    Returns: [N_obs, N_update]
    """
    if obs_inputs.dim() != 3 or update_inputs.dim() != 3:
        raise ValueError("block inputs must have shape [num_streams, N, hidden_dim].")
    if obs_inputs.shape[0] != update_inputs.shape[0]:
        raise ValueError("observation/update layer counts do not match.")
    if obs_inputs.shape[2] != update_inputs.shape[2]:
        raise ValueError("observation/update hidden dimensions do not match.")
    if obs_inputs.shape[0] == 0:
        return torch.zeros(obs_inputs.shape[1], update_inputs.shape[1], dtype=_tensor_compute_dtype(obs_inputs, update_inputs))
    compute_dtype = _tensor_compute_dtype(obs_inputs, update_inputs)
    return torch.einsum("lod,lud->ou", obs_inputs.to(dtype=compute_dtype), update_inputs.to(dtype=compute_dtype))


def compute_aggregated_layer_overlap_from_inputs(obs_inputs, update_inputs):
    """
    Compute <sum_s h_{s,o}, sum_s h_{s,u}> over stacked residual streams.

    obs_inputs: [num_streams, N_obs, d]
    update_inputs: [num_streams, N_update, d]
    Returns: [N_obs, N_update]
    """
    if obs_inputs.dim() != 3 or update_inputs.dim() != 3:
        raise ValueError("block inputs must have shape [num_streams, N, hidden_dim].")
    if obs_inputs.shape[2] != update_inputs.shape[2]:
        raise ValueError("observation/update hidden dimensions do not match.")
    if obs_inputs.shape[0] == 0 or update_inputs.shape[0] == 0:
        return torch.zeros(obs_inputs.shape[1], update_inputs.shape[1], dtype=_tensor_compute_dtype(obs_inputs, update_inputs))
    compute_dtype = _tensor_compute_dtype(obs_inputs, update_inputs)
    obs_sum = obs_inputs.to(dtype=compute_dtype).sum(dim=0)
    update_sum = update_inputs.to(dtype=compute_dtype).sum(dim=0)
    return obs_sum @ update_sum.T


def compute_readout_align_from_factors(model, obs_factors, update_factors):
    """
    Compute <W^T g_o, W^T g_u> for every observation/update token pair.

    g_t is the vocabulary residual e_y - pi. W has shape [vocab_size,
    hidden_dim], so g_t @ W corresponds to W^T g_t.
    """
    compute_dtype = _model_compute_dtype(model)
    readout_weight = get_readout_weight(model).detach().to(dtype=compute_dtype, device="cpu")
    obs_residual = obs_factors["residual"].to(dtype=compute_dtype)
    update_residual = update_factors["residual"].to(dtype=compute_dtype)

    if obs_residual.dim() != 2 or update_residual.dim() != 2:
        raise ValueError("residual factors must have shape [N, vocab_size].")
    if readout_weight.dim() != 2:
        raise ValueError("readout weight must have shape [vocab_size, hidden_dim].")
    if obs_residual.shape[1] != readout_weight.shape[0]:
        raise ValueError("observation residual vocabulary dimension does not match readout weight.")
    if update_residual.shape[1] != readout_weight.shape[0]:
        raise ValueError("update residual vocabulary dimension does not match readout weight.")

    obs_readout_direction = obs_residual @ readout_weight
    update_readout_direction = update_residual @ readout_weight
    return obs_readout_direction @ update_readout_direction.T


def compute_ch2_approx_from_factors(
    model,
    obs_factors,
    update_factors,
    obs_block_inputs=None,
    update_block_inputs=None,
):
    """
    Approximate channel 2 using Eq. (9)'s second term:
        readout_align = <W^T g_o, W^T g_u>
        layer_overlap = sum_s <h_tilde_{s,o}, h_tilde_{s,u}>
        ch2_approx(o, u) = readout_align * layer_overlap

    g_t is the vocabulary residual e_y - pi. W is the readout/lm_head matrix
    with shape [vocab_size, hidden_dim], so g_t @ W corresponds to W^T g_t.
    h_tilde inputs must have shape [num_streams, N, hidden_dim].
    """
    if obs_block_inputs is None or update_block_inputs is None:
        raise ValueError("obs_block_inputs and update_block_inputs are required for the new ch2_approx.")

    readout_align = compute_readout_align_from_factors(model, obs_factors, update_factors)
    layer_overlap = compute_layer_overlap_from_inputs(obs_block_inputs, update_block_inputs)
    return readout_align * layer_overlap


def compute_ch2_approx_variants_from_factors(
    model,
    obs_factors,
    update_factors,
    obs_block_inputs,
    update_block_inputs,
    obs_attn_inputs,
    update_attn_inputs,
    obs_block_inputs_wo_rms,
    update_block_inputs_wo_rms,
):
    """
    Compute ch2_approx variants for comparing layer-overlap choices.

    Returns tensors with shape [N_obs, N_update]:
      ch2_approx: readout_align times RMS-normalized attn+MLP overlap.
      ch2_approx_gwwg: readout_align = <W^T g_o, W^T g_u>.
      ch2_approx_singleh: readout_align times RMS-normalized attn-only overlap.
      ch2_approx_wo_rms: readout_align times raw pre-RMS attn+MLP overlap.
      ch2_approx_aggh: readout_align times aggregated RMS attn+MLP overlap.
    """
    readout_align = compute_readout_align_from_factors(model, obs_factors, update_factors)
    layer_overlap = compute_layer_overlap_from_inputs(obs_block_inputs, update_block_inputs)
    layer_overlap_singleh = compute_layer_overlap_from_inputs(obs_attn_inputs, update_attn_inputs)
    layer_overlap_wo_rms = compute_layer_overlap_from_inputs(
        obs_block_inputs_wo_rms,
        update_block_inputs_wo_rms,
    )
    layer_overlap_aggh = compute_aggregated_layer_overlap_from_inputs(
        obs_block_inputs,
        update_block_inputs,
    )
    return {
        "ch2_approx": readout_align * layer_overlap,
        "ch2_approx_gwwg": readout_align,
        "ch2_approx_singleh": readout_align * layer_overlap_singleh,
        "ch2_approx_wo_rms": readout_align * layer_overlap_wo_rms,
        "ch2_approx_aggh": readout_align * layer_overlap_aggh,
    }


def get_backbone_params(model, exclude_lm_head=False):
    params = []
    names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if exclude_lm_head and (
            "lm_head" in name or "embed_out" in name or "output_projection" in name or "embed_tokens" in name
        ):
            continue
        params.append(param)
        names.append(name)
    return params, names


def token_logprob(model, sample, logit_pos, target_id=None, device=None):
    """
    Compute log p(target_id | logits[logit_pos]) for a supervised sample.
    """
    device = device or _model_device(model)
    input_ids = _as_1d_tensor(sample["input_ids"], device, "input_ids").unsqueeze(0)
    attention_mask = sample.get("attention_mask")
    if attention_mask is not None:
        attention_mask = _as_1d_tensor(attention_mask, device, "attention_mask").unsqueeze(0)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=False,
        output_attentions=False,
        use_cache=False,
    )
    logits = outputs.logits[0, int(logit_pos)].to(dtype=_model_compute_dtype(model))

    if target_id is None:
        if int(logit_pos) + 1 >= input_ids.shape[1]:
            raise ValueError("target_id is required when logit_pos is the last position.")
        target_id = input_ids[0, int(logit_pos) + 1]
    target_id = torch.as_tensor(target_id, device=logits.device).long()
    return F.log_softmax(logits, dim=-1)[target_id]


def grad_inner_product(scalar_a, scalar_b, params):
    grads_a = torch.autograd.grad(
        scalar_a,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    grads_b = torch.autograd.grad(
        scalar_b,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    dot = scalar_a.new_tensor(0.0)
    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is None or grad_b is None:
            continue
        term = torch.sum(grad_a * grad_b)
        dot = dot + term.to(dot.device)
    return dot


def compute_ch2_backbone_pair(
    model,
    obs_sample,
    obs_label_pos,
    update_sample,
    update_label_pos,
    exclude_lm_head=False,
):
    """
    Exact notebook-faithful channel 2:
        <grad_phi log p(y_o|s_o), grad_phi log p(y_u|s_u)>

    phi excludes lm_head/readout parameters by default.
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    device = _model_device(model)
    params, _ = get_backbone_params(model, exclude_lm_head=exclude_lm_head)
    if not params:
        return torch.tensor(0.0, dtype=_model_compute_dtype(model))

    obs_target = _as_1d_tensor(obs_sample["labels"], device, "labels")[int(obs_label_pos)]
    update_target = _as_1d_tensor(update_sample["labels"], device, "labels")[int(update_label_pos)]
    logp_o = token_logprob(
        model,
        obs_sample,
        logit_pos=int(obs_label_pos) - 1,
        target_id=obs_target,
        device=device,
    )
    logp_u = token_logprob(
        model,
        update_sample,
        logit_pos=int(update_label_pos) - 1,
        target_id=update_target,
        device=device,
    )
    ch2 = grad_inner_product(logp_o, logp_u, params)
    model.zero_grad(set_to_none=True)
    return ch2.detach().to(dtype=_safe_compute_dtype(ch2.dtype), device="cpu")


def compute_ch12_real_pair(
    model,
    obs_sample,
    obs_label_pos,
    update_sample,
    update_label_pos,
):
    """
    Exact full-parameter channel score:
        <grad_theta log p(y_o|s_o), grad_theta log p(y_u|s_u)>

    theta includes every trainable model parameter, including embeddings,
    transformer blocks, final norms, and lm_head/readout.
    """
    return compute_ch2_backbone_pair(
        model=model,
        obs_sample=obs_sample,
        obs_label_pos=obs_label_pos,
        update_sample=update_sample,
        update_label_pos=update_label_pos,
        exclude_lm_head=False,
    )


def compute_ch1_ch2_for_update_token(
    model,
    update_sample,
    update_label_pos,
    probe_samples,
    layer=-1,
    max_probe_tokens_per_sample=None,
    exclude_lm_head=False,
):
    """
    Build per-probe-token diagnostic rows for one update token.
    """
    update_factors = extract_ch_factors(
        model,
        update_sample,
        layer=layer,
        label_positions=[update_label_pos],
    )
    update_block_inputs = extract_normalized_block_inputs(
        model,
        update_sample,
        label_positions=[update_label_pos],
        layer=layer,
    )
    update_attn_inputs = extract_normalized_attn_inputs(
        model,
        update_sample,
        label_positions=[update_label_pos],
        layer=layer,
    )
    update_block_inputs_wo_rms = extract_block_inputs_wo_rms(
        model,
        update_sample,
        label_positions=[update_label_pos],
        layer=layer,
    )
    rows = []

    for probe_index, probe_sample in enumerate(probe_samples):
        probe_positions = list(
            iter_supervised_positions(
                probe_sample,
                max_positions=max_probe_tokens_per_sample,
            )
        )
        if not probe_positions:
            continue

        obs_factors = extract_ch_factors(
            model,
            probe_sample,
            layer=layer,
            label_positions=probe_positions,
        )
        obs_block_inputs = extract_normalized_block_inputs(
            model,
            probe_sample,
            label_positions=probe_positions,
            layer=layer,
        )
        obs_attn_inputs = extract_normalized_attn_inputs(
            model,
            probe_sample,
            label_positions=probe_positions,
            layer=layer,
        )
        obs_block_inputs_wo_rms = extract_block_inputs_wo_rms(
            model,
            probe_sample,
            label_positions=probe_positions,
            layer=layer,
        )
        ch1_matrix, gg_matrix, hh_matrix = compute_ch1_from_factors(
            obs_factors,
            update_factors,
            return_parts=True,
        )
        ch1_values = ch1_matrix[:, 0]
        gg_values = gg_matrix[:, 0]
        hh_values = hh_matrix[:, 0]
        ch2_approx_variants = compute_ch2_approx_variants_from_factors(
            model,
            obs_factors,
            update_factors,
            obs_block_inputs=obs_block_inputs,
            update_block_inputs=update_block_inputs,
            obs_attn_inputs=obs_attn_inputs,
            update_attn_inputs=update_attn_inputs,
            obs_block_inputs_wo_rms=obs_block_inputs_wo_rms,
            update_block_inputs_wo_rms=update_block_inputs_wo_rms,
        )

        for local_idx, probe_label_pos in enumerate(probe_positions):
            ch1 = float(ch1_values[local_idx].item())
            gg = float(gg_values[local_idx].item())
            hh = float(hh_values[local_idx].item())
            ch2_tensor = compute_ch2_backbone_pair(
                model=model,
                obs_sample=probe_sample,
                obs_label_pos=probe_label_pos,
                update_sample=update_sample,
                update_label_pos=update_label_pos,
                exclude_lm_head=True,
            )
            ch2 = float(ch2_tensor.item())
            ch12_real_tensor =  compute_ch2_backbone_pair(
                model=model,
                obs_sample=probe_sample,
                obs_label_pos=probe_label_pos,
                update_sample=update_sample,
                update_label_pos=update_label_pos,
                exclude_lm_head=False,
            )
            # ch12_real_tensor = compute_ch12_real_pair(
            #     model=model,
            #     obs_sample=probe_sample,
            #     obs_label_pos=probe_label_pos,
            #     update_sample=update_sample,
            #     update_label_pos=update_label_pos,
            # )
            ch12_real = float(ch12_real_tensor.item())
            ch2_approx = float(ch2_approx_variants["ch2_approx"][local_idx, 0].item())
            ch2_approx_gwwg = float(ch2_approx_variants["ch2_approx_gwwg"][local_idx, 0].item())
            ch2_approx_singleh = float(ch2_approx_variants["ch2_approx_singleh"][local_idx, 0].item())
            ch2_approx_wo_rms = float(ch2_approx_variants["ch2_approx_wo_rms"][local_idx, 0].item())
            ch2_approx_aggh = float(ch2_approx_variants["ch2_approx_aggh"][local_idx, 0].item())
            rows.append(
                {
                    "probe_index": probe_index,
                    "probe_id": _sample_id(probe_sample, probe_index),
                    "probe_dataset": probe_sample.get("dataset"),
                    "probe_domain": probe_sample.get("domain"),
                    "probe_example_id": probe_sample.get("example_id"),
                    "probe_raw_index": probe_sample.get("raw_index"),
                    "probe_label_pos": int(probe_label_pos),
                    "probe_logit_pos": int(probe_label_pos) - 1,
                    "probe_target_id": int(obs_factors["target_ids"][local_idx].item()),
                    "probe_logp_before": float(obs_factors["log_probs"][local_idx].item()),
                    "update_label_pos": int(update_label_pos),
                    "update_logit_pos": int(update_label_pos) - 1,
                    "update_target_id": int(update_factors["target_ids"][0].item()),
                    "layer": int(update_factors["layer"]),
                    "ch1": ch1,
                    "gg": gg,
                    "hh": hh,
                    "ch2": ch2,
                    "ch12_real": ch12_real,
                    "ch2_approx": ch2_approx,
                    "ch2_approx_gwwg": ch2_approx_gwwg,
                    "ch2_approx_singleh": ch2_approx_singleh,
                    "ch2_approx_wo_rms": ch2_approx_wo_rms,
                    "ch2_approx_aggh": ch2_approx_aggh,
                    "ch1_plus_ch2": ch1 + ch2,
                }
            )

    return rows


def _dry_test_channel1_handwritten():
    obs = {
        "hidden": torch.tensor([[3.0, 4.0], [1.0, 0.0]]),
        "residual": torch.tensor([[1.0, -1.0, 0.0], [0.0, 2.0, -1.0]]),
    }
    update = {
        "hidden": torch.tensor([[0.0, 5.0], [2.0, 0.0]]),
        "residual": torch.tensor([[2.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]),
    }

    pi = torch.tensor([0.7, 0.2, 0.1])
    target_y = 1
    token_g = -pi.clone()
    token_g[target_y] += 1.0
    expected_token_g = torch.tensor([-0.7, 0.8, -0.1])
    wrong_sign_token_g = pi.clone()
    wrong_sign_token_g[target_y] -= 1.0
    assert torch.allclose(token_g, expected_token_g)
    assert not torch.allclose(token_g, wrong_sign_token_g)

    ch1, g_dot, h_dot = compute_ch1_from_factors(obs, update, return_parts=True)
    h_norm_obs = obs["hidden"].norm(dim=-1, keepdim=True)
    h_norm_update = update["hidden"].norm(dim=-1, keepdim=True)
    h_cos = h_dot / (h_norm_obs * h_norm_update.T)

    expected_g_dot = torch.tensor([[1.0, -1.0], [2.0, -1.0]])
    expected_h_dot = torch.tensor([[20.0, 6.0], [0.0, 2.0]])
    expected_h_cos = torch.tensor([[0.8, 0.6], [0.0, 1.0]])
    expected_ch1 = torch.tensor([[20.0, -6.0], [0.0, -2.0]])

    assert torch.allclose(g_dot, expected_g_dot)
    assert torch.allclose(h_dot, expected_h_dot)
    assert torch.allclose(h_cos, expected_h_cos)
    assert torch.allclose(ch1, expected_ch1)

    print("channel1_handwritten dry test passed")
    print(f"token_g={token_g.tolist()}")
    print(f"expected_token_g={expected_token_g.tolist()}")
    print(f"wrong_sign_token_g={wrong_sign_token_g.tolist()}")
    print(f"g_dot={g_dot.tolist()}")
    print(f"h_dot={h_dot.tolist()}")
    print(f"h_cos={h_cos.tolist()}")
    print(f"ch1={ch1.tolist()}")


def _dry_test():
    _dry_test_channel1_handwritten()

    class ShiftCheckModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        ):
            batch_size, seq_len = input_ids.shape
            vocab = 16
            logits = torch.full(
                (batch_size, seq_len, vocab),
                -10.0,
                dtype=torch.float32,
                device=input_ids.device,
            )
            logits[:, 0, 5] = 10.0
            logits[:, 1, 5] = -10.0
            logits[:, 1, 6] = 10.0
            logits = logits + self.anchor
            hidden = torch.arange(seq_len * 3, dtype=torch.float32, device=input_ids.device)
            hidden = hidden.reshape(1, seq_len, 3).repeat(batch_size, 1, 1)
            hidden_states = (hidden, hidden + 1.0)
            return type("ToyOutput", (), {"logits": logits, "hidden_states": hidden_states})()

    sample = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "labels": torch.tensor([-100, 5, -100]),
    }
    factors = extract_ch_factors(ShiftCheckModel(), sample, layer=-1)
    assert factors["label_positions"].tolist() == [1]
    assert factors["logit_positions"].tolist() == [0]
    assert factors["target_ids"].tolist() == [5]
    pi = torch.softmax(torch.tensor([-10.0] * 16).index_fill(0, torch.tensor([5]), 10.0), dim=-1)
    expected_residual = -pi
    expected_residual[5] += 1.0
    assert torch.allclose(factors["residual"][0], expected_residual, atol=1e-6)
    assert factors["hidden"].tolist() == [[1.0, 2.0, 3.0]]

    obs = {
        "hidden": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "residual": torch.tensor([[1.0, -1.0], [0.5, 0.5]]),
    }
    upd = {
        "hidden": torch.tensor([[2.0, 1.0], [-1.0, 1.0]]),
        "residual": torch.tensor([[2.0, 0.0], [1.0, 1.0]]),
    }
    ch1, gg, hh = compute_ch1_from_factors(obs, upd, return_parts=True)
    assert torch.allclose(gg, obs["residual"] @ upd["residual"].T)
    assert torch.allclose(hh, obs["hidden"] @ upd["hidden"].T)
    assert torch.allclose(ch1, gg * hh)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2, bias=False)
            self.lm_head = torch.nn.Linear(2, 3, bias=False)

        def forward(
            self,
            input_ids,
            attention_mask=None,
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
        ):
            one_hot = F.one_hot(input_ids % 2, num_classes=2).float()
            hidden = self.backbone(one_hot)
            logits = self.lm_head(hidden)
            hidden_states = (hidden,)
            return type("ToyOutput", (), {"logits": logits, "hidden_states": hidden_states})()

    tiny = TinyModel()
    params, names = get_backbone_params(tiny, exclude_lm_head=True)
    assert names == ["backbone.weight"]
    assert params == [tiny.backbone.weight]
    params_all, names_all = get_backbone_params(tiny, exclude_lm_head=False)
    assert "lm_head.weight" in names_all and len(params_all) == 2

    toy_sample = {
        "input_ids": torch.tensor([0, 1, 0]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "labels": torch.tensor([-100, 1, 2]),
    }
    logp_a = token_logprob(tiny, toy_sample, logit_pos=0, target_id=1)
    logp_b = token_logprob(tiny, toy_sample, logit_pos=1, target_id=2)
    manual_grads_a = torch.autograd.grad(logp_a, [tiny.backbone.weight], retain_graph=True)
    manual_grads_b = torch.autograd.grad(logp_b, [tiny.backbone.weight])
    manual_dot = sum((a * b).sum() for a, b in zip(manual_grads_a, manual_grads_b))
    ch2 = compute_ch2_backbone_pair(tiny, toy_sample, 1, toy_sample, 2)
    assert torch.allclose(ch2, manual_dot.detach().float().cpu())

    print("ch1_ch2_metrics dry test passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--channel1-only":
        _dry_test_channel1_handwritten()
    else:
        _dry_test()
