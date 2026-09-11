import torch


def parse_hidden_layers(layer_arg, n_hidden_states):
    """
    Resolve a layer selector against a model output.hidden_states tuple.

    Supports:
        - "all"
        - "last"
        - "-1"
        - "0,6,12,18,24"
    """
    layer_arg = str(layer_arg).strip()
    if layer_arg == "all":
        return list(range(n_hidden_states))
    if layer_arg == "last":
        return [n_hidden_states - 1]

    layers = []
    for item in layer_arg.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0:
            idx = n_hidden_states + idx
        if idx < 0 or idx >= n_hidden_states:
            raise ValueError(
                f"Layer index {item} is out of range for {n_hidden_states} hidden-state tensors."
            )
        layers.append(idx)
    if not layers:
        raise ValueError("No hidden-state layers were selected.")
    return layers


def _as_batch_tensor(value, device):
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.tensor(value)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def _batch_item(value, sequence_index):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.dim() == 0:
            return value.item()
        item = value[sequence_index]
        return item.item() if item.dim() == 0 else item.tolist()
    if isinstance(value, (list, tuple)):
        return value[sequence_index]
    return value


def _decode_token(tokenizer, token_id):
    if tokenizer is None:
        return None
    return tokenizer.decode([int(token_id)])


def _empty_hidden_by_layer(hidden_states, selected_layers):
    return {
        layer_idx: torch.empty(
            0,
            hidden_states[layer_idx].shape[-1],
            dtype=hidden_states[layer_idx].dtype,
            device="cpu",
        )
        for layer_idx in selected_layers
    }


@torch.no_grad()
def extract_token_hidden_records(model, batch, tokenizer=None, layers="-1"):
    """
    Extract hidden vectors for supervised label-token positions in one model batch.

    Position convention:
        hidden_pos = label_pos - 1

    Args:
        model: causal LM.
        batch: dict containing input_ids, labels, and optionally attention_mask,
            sample_index, or example_id.
        tokenizer: optional tokenizer for token string metadata.
        layers: "all", "last", "-1", or a comma-separated list such as
            "0,6,12,18,24".

    Returns:
        metadata: list[dict], one entry per selected label token.
        hidden_by_layer: dict[int, torch.Tensor], each [num_records, hidden_dim].
    """
    if "input_ids" not in batch or "labels" not in batch:
        raise KeyError("batch must contain input_ids and labels.")

    device = next(model.parameters()).device
    input_ids = _as_batch_tensor(batch["input_ids"], device)
    labels = _as_batch_tensor(batch["labels"], device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = _as_batch_tensor(attention_mask, device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        output_attentions=False,
    )
    hidden_states = outputs.hidden_states
    selected_layers = parse_hidden_layers(layers, len(hidden_states))

    metadata = []
    selected_vectors = {layer_idx: [] for layer_idx in selected_layers}

    batch_size, seq_len = labels.shape
    for sequence_index in range(batch_size):
        for label_pos in range(1, seq_len):
            if labels[sequence_index, label_pos].item() == -100:
                continue

            hidden_pos = label_pos - 1
            if attention_mask is not None:
                if attention_mask[sequence_index, hidden_pos].item() != 1:
                    continue
                if attention_mask[sequence_index, label_pos].item() != 1:
                    continue

            input_token_id = int(input_ids[sequence_index, hidden_pos].item())
            label_token_id = int(labels[sequence_index, label_pos].item())
            row = {
                "sequence_index": sequence_index,
                "hidden_pos": hidden_pos,
                "label_pos": label_pos,
                "input_token_id": input_token_id,
                "label_token_id": label_token_id,
                "input_token_str": _decode_token(tokenizer, input_token_id),
                "label_token_str": _decode_token(tokenizer, label_token_id),
            }
            if attention_mask is not None:
                row["hidden_attention_mask"] = int(attention_mask[sequence_index, hidden_pos].item())
                row["label_attention_mask"] = int(attention_mask[sequence_index, label_pos].item())

            sample_index = _batch_item(batch.get("sample_index"), sequence_index)
            example_id = _batch_item(batch.get("example_id"), sequence_index)
            raw_index = _batch_item(batch.get("raw_index"), sequence_index)
            dataset = _batch_item(batch.get("dataset"), sequence_index)
            domain = _batch_item(batch.get("domain"), sequence_index)
            if sample_index is not None:
                row["sample_index"] = sample_index
            if example_id is not None:
                row["example_id"] = example_id
            if raw_index is not None:
                row["raw_index"] = raw_index
            if dataset is not None:
                row["dataset"] = dataset
            if domain is not None:
                row["domain"] = domain

            metadata.append(row)
            for layer_idx in selected_layers:
                selected_vectors[layer_idx].append(
                    hidden_states[layer_idx][sequence_index, hidden_pos].detach().cpu()
                )

    hidden_by_layer = {}
    for layer_idx in selected_layers:
        if selected_vectors[layer_idx]:
            hidden_by_layer[layer_idx] = torch.stack(selected_vectors[layer_idx], dim=0)
        else:
            hidden_by_layer[layer_idx] = _empty_hidden_by_layer(hidden_states, [layer_idx])[layer_idx]

    return metadata, hidden_by_layer


def print_hidden_record_debug(metadata, hidden_by_layer, max_metadata=10):
    selected_layers = list(hidden_by_layer.keys())
    print("\nHidden record debug:")
    print(f"  number of collected records: {len(metadata)}")
    print(f"  selected layers: {selected_layers}")
    for layer_idx in selected_layers:
        print(f"  hidden_by_layer[{layer_idx}].shape: {tuple(hidden_by_layer[layer_idx].shape)}")

    print(f"  first {min(max_metadata, len(metadata))} metadata entries:")
    for row in metadata[:max_metadata]:
        print(f"    {row}")
