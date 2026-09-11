import torch


def as_pair_index_tensor(pair_indices):
    """
    Normalize explicit pair indices to a long tensor with shape [num_pairs, 2].

    Stage 3 intentionally does not sample pairs. Callers must provide the pairs.
    """
    if torch.is_tensor(pair_indices):
        pairs = pair_indices.detach().cpu().long()
    else:
        pairs = torch.tensor(pair_indices, dtype=torch.long)

    if pairs.dim() != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_indices must have shape [num_pairs, 2].")
    return pairs


def compute_pair_metrics_for_layer(hidden, pair_indices, eps=1e-12):
    """
    Compute h_o^T h_u and cosine similarity for one layer.

    Args:
        hidden: tensor with shape [num_records, hidden_dim].
        pair_indices: explicit index pairs with shape [num_pairs, 2].
        eps: numerical stability floor for cosine normalization.

    Returns:
        dict[str, torch.Tensor] with one-dimensional CPU tensors:
            - left_index
            - right_index
            - dot
            - cosine
            - left_norm
            - right_norm
    """
    if hidden.dim() != 2:
        raise ValueError("hidden must have shape [num_records, hidden_dim].")

    pairs = as_pair_index_tensor(pair_indices)
    num_records = hidden.shape[0]
    if pairs.numel() > 0:
        min_idx = int(pairs.min().item())
        max_idx = int(pairs.max().item())
        if min_idx < 0 or max_idx >= num_records:
            raise IndexError(
                f"pair index out of range for {num_records} hidden records: [{min_idx}, {max_idx}]"
            )

    hidden_cpu = hidden.detach().cpu().float()
    left_index = pairs[:, 0]
    right_index = pairs[:, 1]

    left = hidden_cpu[left_index]
    right = hidden_cpu[right_index]
    dot = (left * right).sum(dim=-1)
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    cosine = compute_pairwise_cosine(hidden_cpu, pairs, centered=False, eps=eps)

    return {
        "left_index": left_index,
        "right_index": right_index,
        "dot": dot,
        "cosine": cosine,
        "left_norm": left_norm,
        "right_norm": right_norm,
    }


def compute_pairwise_cosine(hidden, pair_indices, centered=False, eps=1e-12):
    """
    Compute cosine similarity for explicit hidden-state pairs.

    If centered=True, subtract the record-wise mean vector before selecting pairs.
    """
    if hidden.dim() != 2:
        raise ValueError("hidden must have shape [num_records, hidden_dim].")

    pairs = as_pair_index_tensor(pair_indices)
    hidden_cpu = hidden.detach().cpu().float()
    if centered:
        hidden_cpu = hidden_cpu - hidden_cpu.mean(dim=0, keepdim=True)

    left = hidden_cpu[pairs[:, 0]]
    right = hidden_cpu[pairs[:, 1]]
    dot = (left * right).sum(dim=-1)
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    cosine = dot / (left_norm.clamp_min(eps) * right_norm.clamp_min(eps))
    return cosine.clamp(min=-1.0, max=1.0)


def compute_hidden_pair_metrics(hidden_by_layer, pair_indices, eps=1e-12):
    """
    Compute pairwise hidden alignment metrics for each layer.

    Args:
        hidden_by_layer: dict[int, torch.Tensor], each [num_records, hidden_dim].
        pair_indices: explicit pairs [num_pairs, 2].

    Returns:
        dict[int, dict[str, torch.Tensor]] keyed by layer index.
    """
    pairs = as_pair_index_tensor(pair_indices)
    return {
        layer_idx: compute_pair_metrics_for_layer(hidden, pairs, eps=eps)
        for layer_idx, hidden in hidden_by_layer.items()
    }


def summarize_pair_metrics(pair_metrics):
    """
    Build lightweight per-layer summaries for already-computed pair metrics.

    This is intentionally limited to metric summaries and does not write files or
    choose pair samples.
    """
    summaries = {}
    for layer_idx, metrics in pair_metrics.items():
        dot = metrics["dot"]
        cosine = metrics["cosine"]
        count = int(dot.numel())

        if count == 0:
            summaries[layer_idx] = {
                "count": 0,
                "dot_mean": None,
                "dot_min": None,
                "dot_max": None,
                "dot_negative_count": 0,
                "dot_negative_fraction": None,
                "cosine_mean": None,
                "cosine_min": None,
                "cosine_max": None,
                "cosine_negative_count": 0,
                "cosine_negative_fraction": None,
            }
            continue

        dot_negative = dot < 0
        cosine_negative = cosine < 0
        summaries[layer_idx] = {
            "count": count,
            "dot_mean": float(dot.mean().item()),
            "dot_min": float(dot.min().item()),
            "dot_max": float(dot.max().item()),
            "dot_negative_count": int(dot_negative.sum().item()),
            "dot_negative_fraction": float(dot_negative.float().mean().item()),
            "cosine_mean": float(cosine.mean().item()),
            "cosine_min": float(cosine.min().item()),
            "cosine_max": float(cosine.max().item()),
            "cosine_negative_count": int(cosine_negative.sum().item()),
            "cosine_negative_fraction": float(cosine_negative.float().mean().item()),
        }
    return summaries


def pair_metrics_to_rows(pair_metrics):
    """
    Convert pair metrics to long-form rows for later CSV/JSON writing.
    """
    rows = []
    for layer_idx, metrics in pair_metrics.items():
        num_pairs = int(metrics["dot"].numel())
        for pair_idx in range(num_pairs):
            rows.append(
                {
                    "layer": layer_idx,
                    "pair_index": pair_idx,
                    "left_index": int(metrics["left_index"][pair_idx].item()),
                    "right_index": int(metrics["right_index"][pair_idx].item()),
                    "dot": float(metrics["dot"][pair_idx].item()),
                    "cosine": float(metrics["cosine"][pair_idx].item()),
                    "left_norm": float(metrics["left_norm"][pair_idx].item()),
                    "right_norm": float(metrics["right_norm"][pair_idx].item()),
                }
            )
    return rows


def _dry_test():
    h = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    pairs = torch.tensor([[0, 1], [0, 2], [0, 3]])
    expected_dot = torch.tensor([0.0, -1.0, 1.0])
    expected_cosine = torch.tensor([0.0, -1.0, 1.0 / 2.0**0.5])

    pair_indices = as_pair_index_tensor(pairs)
    assert pair_indices.shape == (3, 2)

    layer_metrics = compute_pair_metrics_for_layer(h, pair_indices)
    assert torch.allclose(layer_metrics["dot"], expected_dot)
    assert torch.allclose(layer_metrics["cosine"], expected_cosine)

    metrics = {0: layer_metrics}
    summaries = summarize_pair_metrics(metrics)
    expected_dot_mean = float(expected_dot.mean().item())
    expected_cosine_mean = float(expected_cosine.mean().item())
    assert summaries[0]["count"] == 3
    assert summaries[0]["dot_mean"] == expected_dot_mean
    assert summaries[0]["dot_min"] == -1.0
    assert summaries[0]["dot_max"] == 1.0
    assert summaries[0]["dot_negative_count"] == 1
    assert abs(summaries[0]["dot_negative_fraction"] - 1.0 / 3.0) < 1e-7
    assert abs(summaries[0]["cosine_mean"] - expected_cosine_mean) < 1e-7
    assert summaries[0]["cosine_min"] == -1.0
    assert abs(summaries[0]["cosine_max"] - float(expected_cosine[-1].item())) < 1e-7
    assert summaries[0]["cosine_negative_count"] == 1
    assert abs(summaries[0]["cosine_negative_fraction"] - 1.0 / 3.0) < 1e-7
    print("toy dot/cosine/summary test passed")

    centered_cosine = compute_pairwise_cosine(h, pairs, centered=True)
    manual_centered_cosine = compute_pairwise_cosine(
        h - h.mean(dim=0, keepdim=True),
        pairs,
        centered=False,
    )
    assert torch.allclose(centered_cosine, manual_centered_cosine)
    print("centered cosine consistency test passed")

    h_with_zero = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    zero_pairs = torch.tensor([[0, 1], [0, 2]])
    zero_cosine = compute_pairwise_cosine(h_with_zero, zero_pairs)
    assert torch.isfinite(zero_cosine).all()
    assert not torch.isnan(zero_cosine).any()
    print("zero-vector stability test passed")

    hidden_by_layer = {
        0: h,
        1: torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [-2.0, 0.0],
                [2.0, 2.0],
            ]
        ),
    }
    layer_metrics = compute_hidden_pair_metrics(hidden_by_layer, pairs)
    layer_summaries = summarize_pair_metrics(layer_metrics)
    assert sorted(layer_metrics.keys()) == [0, 1]
    assert torch.allclose(layer_metrics[0]["dot"], torch.tensor([0.0, -1.0, 1.0]))
    assert torch.allclose(layer_metrics[1]["dot"], torch.tensor([0.0, -4.0, 4.0]))
    assert layer_summaries[0]["dot_mean"] == 0.0
    assert layer_summaries[1]["dot_mean"] == 0.0
    assert layer_summaries[0]["dot_min"] == -1.0
    assert layer_summaries[1]["dot_min"] == -4.0
    print("layer aggregation test passed")

    rows = pair_metrics_to_rows(layer_metrics)
    assert len(rows) == 6
    assert rows[0]["layer"] == 0
    assert rows[1]["left_index"] == 0
    assert rows[1]["right_index"] == 2
    assert rows[1]["dot"] == -1.0
    assert rows[1]["cosine"] == -1.0

    print("hh_metrics dry test passed")
    print(f"layers: {sorted(layer_metrics.keys())}")
    print(f"num rows: {len(rows)}")
    print(f"layer 0 summary: {layer_summaries[0]}")
    print(f"layer 1 summary: {layer_summaries[1]}")


if __name__ == "__main__":
    _dry_test()
