import random

import torch


def enumerate_pair_indices(num_records, include_self=False, directed=False):
    """
    Enumerate record-index pairs as a tensor with shape [num_pairs, 2].

    Args:
        num_records: number of hidden records.
        include_self: include (i, i) pairs.
        directed: include both (i, j) and (j, i). If false, only i < j
            pairs are returned, plus self-pairs when include_self is true.
    """
    if num_records < 0:
        raise ValueError("num_records must be non-negative.")

    pairs = []
    for left in range(num_records):
        start = 0 if directed else left
        for right in range(start, num_records):
            if left == right and not include_self:
                continue
            if not directed and right < left:
                continue
            pairs.append((left, right))

    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)


def sample_pair_indices(
    num_records,
    max_pairs,
    seed=0,
    include_self=False,
    directed=False,
):
    """
    Deterministically sample record-index pairs.

    If max_pairs is None or at least the number of candidate pairs, all candidate
    pairs are returned in deterministic enumeration order.
    """
    candidates = enumerate_pair_indices(
        num_records=num_records,
        include_self=include_self,
        directed=directed,
    )

    if max_pairs is None or max_pairs >= len(candidates):
        return candidates
    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative or None.")
    if max_pairs == 0:
        return torch.empty(0, 2, dtype=torch.long)

    rng = random.Random(seed)
    selected = sorted(rng.sample(range(len(candidates)), k=max_pairs))
    return candidates[selected]


def _metadata_group(metadata_row):
    if "example_id" in metadata_row:
        return ("example_id", metadata_row["example_id"])
    if "sample_index" in metadata_row:
        return ("sample_index", metadata_row["sample_index"])
    if "sequence_index" in metadata_row:
        return ("sequence_index", metadata_row["sequence_index"])
    return None


def filter_pairs_by_metadata(metadata, pair_indices, same_sample=None):
    """
    Filter pairs by metadata grouping.

    Args:
        metadata: list of token-record metadata rows.
        pair_indices: tensor/list with shape [num_pairs, 2].
        same_sample:
            - None: keep all pairs.
            - True: keep only pairs with the same available group.
            - False: keep only pairs with different available groups.

    Group priority is example_id, then sample_index, then sequence_index.
    Rows without any group are kept only when same_sample is None.
    """
    pairs = torch.as_tensor(pair_indices, dtype=torch.long)
    if pairs.dim() != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_indices must have shape [num_pairs, 2].")
    if same_sample is None:
        return pairs

    kept = []
    for left, right in pairs.tolist():
        left_group = _metadata_group(metadata[left])
        right_group = _metadata_group(metadata[right])
        if left_group is None or right_group is None:
            continue
        is_same = left_group == right_group
        if is_same == same_sample:
            kept.append((left, right))

    return torch.tensor(kept, dtype=torch.long).reshape(-1, 2)


def filter_pairs_by_dataset(metadata, pair_indices, different_dataset=None):
    """
    Filter pairs using metadata["dataset"].

    Args:
        different_dataset:
            - None: keep all pairs.
            - True: keep only pairs from different datasets.
            - False: keep only pairs from the same dataset.
    """
    pairs = torch.as_tensor(pair_indices, dtype=torch.long)
    if pairs.dim() != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_indices must have shape [num_pairs, 2].")
    if different_dataset is None:
        return pairs

    kept = []
    for left, right in pairs.tolist():
        left_dataset = metadata[left].get("dataset")
        right_dataset = metadata[right].get("dataset")
        if left_dataset is None or right_dataset is None:
            continue
        is_different = left_dataset != right_dataset
        if is_different == different_dataset:
            kept.append((left, right))

    return torch.tensor(kept, dtype=torch.long).reshape(-1, 2)


def sample_pair_indices_from_metadata(
    metadata,
    max_pairs,
    seed=0,
    include_self=False,
    directed=False,
    same_sample=None,
    different_dataset=None,
):
    """
    Deterministically sample pairs from token-record metadata.

    This function only chooses record indices. It does not compute HH metrics.
    """
    candidates = enumerate_pair_indices(
        num_records=len(metadata),
        include_self=include_self,
        directed=directed,
    )
    candidates = filter_pairs_by_metadata(
        metadata=metadata,
        pair_indices=candidates,
        same_sample=same_sample,
    )
    candidates = filter_pairs_by_dataset(
        metadata=metadata,
        pair_indices=candidates,
        different_dataset=different_dataset,
    )

    if max_pairs is None or max_pairs >= len(candidates):
        return candidates
    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative or None.")
    if max_pairs == 0:
        return torch.empty(0, 2, dtype=torch.long)

    rng = random.Random(seed)
    selected = sorted(rng.sample(range(len(candidates)), k=max_pairs))
    return candidates[selected]


def pair_indices_to_rows(pair_indices, metadata=None):
    """
    Convert pair indices to row dictionaries for later analysis outputs.
    """
    pairs = torch.as_tensor(pair_indices, dtype=torch.long)
    if pairs.dim() != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_indices must have shape [num_pairs, 2].")

    rows = []
    for pair_index, (left, right) in enumerate(pairs.tolist()):
        row = {
            "pair_index": pair_index,
            "left_index": left,
            "right_index": right,
        }
        if metadata is not None:
            row["left_metadata"] = metadata[left]
            row["right_metadata"] = metadata[right]
        rows.append(row)
    return rows


def _dry_test():
    pairs = enumerate_pair_indices(4)
    assert pairs.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    directed = enumerate_pair_indices(3, directed=True)
    assert directed.tolist() == [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]

    with_self = enumerate_pair_indices(3, include_self=True)
    assert with_self.tolist() == [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    sample_a = sample_pair_indices(5, max_pairs=4, seed=123)
    sample_b = sample_pair_indices(5, max_pairs=4, seed=123)
    assert torch.equal(sample_a, sample_b)
    assert sample_a.shape == (4, 2)

    metadata = [
        {"sample_index": 0, "label_pos": 1},
        {"sample_index": 0, "label_pos": 2},
        {"sample_index": 1, "label_pos": 1},
        {"sample_index": 1, "label_pos": 2},
    ]
    same = sample_pair_indices_from_metadata(metadata, max_pairs=None, same_sample=True)
    cross = sample_pair_indices_from_metadata(metadata, max_pairs=None, same_sample=False)
    assert same.tolist() == [[0, 1], [2, 3]]
    assert cross.tolist() == [[0, 2], [0, 3], [1, 2], [1, 3]]

    dataset_metadata = [
        {"sample_index": 0, "dataset": "gsm8k"},
        {"sample_index": 0, "dataset": "gsm8k"},
        {"sample_index": 1, "dataset": "mmlu"},
        {"sample_index": 1, "dataset": "mmlu"},
    ]
    cross_dataset = sample_pair_indices_from_metadata(
        dataset_metadata,
        max_pairs=None,
        different_dataset=True,
    )
    assert cross_dataset.tolist() == [[0, 2], [0, 3], [1, 2], [1, 3]]

    rows = pair_indices_to_rows(same, metadata=metadata)
    assert rows[0]["left_metadata"]["sample_index"] == 0
    assert rows[1]["right_metadata"]["label_pos"] == 2

    print("hh_sampling dry test passed")
    print(f"all pairs: {pairs.tolist()}")
    print(f"sampled pairs: {sample_a.tolist()}")
    print(f"same-sample pairs: {same.tolist()}")
    print(f"cross-sample pairs: {cross.tolist()}")
    print(f"cross-dataset pairs: {cross_dataset.tolist()}")


if __name__ == "__main__":
    _dry_test()
