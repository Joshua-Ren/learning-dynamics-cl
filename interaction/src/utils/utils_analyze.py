import torch
import torch.nn.functional as F

# -------------- The code in this part is mainly used for find_correct_metrics.ipynb,
# Hence it is possible that this part have many developing functions that is not aligned
# with tracking_metrics.py
# However, some basic calculations between them are identical
# !!!!!!! Label shifting is implemented outside these functions.
def topk_entropy(logits, k=100):
    """
    calculate top_k entropy of each context

    Args:
        logits: [L, V]
        k: top-k

    Returns:
        entropy: [L]
    """
    L, V = logits.shape

    if k==-1:
        k=logits.shape[-1]
    topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)  # [L, k]

    topk_probs = F.softmax(topk_values, dim=-1)  # [L, k]

    entropy = -torch.sum(topk_probs * torch.log(topk_probs + 1e-10), dim=-1)  # [L]

    return entropy

def top12_prob_diff(logits, labels):
    """
    Args:
        logits: [L, V]

    Returns:
        diff: [L], top1_prob - top2_prob
        top1_probs: [L], top1 prob
        top2_probs: [L], top2 prob
        yprob: [L], prob of label
    """
    L, V = logits.shape

    probs = F.softmax(logits, dim=-1)  # [L, V]

    top2_probs, top2_indices = torch.topk(probs, k=2, dim=-1)  # [L, 2]

    top1_probs = top2_probs[:, 0]  # [L]
    top2_probs = top2_probs[:, 1]  # [L]

    diff = top1_probs - top2_probs  # [L]

    labels_expanded = labels.unsqueeze(-1)
    label_probs = probs.gather(dim=-1, index=labels_expanded)  # [L, 1]
    yprob = label_probs.squeeze(-1)  # [L]

    return diff, top1_probs, top2_probs, yprob

def get_label_rank_practical(logits, labels, max_rank_to_compute=1000):
    """
        get the rank of labels
    """
    L, V = logits.shape

    probs = F.softmax(logits, dim=-1)
    label_probs = probs[torch.arange(L), labels]

    k = min(max_rank_to_compute, V)
    topk_values, topk_indices = torch.topk(probs, k=k, dim=-1)

    # init all rank to k+1
    ranks = torch.full((L,), k + 1, dtype=torch.long, device=logits.device)

    # check if each candiates among top-k
    for i in range(L):
        matches = (topk_indices[i] == labels[i]).nonzero(as_tuple=True)
        if len(matches[0]) > 0:
            ranks[i] = matches[0][0] + 1  # change to ranking starting from 1

    return ranks