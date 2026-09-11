import torch
import torch.nn.functional as F


def logprob_gradient(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Return d log p(target) / d logits = one_hot(target) - softmax(logits)."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    target_ids = target_ids.reshape(-1).to(logits.device)
    if logits.shape[0] != target_ids.numel():
        raise ValueError("logits rows must match target_ids length.")

    probs = F.softmax(logits.float(), dim=-1)
    grad = -probs
    grad[torch.arange(target_ids.numel(), device=logits.device), target_ids] += 1.0
    return grad


def nll_gradient(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    return -logprob_gradient(logits, target_ids)


def option_gradients(logits: torch.Tensor, option_token_ids: list[int], convention: str = "logprob") -> torch.Tensor:
    target_ids = torch.tensor(option_token_ids, dtype=torch.long, device=logits.device)
    expanded_logits = logits.reshape(1, -1).expand(target_ids.numel(), -1)
    if convention == "logprob":
        return logprob_gradient(expanded_logits, target_ids)
    if convention == "nll":
        return nll_gradient(expanded_logits, target_ids)
    raise ValueError(f"Unsupported gradient convention: {convention}")


def allowed_token_mass_gradient(logits: torch.Tensor, option_token_ids: list[int], convention: str = "logprob") -> torch.Tensor:
    """Gradient of log sum_a p(a) with respect to logits."""
    logits = logits.reshape(-1).float()
    probs = F.softmax(logits, dim=-1)
    option_ids = torch.tensor(option_token_ids, dtype=torch.long, device=logits.device)
    option_mass = probs[option_ids].sum()
    if float(option_mass.item()) <= 0.0:
        raise ValueError("Allowed-token probability mass is zero.")

    q = torch.zeros_like(probs)
    q[option_ids] = probs[option_ids] / option_mass
    grad = q - probs
    if convention == "logprob":
        return grad.unsqueeze(0)
    if convention == "nll":
        return (-grad).unsqueeze(0)
    raise ValueError(f"Unsupported gradient convention: {convention}")


def project_output_gradients(gradients: torch.Tensor, readout_weight: torch.Tensor) -> torch.Tensor:
    """Compute W^T g as g @ W for readout W shaped [vocab, hidden]."""
    if gradients.dim() != 2:
        raise ValueError("gradients must have shape [N, vocab].")
    if readout_weight.dim() != 2:
        raise ValueError("readout_weight must have shape [vocab, hidden].")
    if gradients.shape[1] != readout_weight.shape[0]:
        raise ValueError("Gradient vocabulary size does not match readout weight.")
    return gradients.float() @ readout_weight.float()
