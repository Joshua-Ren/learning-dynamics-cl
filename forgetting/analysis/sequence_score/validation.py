import torch
import torch.nn.functional as F

from .output_gradients import allowed_token_mass_gradient, logprob_gradient


def validate_single_token_logprob_gradient(vocab_size: int = 11, target_id: int = 3, atol: float = 1e-6) -> dict:
    logits = torch.randn(vocab_size, dtype=torch.float64, requires_grad=True)
    objective = F.log_softmax(logits, dim=-1)[target_id]
    (autograd_grad,) = torch.autograd.grad(objective, logits)
    closed = logprob_gradient(logits.detach().float(), torch.tensor([target_id]))[0].double()
    ok = torch.allclose(autograd_grad, closed, atol=atol, rtol=0)
    return {"ok": bool(ok), "max_abs_error": float((autograd_grad - closed).abs().max().item())}


def validate_ifmass_gradient(
    vocab_size: int = 17,
    option_token_ids: tuple[int, int, int, int] = (1, 5, 9, 13),
    atol: float = 1e-6,
) -> dict:
    logits = torch.randn(vocab_size, dtype=torch.float64, requires_grad=True)
    probs = F.softmax(logits, dim=-1)
    objective = torch.log(probs[list(option_token_ids)].sum())
    (autograd_grad,) = torch.autograd.grad(objective, logits)
    closed = allowed_token_mass_gradient(logits.detach().float(), list(option_token_ids))[0].double()
    ok = torch.allclose(autograd_grad, closed, atol=atol, rtol=0)
    return {"ok": bool(ok), "max_abs_error": float((autograd_grad - closed).abs().max().item())}


def validate_causal_alignment(label_positions: torch.Tensor, logit_positions: torch.Tensor) -> dict:
    expected = label_positions - 1
    ok = torch.equal(expected.cpu(), logit_positions.cpu())
    return {
        "ok": bool(ok),
        "num_positions": int(label_positions.numel()),
        "first_label_pos": int(label_positions[0].item()) if label_positions.numel() else None,
        "first_logit_pos": int(logit_positions[0].item()) if logit_positions.numel() else None,
    }


def run_closed_form_validations() -> dict:
    return {
        "single_token_logprob": validate_single_token_logprob_gradient(),
        "ifmass": validate_ifmass_gradient(),
    }
