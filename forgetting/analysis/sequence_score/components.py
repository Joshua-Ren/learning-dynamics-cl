import torch

from .config import PairScoreResult


def _assert_shape(name: str, tensor: torch.Tensor, ndim: int) -> None:
    if tensor.dim() != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {tuple(tensor.shape)}.")


def score_from_components(
    gg: torch.Tensor,
    hh_all: torch.Tensor,
    gwwg: torch.Tensor,
    kembd: torch.Tensor,
    objective: str,
) -> PairScoreResult:
    """Compute CH1/CH2 scores from already extracted pair components.

    Shapes:
        gg: [T, A]
        hh_all: [T, L]
        gwwg: [T, A]
        kembd: [T]
    """
    _assert_shape("gg", gg, 2)
    _assert_shape("hh_all", hh_all, 2)
    _assert_shape("gwwg", gwwg, 2)
    _assert_shape("kembd", kembd, 1)

    if gg.shape != gwwg.shape:
        raise ValueError(f"gg and gwwg shapes differ: {tuple(gg.shape)} vs {tuple(gwwg.shape)}.")
    if gg.shape[0] != hh_all.shape[0] or gg.shape[0] != kembd.shape[0]:
        raise ValueError("Component token dimensions do not match.")
    if hh_all.shape[1] < 1:
        raise ValueError("hh_all must contain at least one layer.")

    gg = gg.float()
    hh_all = hh_all.float()
    gwwg = gwwg.float()
    kembd = kembd.float()

    hh_l = hh_all[:, -1]
    hh_others = hh_all[:, :-1].sum(dim=-1) if hh_all.shape[1] > 1 else torch.zeros_like(hh_l)

    ch1 = gg * hh_l[:, None]
    ch2_hidden = gwwg * hh_others[:, None]
    ch2_embd = gwwg * kembd[:, None]
    ch2 = ch2_hidden + ch2_embd
    total = ch1 + ch2

    ch1_by_token = ch1.mean(dim=1)
    ch2_hidden_by_token = ch2_hidden.mean(dim=1)
    ch2_embd_by_token = ch2_embd.mean(dim=1)
    ch2_by_token = ch2.mean(dim=1)
    total_by_token = total.mean(dim=1)
    num_tokens = int(gg.shape[0])

    if num_tokens == 0:
        raise ValueError("Cannot score an update example with zero supervised tokens.")

    return PairScoreResult(
        objective=objective,  # type: ignore[arg-type]
        ch1_sum=float(ch1_by_token.sum().item()),
        ch2_sum=float(ch2_by_token.sum().item()),
        ch2_hidden_sum=float(ch2_hidden_by_token.sum().item()),
        ch2_embd_sum=float(ch2_embd_by_token.sum().item()),
        total_sum=float(total_by_token.sum().item()),
        ch1_mean=float(ch1_by_token.mean().item()),
        ch2_mean=float(ch2_by_token.mean().item()),
        ch2_hidden_mean=float(ch2_hidden_by_token.mean().item()),
        ch2_embd_mean=float(ch2_embd_by_token.mean().item()),
        total_mean=float(total_by_token.mean().item()),
        num_update_tokens=num_tokens,
        mean_kembd=float(kembd.mean().item()),
        max_kembd=float(kembd.max().item()),
    )
