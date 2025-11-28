import torch
from . import utils

from typing import Literal


def attention_entropy_loss(
        arc_distributions: torch.Tensor,
        to_ignore_mask: torch.Tensor | Literal["triangular"] | None,
        reduction: Literal["sum", "mean", "none"] = "mean",
        input_ids: torch.Tensor | None = None,
        ignore_index: int = -100,
        global_distr: bool = True,
        include_current: bool = True,
        length_weighted: bool = False,
        prefix_dummies: int = 2
        ) -> torch.Tensor:
    """input shape [B, H, S, S] with
    H: number of heads,
    B: batch size,
    S: sequence length.

    output shape
    [B, S] if reduction = 'none'
    else scalar"""

    probs = arc_distributions

    if to_ignore_mask is not None and to_ignore_mask != "triangular":
        to_ignore_mask = to_ignore_mask.sum(0).to(torch.bool)  # type: ignore

    if global_distr:
        probs = get_head_averaged_distribution(probs)
        # [B, H, S, S] -> [B, S, S]

    entropy = get_attention_entropy(
        probs, to_ignore_mask, reduction="none",
        include_current=include_current, length_weighted=length_weighted,
        prefix_dummies=prefix_dummies)  # -> [B, S] or [B, H, S]

    if not global_distr:
        entropy = entropy.mean(1)  # [B, H, S] -> [B, S]

    if input_ids is not None:
        entropy[input_ids == ignore_index] = 0

    reduced = utils.reduce(entropy, reduction)
    return reduced


def distance_loss(
        probs: torch.Tensor,
        to_ignore_mask: torch.Tensor | Literal["triangular"] | None,
        reduction: Literal["sum", "mean", "none"] = "mean",
        input_ids: torch.Tensor | None = None,
        ignore_index: int = -100,
        global_distr: bool = True,
        include_current: bool = True,
        prefix_dummies: int = 2,
        ) -> torch.Tensor:
    """input shape [B, H, S, S] with
    H: number of heads,
    B: batch size,
    S: sequence length.

    output shape
    [B, S] if reduction = 'none'
    else scalar"""

    probs = normalise(probs, not include_current, prefix_dummies > 0)
    if to_ignore_mask is not None:
        if to_ignore_mask == "triangular":
            probs = torch.tril(probs)
        else:
            to_ignore_mask = to_ignore_mask.sum(  # type: ignore
                0).to(torch.bool)
            probs[to_ignore_mask] = 0

    if global_distr:
        probs = get_head_averaged_distribution(probs)
        # [B, H, S, S] -> [B, S, S]

    s = probs.shape[-1]
    r = torch.arange(1, s+1-prefix_dummies, device=probs.device)

    if prefix_dummies > 0:
        prefix = torch.zeros(prefix_dummies, device=probs.device)
        r = torch.concat(
            (prefix, r))

    dist_mat = -1 * (r.repeat(s, 1) - r.reshape(-1, 1))     # + 1
    dist_mat = torch.tril(dist_mat)      # dist_mat.log())
    dist_mat[..., :prefix_dummies] = 0

    dist_mat = dist_mat.unsqueeze(0)  # -> [B, S, S] or [B, H, S, S]

    distances = dist_mat*probs
    cost = (distances).sum(-1)  # -> [B, S] or [B, H, S]

    if not global_distr:
        cost = cost.mean(1)  # [B, H, S] -> [B, S]

    if input_ids is not None:
        cost[input_ids == ignore_index] = 0

    return utils.reduce(cost, reduction)


def get_head_averaged_distribution(
        probs: torch.Tensor
        ) -> torch.Tensor:
    """input: [B, H, S, S]"""
    probs = probs.mean(1)  # [B, S, S]
    return probs


def normalise_without_diagonal(
        probs: torch.Tensor) -> torch.Tensor:
    probs = torch.tril(probs, diagonal=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-4)
    return torch.tril(probs, diagonal=-1)


def normalise(
        probs: torch.Tensor,
        without_diagonal: bool = False,
        without_root_dummy: bool = False) -> torch.Tensor:
    if not without_diagonal and not without_root_dummy:
        return probs

    if without_diagonal:
        probs = torch.tril(probs, diagonal=-1)
    if without_root_dummy:
        probs[..., :2] = 0
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-4)

    if without_diagonal:
        probs = torch.tril(probs, diagonal=-1)
    if without_root_dummy:
        probs[..., :2] = 0
    return probs


def get_attention_entropy(
        probs: torch.Tensor,
        to_ignore: torch.Tensor | Literal["triangular"] | None = None,
        reduction: Literal["sum", "mean", "none"] = "mean",
        include_current: bool = True,
        length_weighted: bool = False,
        prefix_dummies: int = 2) -> torch.Tensor:
    """input shape [..., S, S]
    with S: sequence length.
    output shape: scalar if reduction is 'mean' or 'sum', else [..., S].

    Assumes normalised distribution."""

    entropy = utils.entropy(probs, "none")
    del probs

    if to_ignore is not None:
        if to_ignore == "triangular":
            entropy = entropy.masked_fill(
                torch.tril(
                    torch.ones(
                        *entropy.shape,
                        device=entropy.device)) == 0, 0)
        else:
            entropy[to_ignore] = 0  # type: ignore

    if length_weighted:
        start_at = 1 if include_current else 0
        norm_vector = torch.arange(
            start_at, entropy.shape[-2]+start_at-prefix_dummies,
            dtype=torch.float,
            device=entropy.device)

        if prefix_dummies > 0:
            prefix = torch.zeros(prefix_dummies, device=entropy.device)
            norm_vector = torch.concat(
                (prefix, norm_vector)).unsqueeze(0).t()
        # [S, S] ([[1], [2], [3], ...])
        norm_vector = torch.log2(norm_vector).clamp(min=1e-4)

        entropy = torch.div(entropy, norm_vector)

        # prevent numerical problem for one-item
        # distribution
        entropy = torch.div(entropy, norm_vector)
        entropy[..., (1-start_at)+prefix_dummies, prefix_dummies] = 1

        del norm_vector

    entropy = entropy.sum(-1)

    # TODO: ignore padding tokens

    reduced = utils.reduce(entropy, reduction)
    return reduced
