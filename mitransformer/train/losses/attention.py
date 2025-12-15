import torch
from . import utils

from typing import Literal


def shift_ignore_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    input/outputs: [..., S].
    Prepends 'False' to front of sequence and cuts of last element"""

    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    return mask


def attention_entropy_loss(
        arc_distributions: torch.Tensor,
        to_ignore_mask: torch.Tensor | Literal["triangular"] | None,
        reduction: Literal["sum", "mean", "none"] = "mean",
        label_ids: torch.Tensor | None = None,
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

    if to_ignore_mask is not None and to_ignore_mask != "triangular":
        to_ignore_mask = to_ignore_mask.sum(0).to(torch.bool)  # type: ignore

    if global_distr:
        arc_distributions = get_head_averaged_distribution(arc_distributions)
        # [B, H, S, S] -> [B, S, S]

    entropy = get_attention_entropy(
        arc_distributions, to_ignore_mask, reduction="none",
        include_current=include_current, length_weighted=length_weighted,
        prefix_dummies=prefix_dummies)  # -> [B, S] or [B, H, S]

    if not global_distr:
        entropy = entropy.mean(-2)  # [B, H, S] -> [B, S]

    if label_ids is not None:
        entropy[shift_ignore_mask(label_ids == ignore_index)] = 0

    reduced = utils.reduce(entropy, reduction)
    return reduced


def distance_loss(
        probs: torch.Tensor,
        to_ignore_mask: torch.Tensor | Literal["triangular"] | None,
        reduction: Literal["sum", "mean", "none"] = "mean",
        label_ids: torch.Tensor | None = None,
        ignore_index: int = -100,
        global_distr: bool = True,
        length_weighted: bool = False,
        prefix_dummies: int = 2,
        ) -> torch.Tensor:
    """input shape [..., H, S, S] with
    H: number of heads,
    B: batch size,
    S: sequence length.

    output shape
    [..., S] if reduction = 'none'
    else scalar"""
    if to_ignore_mask is not None:
        if to_ignore_mask == "triangular":
            probs = torch.tril(probs)
        else:
            to_ignore_mask = to_ignore_mask.sum(  # type: ignore
                0).to(torch.bool)
            probs[to_ignore_mask] = 0

    if global_distr:
        probs = get_head_averaged_distribution(probs)
        # [..., H, S, S] -> [..., S, S]

    s = probs.shape[-1]
    r = torch.arange(1, s+1-prefix_dummies, device=probs.device)

    if prefix_dummies > 0:
        prefix = torch.zeros(prefix_dummies, device=probs.device)
        r = torch.concat(
            (prefix, r))

    dist_mat = -1 * (r.repeat(s, 1) - r.reshape(-1, 1))     # + 1
    dist_mat = torch.tril(dist_mat)      # dist_mat.log())
    dist_mat[..., :prefix_dummies] = 0
    # [S, S]

    distances = dist_mat*probs
    # [..., S, S] or [..., H, S, S]

    distances = (distances).sum(-1)  # -> [..., S] or [..., H, S]

    if not global_distr:
        distances = distances.mean(-2)  # [..., H, S] -> [..., S]

    if length_weighted:
        norm_vector = torch.arange(
            0, distances.shape[-1]-prefix_dummies,
            dtype=torch.float,
            device=distances.device)

        if prefix_dummies > 0:
            prefix = torch.zeros(prefix_dummies, device=distances.device)
            norm_vector = torch.concat(
                (prefix, norm_vector))

        norm_vector = norm_vector.clamp(min=1e-4)

        distances = torch.div(distances, norm_vector)

    if label_ids is not None:
        distances[shift_ignore_mask(label_ids == ignore_index)] = 0
    return utils.reduce(distances, reduction)


def get_head_averaged_distribution(
        probs: torch.Tensor
        ) -> torch.Tensor:
    """input: [B, H, S, S]"""
    probs = probs.mean(-3)  # [B, S, S]
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
    output shape: [...] if reduction is 'mean' or 'sum', else [..., S].

    Assumes normalised distribution."""

    entropy = utils.entropy(probs, "none")
    # [..., S, S]
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

    entropy = entropy.sum(-1)
    # [..., S]

    if length_weighted:
        start_at = 1 if include_current else 0
        norm_vector = torch.arange(
            start_at, entropy.shape[-1]+start_at-prefix_dummies,
            dtype=torch.float,
            device=entropy.device)

        if prefix_dummies > 0:
            prefix = torch.zeros(prefix_dummies, device=entropy.device)
            norm_vector = torch.concat(
                (prefix, norm_vector))
        # [S] ([0?, ..., 1, 2, 3, ...])
        norm_vector = torch.log2(norm_vector).clamp(min=1e-4)

        entropy = torch.div(entropy, norm_vector)
        # [..., S]

        del norm_vector

        # prevent numerical problem for one-item
        # distribution
        entropy[..., (1-start_at)+prefix_dummies] = 1

    return utils.reduce(entropy, reduction)
