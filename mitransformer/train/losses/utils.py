import torch

from typing import Literal


def reduce(
        x: torch.Tensor,
        reduction: Literal["sum", "mean", "none"]) -> torch.Tensor:
    match reduction:
        case "none":
            return x
        case "mean":
            return x.mean()
        case "sum":
            return x.sum()
        case _:
            raise Exception(
                "Parameter 'reduction' must be one of "
                f"{locals()['__annotations__']['reduction']}.")


def entropy(
        probs: torch.Tensor,
        reduction: Literal["sum", "none"] = "sum",
        mask_zero: bool = False,
        mask_triangular: bool = False,
        eps: float = 1e-12) -> torch.Tensor:
    entropy = -probs * torch.log2(probs.clamp_min(eps))

    if mask_triangular == "triangular":
        entropy = torch.tril(entropy)

    if mask_zero == "zero":
        entropy = entropy.masked_fill(probs == 0, 0)

    return reduce(entropy, reduction)


def shift_ignore_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    input/outputs: [..., S].
    Prepends 'False' to front of sequence and cuts of last element"""

    out = torch.zeros_like(mask)
    out[..., 1:] = mask[..., :-1].clone()
    return out
