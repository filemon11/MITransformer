import torch
import math

from typing import Literal

LOG2E = 1.0 / math.log(2.0)


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
        reduction: Literal["sum", "none"] = "sum") -> torch.Tensor:
    entropy = -torch.xlogy(probs, probs) / LOG2E
    return reduce(entropy, reduction)


def shift_ignore_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    input/outputs: [..., S].
    Prepends 'False' to front of sequence and cuts of last element"""

    out = torch.zeros_like(mask)
    out[..., 1:] = mask[..., :-1].clone()
    return out
