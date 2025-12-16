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
        reduction: Literal["sum", "none"] = "sum") -> torch.Tensor:
    logprobs = torch.log2(probs.clamp(min=1e-4))
    # attention: this was originally log_e
    entropy = -(probs*logprobs)
    return reduce(entropy, reduction)


def shift_ignore_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    input/outputs: [..., S].
    Prepends 'False' to front of sequence and cuts of last element"""

    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    return mask
