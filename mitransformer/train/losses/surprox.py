
import torch

from . import utils

from typing import Literal


def cosine_loss(
        logits: torch.Tensor,
        label_ids: torch.Tensor,
        ignore_index: int | None = -100,
        reduction: Literal["sum", "mean", "none"] = "mean",
        prefix_dummies: int = 2) -> torch.Tensor:

    if ignore_index is not None:
        label_ids = label_ids.clone()
        label_ids[label_ids == ignore_index] = 0

    cost: torch.Tensor = torch.max(
        logits, dim=-1) - torch.gather(
            logits, -1, label_ids).squeeze(-1)  # type: ignore

    if prefix_dummies > 1:
        cost[..., :prefix_dummies-1] = 0

    if label_ids is not None and ignore_index is not None:
        cost[label_ids == ignore_index] = 0

    zeros = cost.new_zeros([*cost.shape[:-1], 1])
    cost = torch.cat((zeros, cost[..., :-1]), dim=-1)

    return utils.reduce(cost, reduction=reduction)
