
import torch
import torch.nn.functional as F

from . import utils

from typing import Literal


def cosine_loss(
        embeddings: torch.Tensor,
        activations: torch.Tensor,
        label_ids: torch.Tensor | None = None,
        ignore_index: int = -100,
        reduction: Literal["sum", "mean", "none"] = "mean",
        prefix_dummies: int = 2) -> torch.Tensor:
    """_summary_

    Parameters
    ----------
    embeddings : torch.Tensor
        shape [..., S, E]
    activations : torch.Tensor
        shape [..., S, E]
    labels : torch.Tensor
        _description_
    ignore_index : int, optional
        _description_, by default -100
    reduction : Literal["sum", "mean", "none"], optional
        _description_, by default "mean"
    discriminative : bool, optional
        _description_, by default False

    Returns
    -------
    torch.Tensor
        shape [..., S] if reduction = 'none'
    """
    loss = (1 - F.cosine_similarity(
        embeddings[..., 1:, :], activations[..., :-1, :], dim=-1))/2

    if prefix_dummies > 1:
        loss[..., :prefix_dummies-1] = 0

    zeros = loss.new_zeros([*loss.shape[:-1], 1])
    loss = torch.cat((zeros, loss), dim=-1)

    if label_ids is not None:
        loss[utils.shift_ignore_mask(label_ids == ignore_index)] = 0

    return utils.reduce(loss, reduction=reduction)
