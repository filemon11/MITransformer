import numpy as np
import torch
from ufal.chu_liu_edmonds import chu_liu_edmonds  # type: ignore

from typing import overload


@overload
def mst(score_matrix: np.ndarray) -> np.ndarray:
    ...


@overload
def mst(score_matrix: torch.Tensor) -> torch.Tensor:
    ...


def mst(
        score_matrix: np.ndarray | torch.Tensor
        ) -> np.ndarray | torch.Tensor:
    """Computes a maximum spanning tree for a (non)-stochastic
    adjacency matrix containing transition probabilities.
    Expects probabilities between 0 and 1. A row may
    sum to more than 1.

    Parameters
    ----------
    score_matrix : np.ndarray
        Stochastic matrix

    Returns
    -------
    np.ndarray
        Identified head positions of the nodes.
    """

    # Convert to logspace (addition corresponds to
    # multiplication of probabilities)
    if isinstance(score_matrix, np.ndarray):
        with np.errstate(divide='ignore'):
            # 0-probability entries may throw a division error
            # which we ignore
            score_matrix = np.log(score_matrix)
        assert isinstance(score_matrix, np.ndarray)
        heads, _ = chu_liu_edmonds(score_matrix.astype(np.double))
        return np.array(heads)
    else:
        score_matrix = torch.log(score_matrix)
        heads, _ = chu_liu_edmonds(
            score_matrix.cpu().numpy().astype(np.double))
        return torch.tensor(heads)


@overload
def merge_head_child_scores(
        head_scores: torch.Tensor,
        child_scores: torch.Tensor) -> torch.Tensor:
    ...


@overload
def merge_head_child_scores(
        head_scores: np.ndarray,
        child_scores: np.ndarray) -> np.ndarray:
    ...


def merge_head_child_scores(
        head_scores: torch.Tensor | np.ndarray,
        child_scores: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Merge governor and dependent scores (individual probabilities
    between 0 and 1). Throws away the diagonal and applies triangular mask.
    Supports batched input.

    Parameters
    ----------
    head_scores : np.ndarray | torch.Tensor
        _description_
    child_scores : np.ndarray | torch.Tensor
        _description_

    Returns
    -------
    np.ndarray | torch.Tensor
        The merged governor and dependent scores.
    """
    if isinstance(head_scores, torch.Tensor):
        assert isinstance(child_scores, torch.Tensor)
        ht = torch.tril(head_scores, -1)
        ct = torch.tril(child_scores, -1)
        return ht + ct.swapaxes(-1, -2)
    else:
        assert isinstance(head_scores, np.ndarray)
        assert isinstance(child_scores, np.ndarray)
        hn = np.tril(head_scores, -1)
        cn = np.tril(child_scores, -1)
        return hn + cn.swapaxes(-1, -2)


@overload
def dummy_mask_removal(mask: np.ndarray) -> np.ndarray:
    ...


@overload
def dummy_mask_removal(mask: torch.Tensor) -> torch.Tensor:
    ...


def dummy_mask_removal(
        mask: np.ndarray | torch.Tensor
        ) -> np.ndarray | torch.Tensor:
    """Removes dummy row and column from mask.
    Supports batched input.

    Parameters
    ----------
    mask : np.ndarray | torch.Tensor
        Input mask.

    Returns
    -------
    np.ndarray | torch.Tensor
        Input without dummy row and column.
    """
    return mask[..., 1:, 1:]


@overload
def mask_to_headlist(mask: np.ndarray) -> np.ndarray:
    ...


@overload
def mask_to_headlist(mask: torch.Tensor) -> torch.Tensor:
    ...


def mask_to_headlist(
        mask: np.ndarray | torch.Tensor
        ) -> np.ndarray | torch.Tensor:
    """Assumes dummy root with head -1.
    Supports batched input."""
    module = np if isinstance(mask, np.ndarray) else torch
    headlist = module.argmax(mask, -1)  # type: ignore
    headlist[..., 0] = -1
    return headlist


def uas_absolute(
        pred_headlist: np.ndarray,
        gold_headlist: np.ndarray):
    """Supports batched input."""
    return np.sum(pred_headlist == gold_headlist)


def uas(pred_headlist: np.ndarray,
        gold_headlist: np.ndarray):
    """Supports batched input."""
    return np.sum(pred_headlist == gold_headlist) / pred_headlist.size
