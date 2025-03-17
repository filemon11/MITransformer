import numpy as np
import torch
from ufal.chu_liu_edmonds import chu_liu_edmonds  # type: ignore
# from dsplot.graph import Graph  # type: ignore


def mst(score_matrix: np.ndarray) -> np.ndarray:
    """Expects probabilities."""
    # Convert to logspace (addition corresponds to
    # multiplication of probabilities)
    with np.errstate(divide='ignore'):
        score_matrix = np.log(score_matrix)
    heads, _ = chu_liu_edmonds(score_matrix.astype(np.double))
    return np.array(heads)


def merge_head_child_scores(
        head_scores: np.ndarray | torch.Tensor,
        child_scores: np.ndarray | torch.Tensor):
    """Throws away the diagonal; applies triangular mask.
    Supports batched input."""
    head_scores = np.tril(head_scores, -1)
    child_scores = np.tril(child_scores, -1)
    combined = head_scores + child_scores.swapaxes(-1, -2)
    return combined


def dummy_mask_removal(mask: np.ndarray):
    """Supports batched input."""
    return mask[..., 1:, 1:]


def mask_to_headlist(mask: np.ndarray):
    """Assumes dummy root with head -1.
    Supports batched input."""
    headlist = np.argmax(mask, axis=-1)
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
