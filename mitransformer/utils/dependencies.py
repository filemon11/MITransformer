import numpy as np
from ufal.chu_liu_edmonds import chu_liu_edmonds  # type: ignore
# from dsplot.graph import Graph  # type: ignore

from typing import Sequence


def mst(score_matrix: np.ndarray) -> np.ndarray:
    """Expects probabilities."""
    # Convert to logspace (addition corresponds to
    # multiplication of probabilities)
    with np.errstate(divide='ignore'):
        score_matrix = np.log(score_matrix)
    heads, _ = chu_liu_edmonds(score_matrix.astype(np.double))
    return np.array(heads)


def plot_tree(
        output_path: str,
        headlist: Sequence[int],
        labels: Sequence[str] | Sequence[int] | None = None,
        ) -> None:
    if labels is None:
        str_labels = [str(i) for i in range(len(headlist))]
    else:
        assert len(headlist) == len(labels), ("Labels and headlist "
                                              "are not of equal lengths!")
        str_labels = [f"{label}_{num}" for num, label in enumerate(labels)]

    directed_dict: dict[str, list[str]]
    directed_dict = {i: [] for i in str_labels}

    for child, head in enumerate(headlist):
        if head != -1:
            directed_dict[str_labels[head]].append(str_labels[child])

    graph = Graph(directed_dict, directed=True)  # type: ignore
    graph.plot(output_path)


def merge_head_child_scores(
        head_scores: np.ndarray,
        child_scores: np.ndarray):
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


def uas_absolute(pred_headlist: np.ndarray,
                 gold_headlist: np.ndarray):
    """Supports batched input."""
    return np.sum(pred_headlist == gold_headlist)


def uas(pred_headlist: np.ndarray,
        gold_headlist: np.ndarray):
    """Supports batched input."""
    return np.sum(pred_headlist == gold_headlist) / pred_headlist.size
