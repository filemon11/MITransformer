from ..utils.dependencies import (
    mst, merge_head_child_scores,
    dummy_mask_removal, mask_to_headlist, uas_absolute)

import torch
import numpy as np

from typing import (Literal, )

from ..utils.logmaker import getLogger

logger = getLogger(__name__)


def select_true(preds: torch.Tensor,
                labels: torch.Tensor,
                ignore_index: int | None = None) -> torch.Tensor:
    """If ignore_index is given, it selects the probability for element zero"""
    labels = labels.unsqueeze(-1)
    if ignore_index is not None:
        labels = labels.clone()
        labels[labels == ignore_index] = 0
    return torch.gather(preds, -1, labels).squeeze(-1)


def logits_to_probs(logits: torch.Tensor, softmax: bool = True
                    ) -> torch.Tensor:
    if softmax:
        return torch.softmax(logits, dim=-1)
    else:
        return torch.sigmoid(logits)


def logits_to_true_probs(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int | None = None,
        softmax: bool = True) -> torch.Tensor:
    probs = logits_to_probs(logits, softmax)
    return select_true(probs, labels, ignore_index)


def logits_to_surprisal(logits: torch.Tensor,
                        labels: torch.Tensor,
                        ignore_index: int | None = None,
                        softmax: bool = True) -> torch.Tensor:
    return -torch.log(logits_to_true_probs(
        logits, labels, ignore_index, softmax))


def sum_depadded(
        values: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int) -> torch.Tensor:
    values[labels == ignore_index] = 0
    return values.sum(-1)


def mean_depadded(
        values: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int) -> torch.Tensor:
    num_items = (labels != ignore_index).sum(-1)
    sums = sum_depadded(values, labels, ignore_index)
    return sums / num_items


def logits_to_perplexity(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
        softmax: bool = True) -> torch.Tensor:
    # Should we disregard first node (root from dummy?)
    surprisal = logits_to_surprisal(logits, labels, softmax)
    means = mean_depadded(surprisal, labels, ignore_index)
    return torch.exp(means)


def unpad(
        sentences: torch.Tensor, labels: torch.Tensor,
        ignore_index: int) -> list[torch.Tensor]:
    unpadded_list: list[torch.Tensor] = []
    for sentence, sen_labels in zip(sentences, labels):
        unpadded_list.append(sentence[sen_labels != ignore_index])
    return unpadded_list


def unpad_masks(masks: torch.Tensor, labels: torch.Tensor,
                ignore_index: int) -> list[torch.Tensor]:
    unpadded_list: list[torch.Tensor] = []
    for sentence, sen_labels in zip(masks, labels):
        unpadded_list.append(
            sentence[..., sen_labels != ignore_index, :][
                ..., :, sen_labels != ignore_index])
    return unpadded_list


def get_uas_abs(inp: tuple[np.ndarray, np.ndarray, np.ndarray]) -> int:
    pred_arcs, gold_arcs, upto_not_padding = inp
    pred_arcs = pred_arcs[upto_not_padding][:, upto_not_padding]
    gold_arcs = gold_arcs[upto_not_padding][:, upto_not_padding]
    pred_headlist = mst(pred_arcs)
    # one could max for each row as head instead
    # but this would not correspond to the max probability
    # tree given the scores
    gold_headlist = mask_to_headlist(gold_arcs)

    uas_s = uas_absolute(pred_headlist, gold_headlist) + 1
    # add 1 for dummy mask
    return uas_s


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return -torch.log((1-x)/x)


def check_perform_opt(gradient_acc: int | None, i_train: int) -> bool:
    return (gradient_acc is None
            or (i_train+1) % gradient_acc == 0)


def uas_composition(
        score_preds: dict[str, torch.Tensor],
        score_golds: dict[str, torch.BoolTensor],
        label_ids: torch.Tensor,
        ignore_index: int,
        masks_setting: Literal["current", "next"] = "current",
        gov_key: str = "head", dep_key: str = "child") -> int:
    values = torch.stack((
        score_preds[gov_key], score_golds[gov_key].float(),
        score_preds[dep_key], score_golds[dep_key].float()))
    values_np = values.mean(1).detach().cpu().numpy()

    if masks_setting == "next":
        zeros = np.zeros((
            values_np.shape[0],
            values_np.shape[1],
            1, values_np.shape[3]))

        values_np = np.concatenate(
                (zeros, values_np[:, :, :-1]), axis=2)

    preds_arcs = dummy_mask_removal(
        merge_head_child_scores(values_np[0], values_np[2]))
    golds_arcs = dummy_mask_removal(
        merge_head_child_scores(values_np[1], values_np[3]))

    not_padding = (
        label_ids
        != ignore_index).cpu().numpy()[:, 1:]

    return sum(map(get_uas_abs, zip(
        preds_arcs, golds_arcs, not_padding)))
