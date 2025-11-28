import torch
import torch.nn.functional as F

from typing import Literal


def arc_loss(
        score_preds: torch.Tensor,
        score_gold: torch.BoolTensor,
        to_ignore_mask: torch.BoolTensor | None,
        reduction: Literal["sum", "mean"] = "mean",
        arc_loss_weighted: bool = False
        ) -> tuple[torch.Tensor, int]:
    """reduction sum takes a mean across dim 1
    of the mask"""
    # TODO: what if we have more than two masks?
    masks_dim = 0
    batch_dim = 1
    seq1_dim = 2
    seq2_dim = 3
    shape = score_preds.shape
    M = shape[masks_dim]
    B = shape[batch_dim]
    S = shape[seq1_dim]
    # Calculation if unpadded (no ignore mask)
    # assumes that all sentences have same length
    total_len = B * S
    factor = int((S + 1) / 2 * M)
    num_scores = total_len * factor

    if to_ignore_mask is not None:
        # assumes lens are the same for each M
        lens = (~to_ignore_mask).select(
            seq2_dim, 0).select(masks_dim, 0).float().sum(seq1_dim-1)
        # TODO: make it possible to give lens as parameter
        # since we compute them already in normal loss calculation
        # and compute the to_ignore_mask here using broadcasting...
        total_len = int(lens.sum().item())
        # divide through M since each head mask contributes 1x total_len
        num_scores = (
            torch.dot((lens+1), lens) * M / 2).item()  # type: ignore
        # divide score/factor by number of tokens to get average
        # per-token loss
        # score_gold = cast(torch.BoolTensor,
        #                  score_gold[~to_ignore_mask])
        # score_preds = score_preds[~to_ignore_mask]
    # print(score_preds.detach().cpu().numpy().round(2),
    #       score_gold.to(score_preds.dtype).detach().cpu().numpy())
    loss = F.binary_cross_entropy(
        score_preds,
        score_gold.to(score_preds.dtype),
        reduction='none')

    if arc_loss_weighted:
        true_el = torch.sum(score_gold)
        false_el = num_scores - true_el
        f_true = 0.5*num_scores/true_el
        f_false = 0.5*num_scores/false_el
        weights = torch.zeros(
            *score_preds.shape,
            device=score_preds.device)
        weights[score_gold] = f_true
        weights[~score_gold] = f_false
        loss *= weights

    if to_ignore_mask is not None:
        loss = (~to_ignore_mask)*loss
    else:
        loss = torch.tril(loss)

    loss = torch.sum(loss)

    if reduction == "mean":
        loss /= num_scores
    else:
        pass  # loss /= num_scores/total_len  # = factor
        # Each position can be attended to S+1 times

    return loss, int(num_scores)
