import torch
import torch.nn.functional as F

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


def lm_loss(
        logits: torch.Tensor, labels: torch.Tensor,
        ignore_index: int = -100,
        reduction: Literal["sum", "mean", "none"] = "mean",
        discriminative: bool = False) -> torch.Tensor:

    logits = torch.swapaxes(logits, 1, 2)
    if discriminative:
        probs = F.sigmoid(logits)
        mask = labels != ignore_index
        labels_without_negative = labels
        labels_without_negative[~mask] = 0  # to ignore later
        one_hot = F.one_hot(
            labels_without_negative,
            logits.shape[-2])
        loss = F.binary_cross_entropy(
            probs.swapaxes(-1, -2), one_hot.float(), reduction='none')

        # # mask randomly
        # mask_rate = 0.5
        # rand_mask = torch.rand(
        #    loss.shape, device=mask.device) < mask_rate
        # rand_mask[one_hot] = 0  # unmask gold items
        # loss[rand_mask] = 0

        # false continuation factor
        factor = 0.5
        tensor_factor = torch.full(loss.shape, factor, device=mask.device)
        tensor_factor[one_hot.bool()] = 1  # multiply gold items by 1
        loss *= tensor_factor

        # ignore padding tokens
        loss[~mask] = 0

        loss = loss.sum() / mask.sum()

    else:
        loss = F.cross_entropy(
            logits, labels,
            ignore_index=ignore_index,
            reduction=reduction)
    return loss


def attention_entropy_loss(
        arc_distributions: torch.Tensor,
        to_ignore_mask: torch.Tensor | Literal["triangular"] | None,
        reduction: Literal["sum", "mean", "none"] = "mean",
        input_ids: torch.Tensor | None = None,
        ignore_index: int = -100,
        global_distr: bool = True,
        include_current: bool = True,
        length_weighted: bool = False,
        prefix_dummies: int = 2
        ) -> torch.Tensor:
    """input shape [B, H, S, S] with
    H: number of heads,
    B: batch size,
    S: sequence length.

    output shape
    [B, S] if reduction = 'none'
    else scalar"""

    probs = arc_distributions

    if to_ignore_mask is not None and to_ignore_mask != "triangular":
        to_ignore_mask = to_ignore_mask.sum(0).to(torch.bool)  # type: ignore

    if global_distr:
        probs = get_head_averaged_distribution(probs)
        # [B, H, S, S] -> [B, S, S]

    entropy = get_attention_entropy(
        probs, to_ignore_mask, reduction="none",
        include_current=include_current, length_weighted=length_weighted,
        prefix_dummies=prefix_dummies)  # -> [B, S] or [B, H, S]

    if not global_distr:
        entropy = entropy.mean(1)  # [B, H, S] -> [B, S]

    if input_ids is not None:
        entropy[input_ids == ignore_index] = 0

    reduced = reduce(entropy, reduction)
    return reduced


def distance_loss(
        probs: torch.Tensor,
        to_ignore_mask: torch.Tensor | Literal["triangular"] | None,
        reduction: Literal["sum", "mean", "none"] = "mean",
        input_ids: torch.Tensor | None = None,
        ignore_index: int = -100,
        global_distr: bool = True,
        include_current: bool = True,
        prefix_dummies: int = 2,
        ) -> torch.Tensor:
    """input shape [B, H, S, S] with
    H: number of heads,
    B: batch size,
    S: sequence length.

    output shape
    [B, S] if reduction = 'none'
    else scalar"""

    probs = normalise(probs, not include_current, prefix_dummies > 0)
    if to_ignore_mask is not None:
        if to_ignore_mask == "triangular":
            probs = torch.tril(probs)
        else:
            to_ignore_mask = to_ignore_mask.sum(  # type: ignore
                0).to(torch.bool)
            probs[to_ignore_mask] = 0

    if global_distr:
        probs = get_head_averaged_distribution(probs)
        # [B, H, S, S] -> [B, S, S]

    s = probs.shape[-1]
    r = torch.arange(1, s+1-prefix_dummies, device=probs.device)

    if prefix_dummies > 0:
        prefix = torch.zeros(prefix_dummies, device=probs.device)
        r = torch.concat(
            (prefix, r))

    dist_mat = -1 * (r.repeat(s, 1) - r.reshape(-1, 1))     # + 1
    dist_mat = torch.tril(dist_mat)      # dist_mat.log())
    dist_mat[..., :prefix_dummies] = 0

    dist_mat = dist_mat.unsqueeze(0)  # -> [B, S, S] or [B, H, S, S]

    distances = dist_mat*probs
    cost = (distances).sum(-1)  # -> [B, S] or [B, H, S]

    if not global_distr:
        cost = cost.mean(1)  # [B, H, S] -> [B, S]

    if input_ids is not None:
        cost[input_ids == ignore_index] = 0

    return reduce(cost, reduction)


def get_head_averaged_distribution(
        probs: torch.Tensor
        ) -> torch.Tensor:
    """input: [B, H, S, S]"""
    probs = probs.mean(1)  # [B, S, S]
    return probs


def normalise_without_diagonal(
        probs: torch.Tensor) -> torch.Tensor:
    probs = torch.tril(probs, diagonal=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-4)
    return torch.tril(probs, diagonal=-1)


def normalise(
        probs: torch.Tensor,
        without_diagonal: bool = False,
        without_root_dummy: bool = False) -> torch.Tensor:
    if not without_diagonal and not without_root_dummy:
        return probs

    if without_diagonal:
        probs = torch.tril(probs, diagonal=-1)
    if without_root_dummy:
        probs[..., :2] = 0
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-4)

    if without_diagonal:
        probs = torch.tril(probs, diagonal=-1)
    if without_root_dummy:
        probs[..., :2] = 0
    return probs


def get_attention_entropy(
        probs: torch.Tensor,
        to_ignore: torch.Tensor | Literal["triangular"] | None = None,
        reduction: Literal["sum", "mean", "none"] = "mean",
        include_current: bool = True,
        length_weighted: bool = False,
        prefix_dummies: int = 2) -> torch.Tensor:
    """input shape [..., S, S]
    with S: sequence length.
    output shape: scalar if reduction is 'mean' or 'sum', else [..., S].

    Assumes normalised distribution."""

    probs = normalise(probs, not include_current, prefix_dummies > 0)

    logprobs = torch.log2(probs.clamp(min=1e-4))
    # attention: this was originally log_e
    entropy = -(probs*logprobs)
    del probs
    del logprobs

    if to_ignore is not None:
        if to_ignore == "triangular":
            entropy = entropy.masked_fill(
                torch.tril(
                    torch.ones(
                        *entropy.shape,
                        device=entropy.device)) == 0, 0)
        else:
            entropy[to_ignore] = 0  # type: ignore

    if length_weighted:
        start_at = 1 if include_current else 0
        norm_vector = torch.arange(
            start_at, entropy.shape[-2]+start_at-prefix_dummies,
            dtype=torch.float,
            device=entropy.device)

        if prefix_dummies > 0:
            prefix = torch.zeros(prefix_dummies, device=entropy.device)
            norm_vector = torch.concat(
                (prefix, norm_vector)).unsqueeze(0).t()
        # [S, S] ([[1], [2], [3], ...])
        norm_vector = torch.log2(norm_vector).clamp(min=1e-4)

        entropy = torch.div(entropy, norm_vector)

        # prevent numerical problem for one-item
        # distribution
        entropy = torch.div(entropy, norm_vector)
        entropy[..., (1-start_at)+prefix_dummies, prefix_dummies] = 1

        del norm_vector

    entropy = entropy.sum(-1)

    # TODO: ignore padding tokens

    reduced = reduce(entropy, reduction)
    return reduced
