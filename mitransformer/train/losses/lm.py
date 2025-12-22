
import torch
import torch.nn.functional as F

from typing import Literal


@torch.compile
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
        k = 99
        # TODO make this an argument

        # mask ignored labels once
        valid = labels != ignore_index

        # sample negatives
        neg = torch.randint(
            0, logits.shape[1] - 1, (
                logits.shape[0], k, *labels.shape[1:]),
            device=logits.device
        )

        # create pos indices
        pos = torch.where(
            valid, labels, torch.zeros_like(labels)).unsqueeze(1)

        # shift negatives to avoid positive class
        neg = neg + (neg >= pos).long()

        # build indices: [positive | negatives]
        idx = torch.cat([pos, neg], dim=1)

        # gather
        logits = logits.gather(1, idx)

        # targets: positive is always index 0
        labels = torch.zeros_like(labels)
        labels[~valid] = ignore_index

        loss = F.cross_entropy(
            logits, labels,
            ignore_index=ignore_index,
            reduction=reduction)
    return loss
