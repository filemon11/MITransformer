# TODO: decide on tokenisation
"""For each sentence, a special root token is added. The sentence
head receives this
root token as its head. The root token receives itself as its head.
Furthermore, a dummy token is prepended for cases where a node has no left
head (or no left children).
Structure: DUMMY ROOT t1 t2 ... EOS

The headlist is to be converted into an adjacency matrix at the dataloading
stage.
Adding the dummy arcs is also performed at the dataloading stage.

The masks use boolean arrays/tensors.
"""

import torch

from typing import (TypedDict, NotRequired)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)


class BatchIdx(TypedDict):
    idx: torch.Tensor


# for sentences containing masks
class BatchMask(TypedDict):
    masks: dict[str, torch.BoolTensor]


# for memmaped sentences
class BatchIds(TypedDict):
    input_ids: torch.Tensor
    label_ids: torch.Tensor


# for memmaped sentences
class BatchMaskIds(BatchIds, BatchMask):
    pass


class IdBatch(BatchIds, BatchIdx):
    pass


class MaskIdBatch(BatchMaskIds, BatchIdx):
    pass


class BasicBatch(BatchIdx):
    tokens: list[str]
    labels: list[str]
    space_after: NotRequired[torch.BoolTensor]


class BasicMaskedBatch(BasicBatch, BatchMask):
    pass


class TokenisedBatch(BatchIds, BasicBatch):
    pass


class TokenisedMaskedBatch(TokenisedBatch, BatchMask):
    pass


class FastBatch(BatchIdx, BatchIds):
    pass


class FastMaskedBatch(FastBatch, BatchMask):
    pass
