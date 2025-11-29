import numpy as np
import numpy.typing as npt
from typing import (TypedDict, NotRequired)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)


class CoNLLUDict(TypedDict):
    tokens: list[list[str]]
    heads: list[npt.NDArray[np.uint8]]  # max 127 sequence length
    space_after: NotRequired[list[npt.NDArray[np.bool_]]]
    deprels: list[list[str]]


class IdxSentence(TypedDict):
    idx: npt.NDArray[np.int_]


class MaskedSentence(TypedDict):
    masks: dict[str, npt.NDArray[np.bool_] | None]


class IDDict(TypedDict):
    input_ids: npt.NDArray[np.uint32]
    label_ids: npt.NDArray[np.uint32]


class CoNLLUSentence(MaskedSentence):
    tokens: list[str]
    labels: list[str]
    space_after: NotRequired[list[npt.NDArray[np.bool_]]]


class CoNLLUTokenisedSentence(IdxSentence, CoNLLUSentence, IDDict):
    pass


class EssentialSentence(IdxSentence, MaskedSentence, IDDict):
    pass
