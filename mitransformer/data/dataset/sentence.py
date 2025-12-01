import numpy as np
import numpy.typing as npt
from typing import (TypedDict, NotRequired)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)


class BasicDict(TypedDict):
    tokens: list[list[str]]
    space_after: NotRequired[list[npt.NDArray[np.bool_]]]


class CoNLLUDict(BasicDict):
    heads: list[npt.NDArray[np.uint8]]  # max 127 sequence length
    deprels: list[list[str]]


# for tokenised sentence
class SentenceIdx(TypedDict):
    idx: npt.NDArray[np.int_]


# for sentences containing masks
class SentenceMask(TypedDict):
    masks: dict[str, npt.NDArray[np.bool_] | None]


# for memmaped sentences
class SentenceIds(TypedDict):
    input_ids: NotRequired[npt.NDArray[np.uint32]]
    label_ids: NotRequired[npt.NDArray[np.uint32]]


class BasicSentence(SentenceIdx):
    tokens: list[str]
    labels: list[str]
    space_after: NotRequired[list[npt.NDArray[np.bool_]]]


class BasicMaskedSentence(BasicSentence, SentenceMask):
    pass


class TokenisedSentence(SentenceIds, BasicSentence):
    pass


class TokenisedMaskedSentence(TokenisedSentence, SentenceMask):
    pass


class FastSentence(SentenceIdx, SentenceIds):
    pass


class FastMaskedSentence(FastSentence, SentenceMask):
    pass
