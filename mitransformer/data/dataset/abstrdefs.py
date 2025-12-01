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

from torch.utils.data import Dataset as TorchDataset
from conllu.models import TokenList

import numpy as np
import numpy.typing as npt
from . import sentence, transform, utils, functions

from abc import ABC, abstractmethod

from typing import (Iterable,
                    TypeVar, Callable, Mapping,
                    Generic,
                    Self, Iterator)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)

T = TypeVar("T", covariant=True)


class Dataset(TorchDataset, ABC, Generic[T]):

    @classmethod
    @abstractmethod
    def from_file(
            cls, file: str) -> Self:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx) -> T:
        ...

    @abstractmethod
    def __iter__(self, idx) -> Iterator[T]:
        ...


class NLPDataset(Dataset[T]):
    keys_for_tensors: set[str]
    mapped: bool

    @staticmethod
    def make_conlludict(
            tokenlists: Iterable[TokenList]) -> sentence.CoNLLUDict:
        d = utils.filldict(
            ("tokens", "heads", "space_after", "deprels"),
            (
                functions.get_tokens, functions.get_head_list,
                functions.get_space_after, functions.get_deprels),
            tokenlists)
        return sentence.CoNLLUDict(
            tokens=d["tokens"],   # type: ignore
            heads=d["heads"],     # type: ignore
            space_after=d["space_after"],  # type: ignore
            deprels=d["deprels"])     # type: ignore

    @classmethod
    @abstractmethod
    def from_file(
            cls, file: str,
            max_len: int | None = None,
            first_k: int | None = None) -> Self:
        ...


I = TypeVar("I", bound=sentence.SentenceIdx, covariant=True)


class IdxDataset(NLPDataset[I]):
    ...


J = TypeVar("J", bound=sentence.SentenceIds, covariant=True)


# Is this needed? We do not have separate tokenised datasets
# since a dataset can be tokenised.
class TokenisedDataset(NLPDataset[J]):
    keys_for_padding: dict[str, int]
    ...


K = TypeVar("K", bound=sentence.SentenceMask, covariant=True)


class MaskedDataset(NLPDataset[K]):
    keys_for_tensors: set[str]
    keys_for_padding: dict[str, int]
    keys_for_mask_padding: dict[str, bool]
    transform_mask: transform.TransformFunc | None

    @classmethod
    @abstractmethod
    def from_file(
            cls, file: str,
            max_len: int | None = None,
            first_k: int | None = None,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: utils.MasksSetting = "current") -> Self:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx) -> K:
        ...
