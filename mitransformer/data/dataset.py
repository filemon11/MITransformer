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

from torch.utils.data import Dataset
import conllu
import numpy as np
import numpy.typing as npt
from mmap_ninja import RaggedMmap   # type: ignore

from collections import defaultdict

from . import tokeniser

from abc import ABC, abstractmethod

from typing import (Iterable, Iterator, Sequence, TypedDict,
                    TypeVar, Callable, Hashable, Literal, Mapping,
                    Concatenate, NotRequired, Generic, Union, Any)

from ..utils.logmaker import getLogger

logger = getLogger(__name__)

ENCODING = "utf-8"

DUMMY_DEPREL = "dummy"
ROOT_DEPREL = "!root"
EOS_DEPREL = "eos"


TransformFunc = Callable[
    [npt.NDArray[np.bool_]],
    Mapping[str, npt.NDArray[np.bool_]]]

X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z", bound=Hashable)


def listmap(func: Callable[[X], Y], seq: Iterable[X]) -> list[Y]:
    return list(map(func, seq))


def filldict(
        keys: Sequence[Z],
        funcs: Sequence[Callable[[X], Y]],
        seq: Iterable[X]) -> dict[Z, list[Y]]:

    out_dict: dict[Z, list[Y]] = defaultdict(list)

    for entry in seq:
        for key, func in zip(keys, funcs):
            out_dict[key].append(func(entry))

    return out_dict


class MaskTransform(ABC):
    @abstractmethod
    def __call__(
            self, mask: npt.NDArray[np.bool_],
            ) -> dict[str, npt.NDArray[np.bool_]]:
        ...


class TransformMaskHeadChild(MaskTransform):
    def __init__(
            self,
            keys_for_head: set[str] = {"head"},
            keys_for_child: set[str] = {"child"},
            triangulate: int | None = 0,
            connect_with_dummy: bool = True,
            connect_with_self: bool = False,):
        assert not (connect_with_dummy and connect_with_self), (
            "You cannot represent non-existant arcs both with "
            "arcs to the dummy node and with self-arcs."
        )

        self.keys_for_head = keys_for_head
        self.keys_for_child = keys_for_child
        self.triangulate = triangulate
        self.connect_with_dummy = connect_with_dummy
        self.connect_with_self = connect_with_self

    def __call__(
            self,
            mask: npt.NDArray[np.bool_],
            ) -> dict[str, npt.NDArray[np.bool_]]:
        """Assumes a dummy token in the beginning. Therefore: one needs to
        add arcs for nodes that
        do not have a left head and nodes that do not have left children.

        ATTENTION: modifies the matrix inplace."""

        head = mask
        child = mask.T
        # head[range(len(head)), range(len(head))] = True
        # child[range(len(child)), range(len(child))] = True

        if self.connect_with_dummy:
            tril_head = np.tril(head, -1)
            set_true = ~tril_head.any(1)
            head[:, 0] = np.logical_or(set_true, head[:, 0])

            tril_child = np.tril(child, -1)
            set_true = ~tril_child.any(1)
            child[:, 0] = np.logical_or(set_true, child[:, 0])

        if self.connect_with_self:
            child = child.copy()
            tril_head = np.tril(head, -1)
            length = head.shape[0]
            set_true = ~tril_head.any(1)
            head[np.arange(0, length),
                 np.arange(0, length)] = np.logical_or(
                     set_true,
                     head.diagonal(
                         axis1=-1,
                         axis2=-2
                     ))
            tril_child = np.tril(child, -1)
            length = head.shape[0]
            set_true = ~tril_child.any(1)
            child[np.arange(0, length),
                  np.arange(0, length)] = np.logical_or(
                      set_true,
                      child.diagonal(
                          axis1=-1,
                          axis2=-2
                      ))

        if self.triangulate is not None:
            head = np.tril(head, self.triangulate)
            child = np.tril(child, self.triangulate)

        out_dict: dict[str, npt.NDArray[np.bool_]] = {}

        for key in self.keys_for_head:
            out_dict[key] = head

        for key in self.keys_for_child:
            out_dict[key] = child
        # print(child)

        return out_dict


class TransformMaskFull(MaskTransform):
    def __init__(
            self, keys_for_empty: set[str] = {"standard"}):
        self.keys_for_empty = keys_for_empty

    def __call__(
            self,
            mask: npt.NDArray[np.bool_],
            ) -> dict[str, npt.NDArray[np.bool_]]:
        """No arcs are masked"""

        # maybe this should produce all True matrices to easy collation
        trues = np.full(mask.shape, True)
        return {key: trues for key in self.keys_for_empty}


def transform_combined(
        mask: npt.NDArray[np.bool_],
        funcs_and_args: set[
            tuple[
                Callable[
                    Concatenate[npt.NDArray[np.bool_], ...],
                    Mapping[str, npt.NDArray[np.bool_]]],
                dict[str, npt.NDArray[np.bool_]]]]
        ) -> dict[str, npt.NDArray[np.bool_]]:

    out_dict: dict[str, npt.NDArray[np.bool_]] = {}

    for func, kwargs in funcs_and_args:
        out_dict.update(func(mask, **kwargs))

    return out_dict


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


Sen = TypeVar("Sen", bound=MaskedSentence, covariant=True)
IDSen = TypeVar("IDSen", bound=Union[
    CoNLLUTokenisedSentence,
    EssentialSentence],
                covariant=True)
# We treat the dictionaries as frozendicts.


class DepDataset(Dataset, ABC, Generic[Sen]):
    mapped: bool
    keys_for_tensors: set[str]
    keys_for_padding: dict[str, int]
    keys_for_mask_padding: dict[str, bool]
    transform_mask: TransformFunc | None

    @classmethod
    @abstractmethod
    def from_file(
            cls, file: str,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None,
            masks_setting: Literal['complete', 'current', 'next'],
            max_len: int | None,
            first_k: int | None):
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx) -> Sen:
        ...


class CoNLLUDataset(DepDataset[CoNLLUSentence | CoNLLUTokenisedSentence]):
    def __init__(
            self, data: CoNLLUDict,
            transform_mask: TransformFunc | None,
            masks_setting: Literal['complete', 'current', 'next'] = "current"):
        self.mapped: bool = False

        self.tokens: list[list[str]] = data["tokens"]
        self.deprels: list[list[str]] = data["deprels"]
        self.heads: list[npt.NDArray[np.uint8]] = data["heads"]
        self.space_after: list[npt.NDArray[np.bool_]] | None
        self.space_after = data.get("space_after", None)

        self.transform_mask: TransformFunc | None
        self.transform_mask = transform_mask

        # self.masks: dict[str, list[npt.NDArray[np.bool_]]] = dict()
        # if transform_mask is not None:
        #     for d in (transform_mask(head_list_to_adjacency_matrix(hl))
        #               for hl in data["heads"]):
        #         for k, v in d.items():
        #             self.masks[k] = self.masks.get(k, []) + [v]

        self.masks_setting: Literal['complete', 'current', 'next']
        self.masks_setting = masks_setting

        self.tokenised: list[npt.NDArray[np.uint32]] | None = None

        self.token_mapper: tokeniser.TokenMapper | None

        self.keys_for_tensors: set[str] = set()
        self.keys_for_padding: dict[str, int] = {}
        self.keys_for_mask_padding: dict[str, bool] = {}

    @staticmethod
    def make_conlludict(tokenlists: Iterable[conllu.TokenList]) -> CoNLLUDict:
        d = filldict(
            ("tokens", "heads", "space_after", "deprels"),
            (get_tokens, get_head_list, get_space_after, get_deprels),
            tokenlists)
        return CoNLLUDict(
            tokens=d["tokens"],   # type: ignore
            heads=d["heads"],     # type: ignore
            space_after=d["space_after"],  # type: ignore
            deprels=d["deprels"])     # type: ignore

    @classmethod
    def from_file(
            cls, file: str,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: Literal['complete', 'current', 'next'] = "current",
            max_len: int | None = 40,
            first_k: int | None = None):

        tokenlists = load_conllu(file, max_len, first_k)
        data_dict = cls.make_conlludict(tokenlists)

        return cls(data_dict, transform_masks, masks_setting)

    @classmethod
    def from_str(
            cls, conllu_str: str,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: Literal['complete', 'current', 'next'] = "current",
            max_len: int | None = 40):

        tokenlists = load_conllu_from_str(conllu_str, max_len)
        data_dict = cls.make_conlludict(tokenlists)

        return cls(data_dict, transform_masks, masks_setting)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx) -> CoNLLUSentence | CoNLLUTokenisedSentence:
        # creates mask dynamically

        sentence: list[str] = self.tokens[idx][:-1]
        label: list[str] = self.tokens[idx][1:]

        heads: npt.NDArray[np.uint8] = self.heads[idx]      # is not output
        masks: dict[str, npt.NDArray[np.bool_] | None] = dict()
        if self.transform_mask is not None:
            masks.update(
                self.transform_mask(head_list_to_adjacency_matrix(heads)))

        match self.masks_setting:
            case "current":
                masks = {
                    key: None if masks is None else masks[:-1, :-1]
                    for key, masks in masks.items()}

            case "next":
                masks = {
                    key: None if masks is None else masks[1:, :-1]
                    for key, masks in masks.items()}
        keys = dict(
            idx=np.array(idx),
            tokens=sentence,
            labels=label,
            masks=masks)

        if self.space_after is not None:
            keys["space_after"] = self.space_after[idx]

        if self.tokenised is None:
            return CoNLLUSentence(**keys)   # type: ignore
        else:
            return CoNLLUTokenisedSentence(
                **keys,    # type: ignore
                input_ids=self.tokenised[idx][:-1],
                label_ids=self.tokenised[idx][1:])

    def map_to_ids(self, token_mapper: tokeniser.TokenMapper) -> None:
        self.tokenised = [
            np.array(sentence, dtype=np.uint32)
            for sentence in token_mapper(self.tokens)]
        self.token_mapper = token_mapper

        self.keys_for_tensors = {"input_ids", "masks", "label_ids"}
        self.keys_for_padding = {
            "input_ids": token_mapper.pad_id,
            "label_ids": -100}
        self.keys_for_mask_padding = {"masks": False}

        self.mapped = True


class MemMapDataset(DepDataset[EssentialSentence]):
    def __init__(
            self,
            transform_mask: TransformFunc | None,
            file: str | None = None,
            masks_setting: Literal['complete', 'current', 'next'] = "current",
            id_hl: RaggedMmap | None = None,
            max_len: int | None = 40,
            first_k: int | None = None):
        self.mapped: bool = False

        self.file: str | None = file
        self.first_k: int | None = first_k

        self.transform_mask: TransformFunc | None
        self.transform_mask = transform_mask

        self.max_len: int | None = max_len

        self.masks_setting: Literal['complete', 'current', 'next']
        self.masks_setting = masks_setting

        self.id_hl: RaggedMmap | None = id_hl
        if id_hl is not None:
            self.mapped = True

        self.token_mapper: tokeniser.TokenMapper | None

        self.keys_for_tensors: set[str] = set()
        self.keys_for_padding: dict[str, int] = {}
        self.keys_for_mask_padding: dict[str, bool] = {}

    @property
    def sentences(self) -> Iterator[tuple[list[str],
                                    npt.NDArray[np.uint8]]]:
        assert self.file is not None
        return (self.get_sentence(tl) for tl in load_conllu(
            self.file,
            self.max_len,
            first_k=self.first_k))

    @property
    def tokens(self) -> Iterator[list[str]]:
        assert self.file is not None
        return (get_tokens(tl) for tl in load_conllu(
            self.file,
            self.max_len,
            first_k=self.first_k))

    @property
    def heads(self) -> Iterator[npt.NDArray[np.uint8]]:
        assert self.file is not None
        return (get_head_list(tl) for tl in load_conllu(
            self.file,
            self.max_len,
            first_k=self.first_k))

    @staticmethod
    def get_sentence(
            tokenlist: conllu.TokenList
            ) -> tuple[list[str], npt.NDArray[np.uint8]]:
        return get_tokens(tokenlist), get_head_list(tokenlist)

    @classmethod
    def from_file(
            cls, file: str,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: Literal['complete', 'current', 'next'] = "current",
            max_len: int | None = 40,
            first_k: int | None = None):

        return cls(
            transform_masks, file, masks_setting,
            max_len=max_len, first_k=first_k)

    @classmethod
    def from_memmap(
            cls, path: str,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: Literal['complete', 'current', 'next'] = "current",
            pad_id: int = 0,
            max_len: int | None = 40,
            first_k: int | None = None):

        id_hl = RaggedMmap(path)

        dataset = cls(
            transform_masks,
            masks_setting=masks_setting,
            id_hl=id_hl,
            max_len=max_len,
            first_k=first_k)
        dataset.keys_for_tensors = {"input_ids", "masks", "label_ids"}
        dataset.keys_for_padding = {"input_ids": pad_id,
                                    "label_ids": -100}
        dataset.keys_for_mask_padding = {"masks": False}

        return dataset

    def __len__(self) -> int:
        assert self.id_hl is not None
        if self.first_k is not None:
            return min(len(self.id_hl), self.first_k)
        else:
            return len(self.id_hl)

    def __getitem__(self, idx) -> EssentialSentence:
        # creates mask dynamically
        assert self.id_hl is not None
        ids, heads = self.id_hl[idx]
        masks: dict[str, npt.NDArray[np.bool_] | None] = dict()
        if self.transform_mask is not None:
            masks.update(
                self.transform_mask(head_list_to_adjacency_matrix(heads)))

        match self.masks_setting:
            case "current":
                masks = {
                    key: None if masks is None else masks[:-1, :-1]
                    for key, masks in masks.items()}

            case "next":
                masks = {
                    key: None if masks is None else masks[1:, :-1]
                    for key, masks in masks.items()}
        # print(masks)

        return EssentialSentence(
            idx=np.array(idx),
            masks=masks,
            input_ids=ids[:-1],
            label_ids=ids[1:])

    def map_to_ids(
            self, token_mapper: tokeniser.TokenMapper, memdir: str) -> None:
        assert self.file is not None
        id_head_list_generator = (np.stack((
            np.array(token_mapper([tokens])[0], dtype=np.uint32),
            headlist))
            for tokens, headlist in self.sentences)

        self.id_hl = RaggedMmap.from_generator(
            out_dir=memdir,
            sample_generator=id_head_list_generator,
            batch_size=1024,
            verbose=True
            )

        self.token_mapper = token_mapper

        self.keys_for_tensors = {"input_ids", "masks", "label_ids"}
        self.keys_for_padding = {
            "input_ids": token_mapper.pad_id,
            "label_ids": -100}
        self.keys_for_mask_padding = {"masks": False}

        self.mapped = True


class MemMapWindowDataset(MemMapDataset):
    def __init__(
            self,
            transform_mask: TransformFunc | None,
            file: str | None = None,
            masks_setting: Literal['complete', 'current', 'next'] = "current",
            memdir: str | None = None,
            max_len: int = 40,
            first_k: int | None = None):
        self.mapped: bool = False

        self.file: str | None = file
        self.first_k: int | None = first_k

        self.transform_mask: TransformFunc | None
        self.transform_mask = transform_mask

        self.max_len: int = max_len

        self.masks_setting: Literal['complete', 'current', 'next']
        self.masks_setting = masks_setting

        self.memdir: str | None = memdir
        if memdir is not None:
            self.mapped = True

        self.token_mapper: tokeniser.TokenMapper | None

        self.keys_for_tensors: set[str] = set()
        self.keys_for_padding: dict[str, int] = {}
        self.keys_for_mask_padding: dict[str, bool] = {}

        self.arr_len: int | None = None

    @classmethod
    def from_file(
            cls, file: str,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: Literal['complete', 'current', 'next'] = "current",
            max_len: int | None = 40,
            first_k: int | None = None):
        assert max_len is not None
        return cls(
            transform_masks, file, masks_setting,
            max_len=max_len, first_k=first_k)

    @classmethod
    def from_memmap(
            cls, path: str,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: Literal['complete', 'current', 'next'] = "current",
            pad_id: int = 0,
            max_len: int | None = 40,
            first_k: int | None = None):
        assert first_k is None, "first_k not implemented for MMWD."
        assert max_len is not None
        dataset = cls(
            transform_masks,
            masks_setting=masks_setting,
            memdir=path,
            max_len=max_len)
        dataset.keys_for_tensors = {"input_ids", "masks", "label_ids"}
        dataset.keys_for_padding = {"input_ids": pad_id,
                                    "label_ids": -100}
        dataset.keys_for_mask_padding = {"masks": False}

        return dataset

    def __len__(self) -> int:
        assert self.arr_len is not None
        return self.arr_len // self.max_len

    def __getitem__(self, idx) -> EssentialSentence:
        # creates mask dynamically
        assert self.mapped
        assert self.memdir is not None
        assert self.token_mapper is not None

        i = self.max_len * idx
        data = np.memmap(
            self.memdir,
            dtype=np.uint32,
            mode='r',
            shape=(self.arr_len, 2))   # type: ignore
        ids = np.concat(
            (
                np.array([
                    self.token_mapper.dummy_id, self.token_mapper.root_id]),
                data[i:i+self.max_len, 0]))
        heads = data[i:i+self.max_len, 1]
        # print(heads)
        heads = heads + np.arange(heads.shape[-1])+1
        heads[heads == np.arange(heads.shape[-1])+1] = 0
        # print(heads)
        heads[heads < 0] = -1  # make archs out of the window attend to dummy
        # print(heads)
        heads = heads + 1
        heads = np.concat((np.array([0, 0]), heads))
        # print(heads, self.token_mapper.decode([ids.tolist()])[0])

        masks: dict[str, npt.NDArray[np.bool_] | None] = dict()
        if self.transform_mask is not None:
            masks.update(
                self.transform_mask(
                    head_list_to_adjacency_matrix(
                        heads,
                        correct_underflow_overflow=True)))  # type: ignore

        match self.masks_setting:
            case "current":
                masks = {
                    key: None if masks is None else masks[:-1, :-1]
                    for key, masks in masks.items()}

            case "next":
                masks = {
                    key: None if masks is None else masks[1:, :-1]
                    for key, masks in masks.items()}

        return EssentialSentence(
            idx=np.array(idx),
            masks=masks,
            input_ids=ids[:-1],
            label_ids=ids[1:])

    def map_to_ids(
            self, token_mapper: tokeniser.TokenMapper, memdir: str) -> None:
        assert self.file is not None
        # iterates twice through sentences; TODO: improve
        arr_len = 0
        for tokens, _ in self.sentences:
            arr_len += len(tokens)
        self.arr_len = arr_len

        arr = np.memmap(memdir, dtype=np.uint32, mode='w+', shape=(arr_len, 2))

        idx = 0
        for tokens, headlist in self.sentences:
            headlist[headlist == 0] = np.arange(
                len(headlist))[headlist == 0] + 1
            r = np.arange(
                headlist.shape[-1], dtype=headlist.dtype)
            headlist = headlist - (r + 1)
            arr[idx:idx+len(tokens), 0] = np.array(
                token_mapper([tokens])[0],
                dtype=np.uint32)
            arr[idx:idx+len(tokens), 1] = headlist
            idx += len(tokens)
        arr.flush()

        self.memdir = memdir

        self.token_mapper = token_mapper

        self.keys_for_tensors = {"input_ids", "masks", "label_ids"}
        self.keys_for_padding = {
            "input_ids": token_mapper.pad_id,
            "label_ids": -100}
        self.keys_for_mask_padding = {"masks": False}

        self.mapped = True

    @staticmethod
    def get_sentence(
            tokenlist: conllu.TokenList
            ) -> tuple[list[str], npt.NDArray[np.uint8]]:
        return get_tokens(tokenlist, False), get_head_list(tokenlist, False)

    @property
    def sentences(self) -> Iterator[tuple[list[str],
                                    npt.NDArray[np.uint8]]]:
        assert self.file is not None
        return (self.get_sentence(tl) for tl in load_conllu(
            self.file,
            self.max_len//10,
            self.first_k))

    @property
    def tokens(self) -> Iterator[list[str]]:
        assert self.file is not None
        return (get_tokens(tl, False) for tl in load_conllu(
            self.file,
            self.max_len,
            self.first_k))

    @property
    def heads(self) -> Iterator[npt.NDArray[np.uint8]]:
        assert self.file is not None
        return (get_head_list(tl, False) for tl in load_conllu(
            self.file,
            self.max_len,
            self.first_k))


def load_conllu_from_str(
        conllu_str: str, max_len: int | None = 40
        ) -> list[conllu.TokenList]:
    return [tokenlist for tokenlist in conllu.parse(conllu_str)
            if max_len is None or len(tokenlist) <= max_len]


def get_tokens(
        tokenlist: conllu.TokenList,
        add_dummy_and_root: bool = True) -> list[str]:
    tokens = [tokeniser.DUMMY, tokeniser.ROOT] if add_dummy_and_root else []
    tokens.extend(token["form"] for token in tokenlist)
    tokens.append(tokeniser.EOS)
    return tokens


def get_head_list(
        tokenlist: conllu.TokenList,
        add_dummy_and_root: bool = True) -> npt.NDArray[np.uint8]:
    heads = [0, 0] if add_dummy_and_root else []
    heads.extend(
        token["head"]+(1 if add_dummy_and_root else 0) if token["head"]
        is not None else (1 if add_dummy_and_root else 0)
        for token in tokenlist)
    # 0 is already root; therefore add 1 because of dummy # +1
    heads.append(0+(1 if add_dummy_and_root else 0))  # (1)   # EOS token
    return np.asarray(heads, dtype=np.uint8)


def get_deprels(tokenlist: conllu.TokenList,
                add_dummy_and_root: bool = True) -> list[str]:
    tokens = [DUMMY_DEPREL, ROOT_DEPREL] if add_dummy_and_root else []
    tokens.extend(token["deprel"] for token in tokenlist)
    tokens.append(EOS_DEPREL)
    return tokens


def get_space_after(tokenlist: conllu.TokenList) -> npt.NDArray[np.bool_]:
    spaces: list[bool] = []

    def token_space_after(token) -> bool:
        if (isinstance(token["misc"], dict)
                and token["misc"].get("SpaceAfter", "Yes") == "No"):
            return False
        return True

    spaces.extend(
        token_space_after(token)
        for token in tokenlist)
    return np.array(spaces, dtype=np.bool_)


def head_list_to_adjacency_matrix(
        headlist: Sequence[int] | npt.NDArray[np.uint],
        correct_underflow_overflow: bool = False,
        ) -> npt.NDArray[np.bool_]:
    sen_len = len(headlist)
    headlist_arr = np.array(headlist)

    # print("before overunder:", headlist_arr)
    if correct_underflow_overflow:
        headlist_arr[np.logical_or(
            headlist_arr < 0, headlist_arr > sen_len-1)] = 0
    # print("overunder:", headlist_arr)

    adjacenceny_matrix = np.full((sen_len, sen_len), False, dtype=bool)
    adjacenceny_matrix[np.arange(sen_len), headlist_arr] = True
    return adjacenceny_matrix


def get_adjacency_matrix(tokenlist: conllu.TokenList) -> npt.NDArray[np.bool_]:
    return head_list_to_adjacency_matrix(get_head_list(tokenlist))


def apply_to_tokenlist(
        tokenlist: conllu.TokenList,
        funcs: tuple[Callable[[conllu.TokenList], Any], ...]
        ) -> tuple[Any, ...]:
    return tuple(fn(tokenlist) for fn in funcs)


def load_conllu(
        file: str, max_len: int | None = 40,
        first_k: int | None = None
        ) -> Iterator[conllu.TokenList]:
    data_file = open(file, "r", encoding=ENCODING)
    loaded_num = 0
    for tokenlist in conllu.parse_incr(data_file):
        if max_len is None or len(tokenlist) <= max_len:
            # Disregard contracted tokens
            yield conllu.TokenList(
                [token for token in tokenlist
                    if isinstance(token["id"], int)],
                metadata=tokenlist.metadata,
                default_fields=tokenlist.default_fields)
            loaded_num += 1
        if first_k is not None and loaded_num >= first_k:
            break
