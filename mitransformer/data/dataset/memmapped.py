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

from conllu.models import TokenList

import numpy as np
import numpy.typing as npt
from mmap_ninja import RaggedMmap   # type: ignore

from .. import tokeniser
from . import sentence, transform, utils, functions, abstrdefs

from typing import (Iterator, Callable, Mapping,
                    Self)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)

ENCODING = "utf-8"

DUMMY_DEPREL = "dummy"
ROOT_DEPREL = "!root"
EOS_DEPREL = "eos"


class MemMapDataset(
        abstrdefs.IdxDataset[sentence.FastSentence],
        abstrdefs.TokenisedDataset[sentence.FastSentence]):
    def __init__(
            self,
            file: str | None = None,
            id_hl: RaggedMmap | None = None,
            max_len: int | None = 40,
            first_k: int | None = None):
        self.mapped: bool = False

        self.file: str | None = file
        self.first_k: int | None = first_k

        self.max_len: int | None = max_len

        self.id_hl: RaggedMmap | None = id_hl
        if id_hl is not None:
            self.mapped = True

        self.token_mapper: tokeniser.TokenMapper | None

        self.keys_for_tensors: set[str] = set()
        self.keys_for_padding: dict[str, int] = {}

    @property
    def sentences(self) -> Iterator[tuple[list[str],
                                    npt.NDArray[np.uint8]]]:
        assert self.file is not None
        return (functions.get_sentence(tl) for tl in functions.load_conllu(
            self.file,
            self.max_len,
            first_k=self.first_k))

    @property
    def tokens(self) -> Iterator[list[str]]:
        assert self.file is not None
        return (functions.get_tokens(tl) for tl in functions.load_conllu(
            self.file,
            self.max_len,
            first_k=self.first_k))

    @classmethod
    def from_file(
            cls, file: str,
            max_len: int | None = 40,
            first_k: int | None = None,
            ) -> Self:

        return cls(
            file,
            max_len=max_len, first_k=first_k)

    @classmethod
    def from_memmap(
            cls, path: str,
            pad_id: int = 0,
            max_len: int | None = 40,
            first_k: int | None = None) -> Self:

        id_hl = RaggedMmap(path)

        dataset = cls(
            id_hl=id_hl,
            max_len=max_len,
            first_k=first_k)
        dataset.keys_for_tensors = {"input_ids", "masks", "label_ids"}
        dataset.keys_for_padding = {"input_ids": pad_id,
                                    "label_ids": -100}

        return dataset

    def __len__(self) -> int:
        assert self.id_hl is not None
        if self.first_k is not None:
            return min(len(self.id_hl), self.first_k)
        else:
            return len(self.id_hl)

    def __getitem__(self, idx: int) -> sentence.FastSentence:
        # creates mask dynamically
        assert self.id_hl is not None
        ids: np.ndarray
        ids = self.id_hl[idx]  # type: ignore

        return sentence.FastSentence(
            idx=np.array(idx),
            input_ids=ids[:-1],
            label_ids=ids[1:])

    def map_to_ids(
            self, token_mapper: tokeniser.TokenMapper, memdir: str) -> None:
        assert self.file is not None
        id_head_list_generator = (
            np.array(token_mapper([tokens])[0], dtype=np.uint32)
            for tokens, _ in self.sentences)

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


class MemMapDepDataset(
        MemMapDataset,
        abstrdefs.MaskedDataset):
    def __init__(
            self,
            file: str | None = None,
            id_hl: RaggedMmap | None = None,
            max_len: int | None = 40,
            first_k: int | None = None,
            transform_mask: transform.TransformFunc | None = None,
            masks_setting: utils.MasksSetting = "current"):
        super().__init__(
            file, id_hl=id_hl, max_len=max_len,
            first_k=first_k)
        self.transform_mask: transform.TransformFunc | None
        self.transform_mask = transform_mask

        self.masks_setting: utils.MasksSetting
        self.masks_setting = masks_setting

        self.keys_for_mask_padding: dict[str, bool] = {}

    @property
    def heads(self) -> Iterator[npt.NDArray[np.uint8]]:
        assert self.file is not None
        return (functions.get_head_list(tl) for tl in functions.load_conllu(
            self.file,
            self.max_len,
            first_k=self.first_k))

    @classmethod
    def from_file(
            cls, file: str,
            max_len: int | None = 40,
            first_k: int | None = None,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: utils.MasksSetting = "current") -> Self:

        return cls(
            file=file,
            max_len=max_len, first_k=first_k,
            transform_mask=transform_masks,
            masks_setting=masks_setting)

    @classmethod
    def from_memmap(
            cls, path: str,
            pad_id: int = 0,
            max_len: int | None = 40,
            first_k: int | None = None,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: utils.MasksSetting = "current") -> Self:

        id_hl = RaggedMmap(path)

        dataset = cls(
            id_hl=id_hl,
            max_len=max_len,
            first_k=first_k,
            transform_mask=transform_masks,
            masks_setting=masks_setting,)
        dataset.keys_for_tensors = {"input_ids", "masks", "label_ids"}
        dataset.keys_for_padding = {"input_ids": pad_id,
                                    "label_ids": -100}
        dataset.keys_for_mask_padding = {"masks": False}

        return dataset

    def __getitem__(self, idx) -> sentence.FastMaskedSentence:
        # creates mask dynamically
        assert self.id_hl is not None
        ids: np.ndarray
        ids, heads = self.id_hl[idx]  # type: ignore
        masks: dict[str, npt.NDArray[np.bool_] | None] = dict()
        if self.transform_mask is not None:
            masks.update(
                self.transform_mask(
                    functions.head_list_to_adjacency_matrix(heads)))

        masks = functions.shift_masks(self.masks_setting, masks)

        return sentence.FastMaskedSentence(
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


class MemMapWindowDataset(MemMapDepDataset):
    def __init__(
            self,
            transform_mask: transform.TransformFunc | None,
            file: str | None = None,
            masks_setting: utils.MasksSetting = "current",
            memdir: str | None = None,
            max_len: int = 40,
            first_k: int | None = None):
        self.mapped: bool = False

        self.file: str | None = file
        self.first_k: int | None = first_k

        self.transform_mask: transform.TransformFunc | None
        self.transform_mask = transform_mask

        self.max_len = max_len

        self.masks_setting: utils.MasksSetting
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
            max_len: int | None = 40,
            first_k: int | None = None,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: utils.MasksSetting = "current",):
        assert max_len is not None
        return cls(
            transform_masks, file, masks_setting,
            max_len=max_len, first_k=first_k)

    @classmethod
    def from_memmap(
            cls, path: str,
            pad_id: int = 0,
            max_len: int | None = 40,
            first_k: int | None = None,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: utils.MasksSetting = "current"):
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
        assert self.max_len is not None
        return self.arr_len // self.max_len

    def __getitem__(self, idx) -> sentence.FastMaskedSentence:
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
                    functions.head_list_to_adjacency_matrix(
                        heads,
                        correct_underflow_overflow=True)))  # type: ignore

        masks = functions.shift_masks(self.masks_setting, masks)

        return sentence.FastMaskedSentence(
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
            tokenlist: TokenList
            ) -> tuple[list[str], npt.NDArray[np.uint8]]:
        return functions.get_tokens(
            tokenlist, False), functions.get_head_list(tokenlist, False)

    @property
    def sentences(self) -> Iterator[tuple[list[str],
                                    npt.NDArray[np.uint8]]]:
        assert self.file is not None
        assert self.max_len is not None
        return (self.get_sentence(tl) for tl in functions.load_conllu(
            self.file,
            self.max_len//10,
            self.first_k))

    @property
    def tokens(self) -> Iterator[list[str]]:
        assert self.file is not None
        return (
            functions.get_tokens(tl, False) for tl in functions.load_conllu(
                self.file,
                self.max_len,
                self.first_k))

    @property
    def heads(self) -> Iterator[npt.NDArray[np.uint8]]:
        assert self.file is not None
        return (
            functions.get_head_list(tl, False) for tl in functions.load_conllu(
                self.file,
                self.max_len,
                self.first_k))
