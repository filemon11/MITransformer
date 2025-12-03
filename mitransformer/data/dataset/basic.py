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

from .. import tokeniser
from . import sentence, transform, utils, functions, abstrdefs

from typing import (Sequence,
                    Callable, Mapping,
                    Self, cast)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)

ENCODING = "utf-8"

DUMMY_DEPREL = "dummy"
ROOT_DEPREL = "!root"
EOS_DEPREL = "eos"


class SentenceDataset(
        abstrdefs.IdxDataset[sentence.TokenisedSentence],
        abstrdefs.TokenisedDataset[sentence.TokenisedSentence]):
    def __init__(
            self, data: sentence.BasicDict):
        self.mapped: bool = False

        self.tokens: list[list[str]] = data["tokens"]
        self.space_after: list[npt.NDArray[np.bool_]] | None
        self.space_after = data.get("space_after", None)

        self.tokenised: list[npt.NDArray[np.uint32]] | None = None

        self.token_mapper: tokeniser.TokenMapper | None

        self.keys_for_tensors: set[str] = set()
        self.keys_for_padding: dict[str, int] = {}

    @classmethod
    def from_file(
            cls, file: str,
            max_len: int | None = 40,
            first_k: int | None = None) -> Self:

        tokenlists = functions.load_conllu(file, max_len, first_k)
        data_dict = cls.make_conlludict(tokenlists)

        return cls(data_dict)

    @classmethod
    def from_str(
            cls, conllu_str: str,
            max_len: int | None = 40) -> Self:

        tokenlists = functions.load_conllu_from_str(conllu_str, max_len)
        return cls.from_conllu(tokenlists)

    @classmethod
    def from_conllu(
            cls, tokenlists: Sequence[TokenList]) -> Self:

        data_dict = cls.make_conlludict(tokenlists)

        return cls(data_dict)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx) -> sentence.TokenisedSentence:
        # creates mask dynamically

        sen: list[str] = self.tokens[idx][:-1]
        label: list[str] = self.tokens[idx][1:]

        keys = dict(
            idx=np.array(idx),
            tokens=sen,
            labels=label)

        if self.space_after is not None:
            keys["space_after"] = self.space_after[idx]

        if self.tokenised is None:
            return sentence.TokenisedSentence(**keys)   # type: ignore
        else:
            return sentence.TokenisedSentence(
                **keys,    # type: ignore
                input_ids=self.tokenised[idx][:-1],
                label_ids=self.tokenised[idx][1:])

    def map_to_ids(self, token_mapper: tokeniser.TokenMapper) -> None:
        self.tokenised = [
            np.array(sen, dtype=np.uint32)
            for sen in token_mapper(self.tokens)]
        self.token_mapper = token_mapper

        self.keys_for_tensors = {"input_ids", "label_ids"}
        self.keys_for_padding = {
            "input_ids": token_mapper.pad_id,
            "label_ids": -100}

        self.mapped = True


class CoNLLUDataset(
        SentenceDataset,
        abstrdefs.MaskedDataset,
        ):
    def __init__(
            self, data: sentence.CoNLLUDict,
            transform_mask: transform.TransformFunc | None,
            masks_setting: utils.MasksSetting = "current"):
        super().__init__(data)
        self.deprels: list[list[str]] = data["deprels"]
        self.heads: list[npt.NDArray[np.uint8]] = data["heads"]

        self.transform_mask: transform.TransformFunc | None
        self.transform_mask = transform_mask

        # self.masks: dict[str, list[npt.NDArray[np.bool_]]] = dict()
        # if transform_mask is not None:
        #     for d in (transform_mask(head_list_to_adjacency_matrix(hl))
        #               for hl in data["heads"]):
        #         for k, v in d.items():
        #             self.masks[k] = self.masks.get(k, []) + [v]

        self.masks_setting: utils.MasksSetting
        self.masks_setting = masks_setting

        self.keys_for_mask_padding: dict[str, bool] = {}

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
            masks_setting: utils.MasksSetting = "current"):

        tokenlists = functions.load_conllu(file, max_len, first_k)
        data_dict = cls.make_conlludict(tokenlists)

        return cls(data_dict, transform_masks, masks_setting)

    @classmethod
    def from_str(
            cls, conllu_str: str,
            max_len: int | None = 40,
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: utils.MasksSetting = "current"):

        tokenlists = functions.load_conllu_from_str(conllu_str, max_len)
        return cls.from_conllu(tokenlists, transform_masks, masks_setting)

    @classmethod
    def from_conllu(
            cls, tokenlists: Sequence[TokenList],
            transform_masks: Callable[
                [npt.NDArray[np.bool_]],
                Mapping[
                    str,
                    npt.NDArray[np.bool_]]] | None = None,
            masks_setting: utils.MasksSetting = "current"):

        data_dict = cls.make_conlludict(tokenlists)

        return cls(data_dict, transform_masks, masks_setting)

    def __getitem__(self, idx) -> sentence.TokenisedMaskedSentence:
        # creates mask dynamically
        non_mask_item = super().__getitem__(idx)

        heads: npt.NDArray[np.uint8] = self.heads[idx]      # is not output
        masks: dict[str, npt.NDArray[np.bool_] | None] = dict()
        if self.transform_mask is not None:
            masks.update(
                self.transform_mask(
                    functions.head_list_to_adjacency_matrix(heads)).items())

        masks = functions.shift_masks(self.masks_setting, masks)

        mask_item = cast(sentence.TokenisedMaskedSentence, non_mask_item)
        mask_item["masks"] = masks
        return mask_item

    def map_to_ids(self, token_mapper: tokeniser.TokenMapper) -> None:
        super().map_to_ids(token_mapper)
        self.keys_for_tensors.add("masks")
        self.keys_for_mask_padding = {"masks": False}
