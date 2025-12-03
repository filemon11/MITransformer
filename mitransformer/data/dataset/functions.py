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

import conllu
from conllu.models import TokenList

import numpy as np
import numpy.typing as npt

from .. import tokeniser
from . import utils

from typing import (Iterator, Sequence,
                    Callable, Mapping,
                    Any)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)

ENCODING = "utf-8"

DUMMY_DEPREL = "dummy"
ROOT_DEPREL = "!root"
EOS_DEPREL = "eos"


def load_conllu_from_str(
        conllu_str: str, max_len: int | None = 40
        ) -> list[TokenList]:
    return [tokenlist for tokenlist in conllu.parse(conllu_str)
            if max_len is None or len(tokenlist) <= max_len]


def get_sentence(
        tokenlist: TokenList
        ) -> tuple[list[str], npt.NDArray[np.uint8]]:
    return get_tokens(tokenlist), get_head_list(tokenlist)


def get_tokens(
        tokenlist: TokenList,
        add_dummy_and_root: bool = True,
        add_eos: bool = True) -> list[str]:
    tokens = [tokeniser.DUMMY, tokeniser.ROOT] if add_dummy_and_root else []
    tokens.extend(token["form"] for token in tokenlist)
    if add_eos:
        tokens.append(tokeniser.EOS)
    return tokens


def get_head_list(
        tokenlist: TokenList,
        add_dummy_and_root: bool = True,
        add_eos: bool = True) -> npt.NDArray[np.uint8]:
    heads = [0, 0] if add_dummy_and_root else []
    heads.extend(
        token["head"]+(1 if add_dummy_and_root else 0) if token["head"]
        is not None else (1 if add_dummy_and_root else 0)
        for token in tokenlist)
    # 0 is already root; therefore add 1 because of dummy # +1
    if add_eos:
        heads.append(0+(1 if add_dummy_and_root else 0))  # (1)   # EOS token
    return np.asarray(heads, dtype=np.uint8)


def get_deprels(tokenlist: TokenList,
                add_dummy_and_root: bool = True,
                add_eos: bool = True) -> list[str]:
    tokens = [DUMMY_DEPREL, ROOT_DEPREL] if add_dummy_and_root else []
    tokens.extend(token["deprel"] for token in tokenlist)
    if add_eos:
        tokens.append(EOS_DEPREL)
    return tokens


def get_space_after(tokenlist: TokenList) -> npt.NDArray[np.bool_]:
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
        headlist: (
            Sequence[int] | npt.NDArray[np._IntegerT] | npt.NDArray[np.uint]),
        correct_underflow_overflow: bool = False,
        ) -> npt.NDArray[np.bool_]:
    sen_len = len(headlist)
    headlist_arr = np.array(headlist)

    headlist_arr[1:][
        np.equal(headlist_arr[1:], np.arange(0, len(headlist_arr))[1:])] = 1

    # print("before overunder:", headlist_arr)
    if correct_underflow_overflow:
        headlist_arr[np.logical_or(
            headlist_arr < 0, headlist_arr > sen_len-1)] = 0
    # print("overunder:", headlist_arr)

    adjacenceny_matrix = np.full((sen_len, sen_len), False, dtype=bool)
    adjacenceny_matrix[np.arange(sen_len), headlist_arr] = True

    return adjacenceny_matrix


def get_adjacency_matrix(tokenlist: TokenList) -> npt.NDArray[np.bool_]:
    return head_list_to_adjacency_matrix(get_head_list(tokenlist))


def apply_to_tokenlist(
        tokenlist: TokenList,
        funcs: tuple[Callable[[TokenList], Any], ...]
        ) -> tuple[Any, ...]:
    return tuple(fn(tokenlist) for fn in funcs)


def load_conllu(
        file: str, max_len: int | None = 40,
        first_k: int | None = None
        ) -> Iterator[TokenList]:
    data_file = open(file, "r", encoding=ENCODING)
    loaded_num = 0
    for tokenlist in conllu.parse_incr(data_file):
        if max_len is None or len(tokenlist) <= max_len:
            # Disregard contracted tokens
            yield TokenList(
                [token for token in tokenlist
                    if isinstance(token["id"], int)],
                metadata=tokenlist.metadata,
                default_fields=tokenlist.default_fields)
            loaded_num += 1
        if first_k is not None and loaded_num >= first_k:
            break


def shift_masks(
        masks_setting: utils.MasksSetting,
        masks: Mapping[str, npt.NDArray[np.bool_] | None]
        ) -> dict[str, npt.NDArray[np.bool_] | None]:
    mode_to_slice = {}
    if masks_setting != "next" and masks_setting != "complete":
        mode_to_slice["current"] = (slice(None, -1), slice(None, -1))
    if masks_setting != "current" and masks_setting != "complete":
        mode_to_slice["next"] = (slice(1, None), slice(None, -1))

    if len(mode_to_slice) > 0:
        masks = {
                f"{key}_{m}": None if masks is None else masks[*s]
                for key, masks in masks.items()
                for m, s in mode_to_slice.items()}
    else:
        masks = dict(masks)
    return masks
