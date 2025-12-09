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
import numpy as np

from collections import defaultdict
from abc import ABC, abstractmethod

from typing import (Any, )

from ...utils.logmaker import getLogger

logger = getLogger(__name__)


class CollateBase(ABC):
    @abstractmethod
    def __call__(
            self,
            list_of_sentences: list[dict[str, object]]
            ) -> dict[str, Any]:
        ...


class Collate(CollateBase):
    def __init__(
            self,
            keys_to_torch: set[str] = set()
            ):
        """Does not support masks of different types for
        (e.g. None and array) for individual sentences"""
        self.keys_to_torch = keys_to_torch

    def __call__(
            self,
            list_of_sentences: list[dict[str, object]]
            ) -> dict[str, Any]:
        output: dict[str, dict[str, list] | list] = defaultdict(list)
        for sentence in list_of_sentences:
            for key, content in sentence.items():
                if isinstance(content, dict):
                    if isinstance(output[key], list):
                        output[key] = dict()
                    for subkey, subcontent in content.items():
                        assert isinstance(output[key], dict)
                        output[key][subkey] = output[key].get(  # type: ignore
                            subkey,
                            [])
                        output[key][subkey] += [subcontent]

                else:
                    output[key].append(content)     # type: ignore

        def dict_to_torch(
                dictionary: dict[str, Any],
                keys_to_torch: set[str]) -> None:
            for key in keys_to_torch:
                if isinstance(dictionary[key], np.ndarray):
                    dictionary[key] = torch.from_numpy(
                        dictionary[key].astype(np.int64))

                elif isinstance(dictionary[key], dict):
                    dict_to_torch(dictionary[key], set(dictionary[key].keys()))

                elif dictionary[key] is None:
                    continue

                else:
                    if dictionary[key][0] is None:
                        continue

                    if isinstance(dictionary[key][0], np.ndarray):
                        print([i.shape for i in dictionary[key]])
                        dictionary[key] = torch.from_numpy(
                            np.stack(dictionary[key]).astype(np.int64))

                    else:
                        dictionary[key] = torch.from_numpy(
                            np.array(dictionary[key]).astype(np.int64))

        output_dict: dict[str, Any] = dict(output)
        dict_to_torch(output_dict, self.keys_to_torch)
        return output_dict


class PaddingCollate(Collate):
    def __init__(
            self,
            keys_to_torch: set[str] = set(),
            pad_with: dict[str, int] = dict(),
            pad_mask_with: dict[str, bool] = dict(),
            connect_with_dummy: bool = True,
            connect_with_self: bool = False
            ):
        super().__init__(keys_to_torch)
        self.pad_with = pad_with
        self.pad_mask_with = pad_mask_with
        self.connect_with_dummy = connect_with_dummy
        self.connect_with_self = connect_with_self
        assert not connect_with_self, "not implemented"

    def __call__(
            self,
            list_of_sentences: list[dict[str, object]]
            ) -> dict[str, Any]:

        max_lens: defaultdict[str, int] = defaultdict(int)
        for sentence in list_of_sentences:
            for key in self.pad_with.keys():
                max_lens[key] = max(max_lens[key],
                                    len(sentence[key]))  # type: ignore
            for key in self.pad_mask_with.keys():
                max_lens[key] = max(
                    max_lens[key],
                    len(next(iter(sentence[key].values()))))  # type: ignore
        # one mask field should not contain masks of different lengths

        new_sentence_list: list[dict[str, object]] = []
        for sentence in list_of_sentences:
            new_sentence = dict(sentence)
            for key, pad in self.pad_with.items():
                max_len = max_lens[key]
                if isinstance(new_sentence[key], list):
                    new_sentence[key] = (
                        new_sentence[key]
                        + [pad]
                        * (max_len - len(new_sentence[key])))  # type: ignore
                elif isinstance(new_sentence[key], np.ndarray):
                    new_sentence[key] = np.pad(
                        new_sentence[key].astype(np.int64),  # type: ignore
                        (0, max_len - len(new_sentence[key])),  # type: ignore
                        constant_values=pad)
                else:
                    raise Exception("Unknown type. Given:",
                                    type(new_sentence[key]))
                print(new_sentence[key].shape)
            for key, b in self.pad_mask_with.items():
                for mask_k, mask in new_sentence[key].items():  # type: ignore
                    new_mask = np.full((max_lens[key], max_lens[key]), b)
                    if self.connect_with_dummy:
                        # prevent rows without a single true value
                        new_mask[:, 0] = not b
                    elif self.connect_with_self:
                        new_mask[
                            np.arange(new_mask.shape[0]),
                            np.arange(new_mask.shape[0])] = not b
                    new_mask[:mask.shape[0], :mask.shape[1]] = mask
                    new_sentence[key][mask_k] = new_mask  # type: ignore

            new_sentence_list.append(new_sentence)
        return super().__call__(new_sentence_list)
