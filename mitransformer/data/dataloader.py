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
from torch.utils.data import Sampler
from torch.utils.data import DataLoader as torchDataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import numpy.typing as npt

from random import shuffle
from collections import defaultdict

from . import dataset

from abc import ABC, abstractmethod

from typing import (Iterator, TypedDict,
                    TypeVar, Any, NotRequired, Generic, Union, overload)

from ..utils.logmaker import getLogger

logger = getLogger(__name__)


class BySequenceLengthSampler(Sampler):
    def __init__(
            self, data_source: dataset.DepDataset,
            bucket_boundaries, batch_size=64,
            drop_last=True, include_smaller=False, include_larger=False):
        self.data_source = data_source
        ind_n_len = []
        for i, s in enumerate(data_source):     # type: ignore
            ind_n_len.append((i, len(s['tokens'])))

        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.drop_last = drop_last

        if self.drop_last:
            print(
                "WARNING: drop_last=True, dropping last non batch-size"
                "batch in every bucket ... ")

        boundaries = list(self.bucket_boundaries)
        if include_smaller:
            boundaries = [np.iinfo(np.int16).min] + boundaries
        if include_larger:
            boundaries = boundaries + [np.iinfo(np.int16).max]

        self.buckets_min = torch.tensor(boundaries[:-1])
        self.buckets_max = torch.tensor(boundaries[1:])
        self.boundaries = torch.tensor(self.bucket_boundaries)

    def shuffle_tensor(self, t):
        return t[torch.randperm(len(t))]

    def __iter__(self):
        data_buckets = defaultdict(list)
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)

            if pid is None:
                continue

            data_buckets[pid].append(p)

        tensored: dict[str, list | torch.Tensor] = dict(data_buckets)
        for k in tensored.keys():
            tensored[k] = torch.tensor(tensored[k])

        iter_list = []
        for k in tensored.keys():

            t = self.shuffle_tensor(tensored[k])
            batch = torch.split(t, self.batch_size, dim=0)

            if self.drop_last and len(batch[-1]) != self.batch_size:
                batch = batch[:-1]

            iter_list += batch

        shuffle(iter_list)
        # shuffle all the batches so they arent ordered by bucket

        # size
        for i in iter_list:
            yield i.numpy().tolist()    # as it was stored in an array

    def __len__(self):
        return len(self.data_source)

    def element_to_bucket_id(self, x, seq_length):

        valid_buckets: torch.Tensor
        valid_buckets = ((seq_length >= self.buckets_min)
                         * (seq_length < self.buckets_max))

        nonzero = valid_buckets.nonzero()
        if nonzero.shape[0] == 0:
            return None

        bucket_id = nonzero[0].item()

        return bucket_id


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


class BaseBatch(TypedDict):
    idx: torch.Tensor


class MaskedBatch(TypedDict):
    masks: dict[str, torch.BoolTensor]


class IDDictBatch(TypedDict):
    input_ids: torch.Tensor
    label_ids: torch.Tensor


class CoNLLUBatch(MaskedBatch):
    tokens: list[str]
    labels: list[str]
    space_after: NotRequired[list[npt.NDArray[np.bool_]]]


class CoNLLUTokenisedBatch(BaseBatch, CoNLLUBatch, IDDictBatch):
    pass


class EssentialBatch(BaseBatch, MaskedBatch, IDDictBatch):
    pass


Batch = TypeVar("Batch", bound=CoNLLUTokenisedBatch | EssentialBatch)
D = TypeVar("D", bound=dataset.DepDataset)


class DataLoader(torchDataLoader[Batch], Generic[Batch, D]):
    dataset: D

    def __iter__(self) -> Iterator[Batch]:  # type: ignore
        return super().__iter__()  # type: ignore


IDBatch = TypeVar(
    "IDBatch", bound=Union[
        CoNLLUTokenisedBatch,
        EssentialBatch],
    covariant=True)


@overload
def get_loader(
        ds: dataset.DepDataset[dataset.CoNLLUTokenisedSentence],
        batch_size: int,
        bucket: bool = True, min_size: int = 5,
        max_size: int = 50,
        shuffle: bool = True,
        droplast: bool = True,
        rank: int | None = 0,
        world_size: int = 1,
        n_workers: int = 0,
        ) -> DataLoader[CoNLLUTokenisedBatch, dataset.DepDataset]:
    ...


@overload
def get_loader(
        ds: dataset.DepDataset[
            dataset.EssentialSentence],
        batch_size: int,
        bucket: bool = True, min_size: int = 5,
        max_size: int = 50,
        shuffle: bool = True,
        droplast: bool = True,
        rank: int | None = 0,
        world_size: int = 1,
        n_workers: int = 0,
        ) -> DataLoader[EssentialBatch, dataset.DepDataset]:
    ...


def get_loader(
        ds: (
            dataset.DepDataset[dataset.CoNLLUTokenisedSentence]
            | dataset.DepDataset[dataset.EssentialSentence]),
        batch_size: int,
        bucket: bool = True, min_size: int = 5,
        max_size: int = 50,
        shuffle: bool = True,
        droplast: bool = True,
        rank: int | None = 0,
        world_size: int = 1,
        n_workers: int = 0,
        ) -> (
            DataLoader[CoNLLUTokenisedBatch, dataset.DepDataset]
            | DataLoader[EssentialBatch, dataset.DepDataset]):

    # TODO: include attention mask to disregard masked tokens
    # in loss calculation
    assert ds.mapped is True

    if bucket:
        assert world_size == 1, (
            "Distributed sampling not implemented"
            "for bucketed sampling.")
        return DataLoader(
            ds,
            batch_sampler=BySequenceLengthSampler(
                ds,
                np.arange(min_size, max_size, 1),
                batch_size),
            collate_fn=Collate(
                ds.keys_for_tensors
                ),
            pin_memory=True,
            persistent_workers=True if n_workers > 0 else False,
            num_workers=n_workers,)

    else:
        sampler: DistributedSampler | None
        if world_size == 1:
            sampler = None
        else:
            sampler = DistributedSampler(
                ds, num_replicas=world_size,
                rank=rank, shuffle=shuffle, drop_last=False)

        connect_with_dummy = False
        connect_with_self = False
        if isinstance(
                ds.transform_mask,
                dataset.TransformMaskHeadChild):
            connect_with_dummy = ds.transform_mask.connect_with_dummy
            connect_with_self = ds.transform_mask.connect_with_self
        return DataLoader(
            ds,
            shuffle=False if sampler is not None else shuffle,
            batch_size=batch_size // world_size,
            drop_last=droplast,
            collate_fn=PaddingCollate(
                ds.keys_for_tensors,
                ds.keys_for_padding,
                ds.keys_for_mask_padding,
                connect_with_dummy=connect_with_dummy,
                connect_with_self=connect_with_self),
            sampler=sampler,
            pin_memory=True,
            num_workers=n_workers,
            persistent_workers=True if n_workers > 0 else False)
