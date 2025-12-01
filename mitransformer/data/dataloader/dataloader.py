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

from torch.utils.data import DataLoader as torchDataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np


from .. import dataset
from . import collator, sampler, batches

from typing import (Iterator,
                    TypeVar, Generic, overload)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)


B = TypeVar("B", bound=batches.BatchIdx, covariant=True)  # batch
S = TypeVar("S", bound=dataset.SentenceIdx, covariant=True)  # sentence


class DataLoader(torchDataLoader[S], Generic[S, B]):

    def __iter__(self) -> Iterator[B]:  # type: ignore
        return super().__iter__()  # type: ignore


@overload
def get_loader(
        ds: dataset.CoNLLUDataset,
        batch_size: int,
        bucket: bool = True,
        min_size: int = 5,
        max_size: int = 50,
        shuffle: bool = True,
        droplast: bool = True,
        rank: int | None = 0,
        world_size: int = 1,
        n_workers: int = 0,
        ) -> DataLoader[
            dataset.TokenisedMaskedSentence, batches.TokenisedMaskedBatch]:
    ...


@overload
def get_loader(
        ds: dataset.SentenceDataset,
        batch_size: int,
        bucket: bool = True,
        min_size: int = 5,
        max_size: int = 50,
        shuffle: bool = True,
        droplast: bool = True,
        rank: int | None = 0,
        world_size: int = 1,
        n_workers: int = 0,
        ) -> DataLoader[
            dataset.TokenisedSentence, batches.TokenisedBatch]:
    ...


@overload
def get_loader(
        ds: dataset.MemMapDepDataset,
        batch_size: int,
        bucket: bool = True,
        min_size: int = 5,
        max_size: int = 50,
        shuffle: bool = True,
        droplast: bool = True,
        rank: int | None = 0,
        world_size: int = 1,
        n_workers: int = 0,
        ) -> DataLoader[
            dataset.FastMaskedSentence, batches.FastMaskedBatch]:
    ...


@overload
def get_loader(
        ds: dataset.MemMapDataset,
        batch_size: int,
        bucket: bool = True,
        min_size: int = 5,
        max_size: int = 50,
        shuffle: bool = True,
        droplast: bool = True,
        rank: int | None = 0,
        world_size: int = 1,
        n_workers: int = 0,
        ) -> DataLoader[
            dataset.FastSentence, batches.FastBatch]:
    ...


def get_loader(
        ds: dataset.NLPDataset,
        batch_size: int,
        bucket: bool = True,
        min_size: int = 5,
        max_size: int = 50,
        shuffle: bool = True,
        droplast: bool = True,
        rank: int | None = 0,
        world_size: int = 1,
        n_workers: int = 0,
        ) -> (
            DataLoader[dataset.SentenceIdx, batches.BatchIdx]):

    # TODO: include attention mask to disregard masked tokens
    # in loss calculation
    assert ds.mapped is True

    if bucket:
        assert world_size == 1, (
            "Distributed sampling not implemented"
            "for bucketed sampling.")
        return DataLoader(
            ds,
            batch_sampler=sampler.BySequenceLengthSampler(
                ds,
                np.arange(min_size, max_size, 1),
                batch_size),
            collate_fn=collator.Collate(
                ds.keys_for_tensors
                ),
            pin_memory=True,
            persistent_workers=True if n_workers > 0 else False,
            num_workers=n_workers,)

    else:
        sampl: DistributedSampler | None
        if world_size == 1:
            sampl = None
        else:
            sampl = DistributedSampler(
                ds, num_replicas=world_size,
                rank=rank, shuffle=shuffle, drop_last=False)

        connect_with_dummy = False
        connect_with_self = False
        pad_mask_with: dict[str, bool] = {}
        # TODO: 
        if isinstance(ds, dataset.MaskedDataset):
            if isinstance(
                    ds.transform_mask,
                    dataset.TransformMaskHeadChild):
                connect_with_dummy = ds.transform_mask.connect_with_dummy
                connect_with_self = ds.transform_mask.connect_with_self
            pad_mask_with = ds.keys_for_mask_padding
        return DataLoader(
            ds,
            shuffle=False if sampl is not None else shuffle,
            batch_size=batch_size // world_size,
            drop_last=droplast,
            collate_fn=collator.PaddingCollate(
                ds.keys_for_tensors,
                ds.keys_for_padding,
                pad_mask_with,
                connect_with_dummy=connect_with_dummy,
                connect_with_self=connect_with_self),
            sampler=sampl,
            pin_memory=True,
            num_workers=n_workers,
            persistent_workers=True if n_workers > 0 else False)
