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
from torch.utils.data.dataset import Dataset
from torch.utils.data import BatchSampler
from torch.utils.data.distributed import (
    DistributedSampler as BaseDistributedSampler,
    _T_co)
import torch.distributed as dist

import numpy as np
import math
import bisect

from .. import dataset

from typing import (Iterator, Optional)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)


class BySequenceLengthSampler(BatchSampler):
    def __init__(
            self, data_source: dataset.TokenisedDataset[dataset.SentenceIds],
            min_size: int, max_size: int, batch_size=64,
            drop_last=True, fill_incomplete: bool = True,
            include_smaller=False, include_larger=False,
            seed: int = 0):

        self.epoch = 0
        self.seed = seed
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.fill_incomplete: bool = fill_incomplete

        boundaries: list[int] = list(range(min_size, max_size + 2))
        # includes max_size items

        if include_smaller:
            boundaries = [0] + boundaries
        if include_larger:
            boundaries = boundaries + [np.iinfo(np.int16).max]

        self.boundaries = boundaries
        self.num_buckets = len(boundaries) - 1

        self.buckets: list[list[int]] = [[] for _ in range(self.num_buckets)]

        for idx, sample in enumerate(iter(data_source)):
            assert "input_ids" in sample
            seq_len = len(sample["input_ids"])

            bucket_id = bisect.bisect_right(boundaries, seq_len) - 1
            if 0 <= bucket_id < self.num_buckets:
                self.buckets[bucket_id].append(idx)

        self._num_batches = 0
        for bucket in self.buckets:
            n = len(bucket)
            if n == 0:
                continue
            if self.drop_last:
                self._num_batches += n // self.batch_size
            else:
                self._num_batches += math.ceil(n / self.batch_size)

    def shuffle_tensor(self, t: torch.Tensor) -> torch.Tensor:
        return t[torch.randperm(len(t))]

    @torch.compile()
    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        batches: list[list[int]] = []

        for bucket in self.buckets:
            if len(bucket) == 0:
                continue

            # Shuffle indices inside bucket
            perm = torch.randperm(len(bucket), generator=g)
            shuffled = [bucket[i] for i in perm.tolist()]

            # Split into batches
            for i in range(0, len(shuffled), self.batch_size):
                batch = shuffled[i:i + self.batch_size]
                if len(batch) < self.batch_size:
                    if self.drop_last:
                        continue
                    elif self.fill_incomplete:
                        # add extra samples to make it evenly divisible
                        padding_size = self.batch_size - len(batch)
                        pad_idx = torch.randint(
                            0, len(shuffled), (padding_size,), generator=g)
                        batch += [shuffled[i] for i in pad_idx.tolist()]
                batches.append(batch)

        # Shuffle batches across buckets
        perm = torch.randperm(len(batches), generator=g)
        for i in perm.tolist():
            yield batches[i]

    def __len__(self):
        return self._num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedBySequenceLengthSampler(BatchSampler):
    def __init__(
            self, data_source: dataset.TokenisedDataset[dataset.SentenceIds],
            min_size: int, max_size: int, batch_size=64,
            drop_last=True, fill_incomplete: bool = True,
            include_smaller=False, include_larger=False,
            seed: int = 0, num_replicas: int | None = None,
            rank: int | None = None):

        # ---- distributed setup ----
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.batch_size = batch_size
        self.total_batch_size = batch_size * num_replicas
        self.drop_last = drop_last
        self.fill_incomplete: bool = fill_incomplete

        # ---- bucket boundaries ----

        boundaries: list[int] = list(range(min_size, max_size + 2))
        # includes max_size items

        if include_smaller:
            boundaries = [0] + boundaries
        if include_larger:
            boundaries = boundaries + [np.iinfo(np.int16).max]

        self.boundaries = boundaries
        self.num_buckets = len(boundaries) - 1

        self.buckets: list[list[int]] = [[] for _ in range(self.num_buckets)]

        for idx, sample in enumerate(iter(data_source)):
            assert "input_ids" in sample
            seq_len = len(sample["input_ids"])

            bucket_id = bisect.bisect_right(boundaries, seq_len) - 1
            if 0 <= bucket_id < self.num_buckets:
                self.buckets[bucket_id].append(idx)

    def shuffle_tensor(self, t: torch.Tensor) -> torch.Tensor:
        return t[torch.randperm(len(t))]

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        batches: list[list[int]] = []

        for bucket in self.buckets:
            n = len(bucket)
            if n == 0:
                continue

            perm = torch.randperm(len(bucket), generator=g)
            shuffled = [bucket[i] for i in perm.tolist()]

            for i in range(0, n, self.total_batch_size):
                batch = shuffled[i:i + self.total_batch_size]

                if len(batch) < self.total_batch_size:
                    if self.drop_last:
                        continue
                    elif self.fill_incomplete:
                        # add extra samples to make it evenly divisible
                        padding_size = self.total_batch_size - len(batch)
                        pad_idx = torch.randint(
                            0, len(shuffled), (padding_size,), generator=g)
                        batch += [shuffled[i] for i in pad_idx.tolist()]

                batches.append(batch)

        # shuffle batches globally
        perm = torch.randperm(len(batches), generator=g)
        shuffled = [batches[i] for i in perm.tolist()]

        # shard by rank
        for batch in shuffled:
            yield batch[self.rank: self.total_batch_size: self.num_replicas]

    def __len__(self):
        total = 0
        for bucket in self.buckets:
            n = len(bucket)
            if n == 0:
                continue
            if self.drop_last:
                total += n // self.total_batch_size
            else:
                total += math.ceil(n / self.total_batch_size)
        return total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedSampler(BaseDistributedSampler):
    """Is the standard torch implementation with the
    addition of a ``fill_incomplete`` argument.
    If false and ``drop_last`` is also false,
    returns unequal number of elements for the workers
    if not cleanly divisible."""
    def __init__(
            self,
            dataset: Dataset,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            seed: int = 0,
            drop_last: bool = False,
            fill_incomplete: bool = True,
            ) -> None:
        super().__init__(
            dataset, num_replicas, rank, shuffle, seed,
            drop_last
        )
        self.fill_incomplete = fill_incomplete

    def __iter__(self) -> Iterator[_T_co]:  # type: ignore
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(
                len(self.dataset),  # type: ignore[arg-type]
                generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            if self.fill_incomplete:
                # add extra samples to make it evenly divisible
                padding_size = self.total_size - len(indices)
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (
                        indices * math.ceil(padding_size / len(indices)))[
                        :padding_size
                    ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]

        if not self.drop_last and not self.fill_incomplete:
            assert len(indices) == len(
                    self.dataset), (  # type: ignore[arg-type]
                f"Faulty length in sampler: {len(indices)} but expected "
                f"{len(self.dataset)}.")  # type: ignore[arg-type]
        # subsample
        indices = indices[self.rank:: self.num_replicas]

        return iter(indices)  # type: ignore
