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

import numpy as np

from random import shuffle
from collections import defaultdict

from .. import dataset

from typing import (TypedDict,)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)


SentenceTokens = TypedDict("SentenceTokens", {"tokens": list[str]})


class BySequenceLengthSampler(Sampler):
    def __init__(
            self, data_source: dataset.NLPDataset[SentenceTokens],
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
