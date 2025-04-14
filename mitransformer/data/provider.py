
import os
from pathlib import Path

from . import dataset, tokeniser
from .. import utils

from dataclasses import dataclass
from typing import (TypedDict, Literal, Any,
                    NotRequired, overload,
                    TypeVarTuple, Self)

from ..utils.logmaker import getLogger, info

logger = getLogger(__name__)

ENCODING = "utf-8"

DUMMY_DEPREL = "dummy"
ROOT_DEPREL = "!root"
EOS_DEPREL = "eos"

SplitSelection = (
    tuple[
        Literal["train"],
        Literal["eval"],
        Literal["test"]]
    | tuple[Literal["train"], Literal["eval"]]
    | tuple[Literal["test"]])

TupleSelection = tuple[str, str, str] | tuple[str, str] | tuple[str]


# TODO: should switch to Python 3.13 and add ReadOnly to attributes
class DatasetDetails(TypedDict):
    dirs: NotRequired[TupleSelection]
    memmap_dir: str
    memmaped: NotRequired[SplitSelection]
    tokmap_dir: str
    tokmap_is_trained: NotRequired[bool]


class DatasetDetailsFull(DatasetDetails):
    dirs: NotRequired[tuple[str, str, str]]  # type: ignore
    memmaped: NotRequired[tuple[Literal["train"],  # type: ignore
                                Literal["eval"],
                                Literal["test"]]]


class DatasetDict(TypedDict):
    token_mapper: tokeniser.TokenMapper
    train: NotRequired[dataset.MemMapDataset]
    eval: NotRequired[dataset.MemMapDataset]
    test: NotRequired[dataset.MemMapDataset]


T = TypeVarTuple("T")


def make_name(
        dir: str, naming_pattern: str,
        splits: tuple[*T]) -> tuple[*T]:
    return tuple(
        os.path.join(
            dir, naming_pattern.format(split))
        for split in splits)  # type: ignore


ud = "./Universal Dependencies 2.14/ud-treebanks-v2.14"
EWT_dir = os.path.join(ud, "UD_English-EWT")
EWT_name = "en_ewt-ud-{}.conllu"


EWT = DatasetDetailsFull(
    dirs=make_name(
        EWT_dir, EWT_name,
        ("train", "dev", "test")),
    memmap_dir="./processed/EWT/memmap",
    tokmap_dir="./processed/EWT/"
    )

Wikitext_raw = DatasetDetailsFull(
    dirs=make_name(
        "./Wikitext_raw", "wikitext_spacy_{}.conllu",
        ("train", "dev", "test")),
    memmap_dir="./processed/Wikitext_raw/memmap",
    tokmap_dir="./processed/Wikitext_raw/"
    )

Wikitext_raw_memmapped = DatasetDetailsFull(
    memmaped=("train", "eval", "test"),
    memmap_dir="./processed/Wikitext_raw/memmap",
    tokmap_dir="./processed/Wikitext_raw/"
    )

Wikitext_processed = DatasetDetailsFull(
    dirs=make_name(
        "./Wikitext_processed", "wikitext_spacy_{}.conllu",
        ("train", "dev", "test")),
    memmap_dir="./processed/Wikitext_processed/memmap",
    tokmap_dir="./processed/Wikitext_processed/"
    )

Wikitext_processed_memmapped = DatasetDetailsFull(
    memmaped=("train", "eval", "test"),
    memmap_dir="./processed/Wikitext_processed/memmap",
    tokmap_dir="./processed/Wikitext_processed/"
    )

Sample = DatasetDetailsFull(
    dirs=tuple(3*["./sample.conllu"]),  # type: ignore
    memmap_dir="./processed/Sample/memmap",
    tokmap_dir="./processed/Sample/"
    )

Sample_memmapped = DatasetDetails(
    memmaped=("train", "eval"),
    memmap_dir="./processed/Sample/memmap",
    tokmap_dir="./processed/Sample/"
    )

NaturalStoriesOld = DatasetDetails(
    dirs=("./naturalstories-master/parses/ud/stories-aligned.conllx",),
    memmap_dir="./processed/NaturalStories/memmap",
    tokmap_dir="./processed/NaturalStories/"
    )

dataset_details_full = dict(
    EWT=EWT,
    Wikitext_raw=Wikitext_raw,
    Wikitext_processed=Wikitext_processed,
    Sample=Sample
    )

dataset_details_full_memmaped = dict(
    Wikitext_raw=Wikitext_raw_memmapped,
    Wikitext_processed=Wikitext_processed_memmapped,
    # Sample=Sample_memmapped
    )

dataset_details = (
    dataset_details_full
    | {"NaturalStoriesOld": NaturalStoriesOld})


def mmd_splits(memmap_dir: str, splits: SplitSelection) -> TupleSelection:
    return tuple(
        os.path.join(memmap_dir, split)
        for split in splits)  # type: ignore


@overload
def load_dataset(
        details: DatasetDetailsFull,
        max_len_train: int | None = 40,
        max_len_eval_test: int | None = None,
        vocab_size: int | None = 50_000,
        first_k: int | None = None,
        first_k_eval_test: int | None = None,
        triangulate: int | None = 0,
        connect_with_dummy: bool = True,
        connect_with_self: bool = False,
        masks_setting: dataset.MasksSetting = "current",
        *args, **kwargs
        ) -> DatasetDict:
    ...


@overload
def load_dataset(
        details: DatasetDetails,
        max_len_train: int | None = 40,
        max_len_eval_test: int | None = None,
        vocab_size: int | None = 50_000,
        first_k: int | None = None,
        first_k_eval_test: int | None = None,
        triangulate: int | None = 0,
        connect_with_dummy: bool = True,
        connect_with_self: bool = False,
        masks_setting: dataset.MasksSetting = "current",
        *args, **kwargs
        ) -> DatasetDict:
    ...


def load_dataset(
        details: DatasetDetails | DatasetDetailsFull,       # type: ignore
        max_len_train: int | None = 40,  # should mark all
        max_len_eval_test: int | None = None,
        vocab_size: int | None = 50_000,  # of this with ReadOnly
        first_k: int | None = None,
        first_k_eval_test: int | None = None,
        triangulate: int | None = 0,
        connect_with_dummy: bool = True,
        connect_with_self: bool = False,
        masks_setting: dataset.MasksSetting = "current",
        *args, **kwargs) -> DatasetDict:
    # TODO: implement option to read raw string data and parse,
    # to accept raw string as input and to save parsed data
    transform = dataset.TransformMaskHeadChild(
        keys_for_head={"head"},
        keys_for_child={"child"},
        triangulate=triangulate,
        connect_with_dummy=connect_with_dummy,
        connect_with_self=connect_with_self)

    memmap_dir = details["memmap_dir"]

    load_memmap = False
    if "memmaped" in details:
        load_memmap = True
        assert "dirs" not in details, (
            "Cannot specify existing memmap and raw dataset"
            "at the same time")
        dirs = mmd_splits(memmap_dir, details["memmaped"])
    else:
        assert "dirs" in details
        dirs = details["dirs"]

    train_token_mapper = True
    if load_memmap or (
            "tokmap_is_trained" in details
            and details["tokmap_is_trained"]):
        train_token_mapper = False

    assert not (load_memmap and train_token_mapper), (
        "Cannot train token mapper with a memmap.")

    def load_dataset(
            dir: str, is_train: bool,
            max_len: int | None = None) -> dataset.MemMapDataset:
        first_k_param = first_k if is_train else first_k_eval_test
        if load_memmap:
            return dataset.MemMapDataset.from_memmap(
                dir, transform,
                max_len=max_len,
                first_k=first_k_param,
                masks_setting=masks_setting)
        else:
            return dataset.MemMapDataset.from_file(
                dir, transform,
                max_len=max_len,
                first_k=first_k_param,
                masks_setting=masks_setting)

    train: None | dataset.MemMapDataset = None
    eval: None | dataset.MemMapDataset = None
    test: None | dataset.MemMapDataset = None
    sets: tuple[dataset.MemMapDataset, ...] = tuple()
    splits: tuple[str, ...] = tuple()
    if len(dirs) == 1:
        # Only test
        test = load_dataset(dirs[0], True, max_len_eval_test)
        sets = (test,)
        splits = ("test",)
    elif len(dirs) > 1 and len(dirs) < 4:
        # train, eval and optional test
        train = load_dataset(dirs[0], True, max_len_train)
        eval = load_dataset(dirs[1], False, max_len_eval_test)
        sets = (train, eval)
        splits = ("train", "eval")
        if len(dirs) == 3:
            test = load_dataset(dirs[2], False, max_len_eval_test)
            sets = (train, eval, test)
            splits = ("train", "eval", "test")
    else:
        raise Exception("Too many or not enough dataset dirs provided.")

    tokmap_dir = details["tokmap_dir"]
    if train_token_mapper:
        assert train is not None, (
            "Must train token_mapper on train set. "
            "Please provide a train set.")
        assert vocab_size is not None, (
            "Vocabulary size must be specified. Given: None.")

        token_mapper = tokeniser.TokenMapper.train(
            train.tokens,
            keep_top_k=vocab_size)

        Path(tokmap_dir).mkdir(parents=True, exist_ok=True)
        token_mapper.save(os.path.join(tokmap_dir, "mapper"))

        info(
            None, logger,
            f"Trained tokenmapper with vocab size {token_mapper.vocab_size}.")

    else:
        token_mapper = tokeniser.TokenMapper.load(
            os.path.join(tokmap_dir, "mapper"))

    for split, split_name in zip(sets, splits):
        if split is not None and not split.mapped:
            Path(memmap_dir).mkdir(parents=True, exist_ok=True)
            split.map_to_ids(
                token_mapper,
                os.path.join(memmap_dir, split_name))

    if train is not None:
        assert eval is not None
        if test is not None:
            return DatasetDict(
                token_mapper=token_mapper,
                train=train,
                eval=eval,
                test=test
                )
        else:
            return DatasetDict(
                token_mapper=token_mapper,
                train=train,
                eval=eval,
                )
    else:
        assert test is not None
        return DatasetDict(
            token_mapper=token_mapper,
            test=test
            )


@dataclass
class DataConfig(utils.Params):
    dataset_name: str
    include_test: bool
    memmapped: bool
    max_len_train: int | None = 40
    max_len_eval_test: int | None = None
    vocab_size: int | None = 50_000
    first_k: int | None = None
    first_k_eval_test: int | None = None
    triangulate: int | None = 0
    connect_with_dummy: bool = True
    connect_with_self: bool = False
    masks_setting: dataset.MasksSetting = "current"


class DataProvider():
    def __init__(self, config: DataConfig, rank: int | None = None):
        self.config = config

        if config.memmapped:
            details = dataset_details_full_memmaped[config.dataset_name]
            if not config.include_test:
                details["memmaped"] = details["memmaped"][0:2]  # type: ignore
        else:
            details = dataset_details_full[config.dataset_name]
            if not config.include_test:
                details["dirs"] = details["dirs"][0:2]  # type: ignore

        self.details = details
        self.datasets = load_dataset(details, **self.config.to_dict())

        info(
            rank, logger,
            "Initialised Data Provider with params:\n")
        info(rank, logger, config.info)
        info(rank, logger, (
            "Loaded datasets with "
            + ', '.join([
                str(len(ds)) for ds  # type: ignore
                in self.datasets.values() if isinstance(ds, dataset.Dataset)])
            + " sentences."))

    @classmethod
    def load(cls, file: str, rank: int | None = None,
             **optional_config: Any) -> Self:
        '''The `load` class method in Python reads
        and deserializes `DataProvider` object from a file using pickle.

        Parameters
        ----------
        file : str
            The `filename` parameter is a string that represents the
            name of the file from which data will be loaded.

        Returns
        -------
        Self
            An instance of the `DataProvider` class.
        '''
        return cls(DataConfig.load(file, **optional_config), rank)

    def save(self, file: str) -> None:
        '''The `save` function saves the object
        to a file using pickle serialization.

        Parameters
        ----------
        file : str
            A string that represents the name of the file
            where the object will be saved using pickle.dump.

        '''
        self.config.save(file)
