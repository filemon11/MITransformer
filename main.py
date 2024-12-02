import torch
import torch.distributed as dist

from data import (load_dataset, dataset_details_full, DatasetDictTrain,
                  dataset_details_full_memmaped)
from trainer import (LMTrainer, TrainConfig, MITransformerConfig,
                     Metric, MetricWriter)
from params import Params

from hyperopt import hyperopt
import pandas as pd
import os.path
import sys
from contextlib import contextmanager
from dataclasses import dataclass
import argparse
from ast import literal_eval as make_tuple


from logmaker import getLogger, info, logging_config, get_timestr

from typing import (
    Literal, Any, Iterable, cast, Iterator, Type, TypeVar, Generic)

logger = getLogger(__name__)

# set the random seed, for reproducibility
torch.manual_seed(42)


T = TypeVar("T")

class OptNone(Generic[T]):
    def __init__(self, type: Type[T]):
        self.type = type

    def __call__(self, value: str) -> None | T:
        if value.lower() == 'none':
            return None
        try:
            return self.type(value)  # type: ignore
        except TypeError:
            raise Exception(
                f"Constructor for {self.type} does not accept an argument")


def make_device_str(string: str) -> str:
    try:
        return "cuda:" + str(int(string))
    except ValueError:
        return string


"""
TOOD:
- establish dataset naming and loading by name
- save loading config with dataset
- if no name: save as Wikitext_1, _2 etc.
- when loading look for all datasets in dataset folder
and check if there is a config that fits;
then load this dataset
and if no, search for dataset having this name
if no, search for preset datasets and load new
if not, search on huggingface and parse and load new.
"""


def main_dataprep(args: "ParserArgs") -> None:
    details = dataset_details_full[args.dataset_name]
    details["dirs"] = details["dirs"][0:2]  # type: ignore
    load_dataset(
        details,
        args.max_len_train,
        args.max_len_eval_test,
        args.vocab_size,
        args.triangulate,
        args.first_k,
        args.first_k_eval_test,
        args.connect_with_dummy,
        args.connect_with_self)


def main_train(
        args: "TrainParserArgs",
        world_size: int) -> None:
    args.vocab_size = None
    # device: where to execute computation
    if world_size > 1:
        assert args.rank is not None, "Rank cannot be None if word_size > 1."

    train_config = TrainConfig.from_kwargs(
        **args.to_dict())

    datasets: DatasetDictTrain
    # TODO: do this entirely in the load dataset method
    # load memmap
    details = dataset_details_full_memmaped[args.dataset_name]
    details["memmaped"] = details["memmaped"][0:2]  # type: ignore
    datasets = load_dataset(
        details,
        max_len_train=args.max_len_train,
        max_len_eval_test=args.max_len_eval_test,
        vocab_size=args.vocab_size,
        triangulate=args.triangulate,
        first_k=args.first_k,
        first_k_eval_test=args.first_k_eval_test,
        connect_with_dummy=args.connect_with_dummy,
        connect_with_self=args.connect_with_self)
    assert isinstance(datasets, dict)

    # Model
    # 24 heads, one layer approximately matches CBR-RRN
    # TODO: Make this a proper config
    args.vocab_size = datasets["token_mapper"].vocab_size
    transformer_config = MITransformerConfig.from_kwargs(
        **args.to_dict(),
        use_input_mask=(args.mode == "input"))

    trainer = LMTrainer.new(transformer_config, train_config)
    trainer.train(**datasets)

    if args.rank is None or args.rank == 0:
        generated = []
        for _ in range(20):
            generated.append(trainer.generate(datasets["token_mapper"]))
        info(args.rank, logger, f"Generated model output sample: {generated}")

    del trainer


def main_hyperopt(args: "HyperoptParserArgs",
                  world_size: int) -> None:
    hyperopt(args.rank, world_size, args.objective,
             10, args.dataset_name, args.n_trials)


def options(seq_of_items: list[tuple[str, Iterable[Any]]]
            ) -> Iterable[dict[str, Any]]:
    name, values = seq_of_items[0]
    for value in values:
        if len(seq_of_items[1:]) == 0:
            yield {name: value}
        else:
            for d in options(seq_of_items[1:]):
                d[name] = value
                yield d


def setup_ddp(rank, world_size) -> bool:
    if world_size > 1:
        info(rank, logger,
             ("Initialising process with world_size "
              f"{world_size} and rank {rank}."))
        dist.init_process_group("nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(torch.distributed.get_rank())
        return True
    else:
        return False


def clean_ddp(world_size) -> None:
    if world_size > 1:
        dist.destroy_process_group()


@contextmanager
def ddp(rank: int | None, world_size: int) -> Iterator[bool]:
    try:
        yield setup_ddp(rank, world_size)
    finally:
        clean_ddp(world_size)


def main(args: "ParserArgs") -> None:
    if args.mode == "dataprep":
        main_dataprep(args)
    else:
        n_devices = torch.cuda.device_count() if args.use_ddp else 1
        info(None, logger, f"Running on {n_devices} devices.")
        with ddp(args.rank, n_devices) as ddp_status:
            info(args.rank, logger, f"Using DDP: {ddp_status}")
            mode = args.mode
            if mode == "train":
                assert isinstance(args, TrainParserArgs)
                info(args.rank, logger, "Launching model training.")
                main_train(args, n_devices)
            else:
                assert isinstance(args, HyperoptParserArgs)
                info(args.rank, logger, "Launching hyperparameter tuning.")
                main_hyperopt()


@dataclass
class ParserArgs(Params):
    mode: Literal["train", "hyperopt"]
    rank: int | None
    n_workers: int
    name: str
    device: str
    use_ddp: bool
    dataset_name: str
    max_len_train: None | int
    max_len_eval_test: None | int
    vocab_size: int
    triangulate: bool
    first_k: int | None
    first_k_eval_test: int | None
    connect_with_dummy: bool
    connect_with_self: bool


@dataclass
class TrainParserArgs(ParserArgs):
    dependency_mode: Literal["supervised", "input", "standard"]
    batch_size: int
    eval_interval: int
    abort_after: int | None
    epochs: int
    learning_rate: float
    loss_alpha: float | None
    arc_loss_weighted: bool

    transformer_description: str
    d_ff: int
    dropout: float | None
    dropout_attn: float | None
    dropout_resid: float | None
    dropout_ff: float | None
    dropout_embd: float | None
    dropout_lstm: float | None
    block_size: int
    n_embd: int
    overlay_causal: bool
    use_dual_fixed: bool
    bias: bool


@dataclass
class HyperoptParserArgs(ParserArgs):
    pass


@dataclass
class DataprepParserArgs(ParserArgs):
    pass


def parse_args() -> TrainParserArgs | HyperoptParserArgs | DataprepParserArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rank', '--local-rank', type=OptNone(int), default=None,
        help="which rank this process runs on; set by torchrun.")
    parser.add_argument(
        '--n_workers', type=OptNone(int), default=None,
        help="number of workers for the dataloader")
    parser.add_argument(
        '--name', type=str,
        default=get_timestr(),
        help="experiment name. Defaults to current time")
    parser.add_argument(
        '--device', type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to run models on; must be 'cuda' if --use_ddp is set"
    )
    parser.add_argument(
        '--use_ddp', type=bool, default=torch.cuda.device_count() > 1,
        help="whether to use distributed GPU training"
    )

    # Data parser group
    data_group = parser.add_argument_group('data')
    data_group.add_argument(
        '--dataset_name', type=str, help='name of the dataset to load',
        default='Wikitext')
    data_group.add_argument(
        '--max_len_train', type=OptNone(int), default=40,
        help='maximum number of tokens in training set')
    data_group.add_argument(
        '--max_len_eval_test', type=OptNone(int), default=None,
        help='maximum number of tokens in eval set')
    data_group.add_argument(
        '--triangulate', type=bool, default=True,
        help='TODO')
    data_group.add_argument(
        '--vocab_size', type=OptNone(int), default=50_000,
        help=('number of most frequent tokens to embed; all other '
              'tokens are replaced with an UNK token'))
    data_group.add_argument(
        '--first_k', type=OptNone(int), default=None,
        help='only load first k sentences of the training set')
    data_group.add_argument(
        '--first_k_eval_test', type=OptNone(int), default=None,
        help='only load first k sentences of the eval and test sets')
    data_group.add_argument(
        '--connect_with_dummy', type=bool, default=True,
        help=('Establish an arc to a dummy token when there is no '
              'parent/child among the precedents?'))
    data_group.add_argument(
        '--connect_with_self', type=bool, default=False,
        help=('Establish a recursive arc to the token itself when there '
              'is not parent/child among the precedents?'))

    # Subparsers
    # # Training Parser
    subparsers = parser.add_subparsers(dest="mode")
    train_parser = subparsers.add_parser(
        "train", help="training mode")

    # # # Trainer parser group
    trainer_group = train_parser.add_argument_group('trainer')
    trainer_group.add_argument(
        '--dependency_mode', type=str,
        choices=("supervised", "input", "standard"),
        default="supervised",
        help="how to use dependency information")
    trainer_group.add_argument(
        '--batch_size', type=int, default=32,
        help=("batch size; in case of multiple GPUs it is"
              "chunked across the devices"))
    trainer_group.add_argument(
        '--eval_interval', type=int,
        default=1, help="frequency to perform evaluations in")
    trainer_group.add_argument(
        '--abort_after', type=OptNone(int), default=None,
        help=("abort training after x evaluations without"
              "improvement on the eval score"))
    trainer_group.add_argument(
        '--epochs', type=int, default=100,
        help="how many epochs to train for")
    trainer_group.add_argument(
        '--learning_rate', type=float, default=1e-3,
        help="learning rate for the optimiser")
    trainer_group.add_argument(
        '--loss_alpha', type=OptNone(float), default=0.5,
        help=("loss weight for supervised learning; 1.0 is only"
              "language model training while 0.0 is only arc training"))
    trainer_group.add_argument(
        '--arc_loss_weighted', type=bool, default=False,
        help="Overrepresent arcs against non-arcs in arc loss calculation")

    # # # Model parser group
    model_group = train_parser.add_argument_group('model')
    model_group.add_argument(
        '--transformer_description',
        type=str, default="((('head', 'child'), 1),)",
        help=("Architecture of the transformer model. Tuple of layers "
              "where each layer is a tuple of a tuple of "
              "head types and a width."
              "The width is applied to every head type in the layer."))
    model_group.add_argument(
        '--d_ff', type=int, default=2000,
        help="hidden dimensionality of the feed-forward layers")
    model_group.add_argument(
        '--dropout', type=OptNone(float), default=0.3,
        help=("hidden dimensionality of the feed-forward layers; "
              "can be further specified by additional dropout params"))
    model_group.add_argument(
        '--dropout_attn', type=OptNone(float), default=-1,
        help=("dropout for the attention module; "
              "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_resid', type=OptNone(float), default=-1,
        help=("dropout for the residual connections; "
              "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_ff', type=OptNone(float), default=-1,
        help=("dropout for the feed-forward layers; "
              "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_embd', type=OptNone(float), default=-1,
        help=("dropout for the embedding layer; "
              "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--dropout_lstm', type=OptNone(float), default=-1,
        help=("dropout for the LSTM (if existant); "
              "overridden by --dropout if set to -1"))
    model_group.add_argument(
        '--block_size', type=int, default=500,
        help="maximum sequence length of the model")
    model_group.add_argument(
        '--n_embd', type=int, default=500,
        help="model embedding size")
    model_group.add_argument(
        '--overlay_causal', type=bool, default=True,
        help=("whether to overlay a casual mask if providing"
              "masks as additional inputs"))
    model_group.add_argument(
        '--use_dual_fixed', type=bool, default=False,
        help=("whether to cross-fix key and query weights of "
              "the head and child attention heads. Only allowed "
              "if both are present in the description with width 1"))
    model_group.add_argument(
        '--bias', type=bool, default=False,
        help="Whether to use bias in all of the model weights")

    # # Hyperopt Parser
    hyperopt = subparsers.add_parser(
        "hyperopt", help="hyperopt mode")

    # # Dataprep Parser
    dataprep = subparsers.add_parser(
        "dataprep", help="dataprep mode")

    args = parser.parse_args()
    match args.mode:
        case "train":
            return TrainParserArgs(**vars(args))
        case "hyperopt":
            return HyperoptParserArgs(**vars(args))
        case _:
            return DataprepParserArgs(**vars(args))


if __name__ == "__main__":
    args: TrainParserArgs | HyperoptParserArgs | DataprepParserArgs
    args = parse_args()
    logging_config(logname=args.name)
    # logging_config(
    #     logname="log",
    #     logpath=os.path.join(LMTrainer.model_dir, name))
    #  TODO: save trainer and model log in model dir
    #  but general and hyperopt log in normal logdir

    # tokenise dataset
    # TODO: do this automatically
    # problem: cannot do in parallel and
    # therefore leads to timeout when doing
    # on only one rank.

    args.device = make_device_str(args.device)

    if isinstance(args, TrainParserArgs):
        # parse transformer description
        args.transformer_description = make_tuple(args.transformer_description)

        # override dropout that was not set specifically
        for specific_dropout in ("dropout_attn", "dropout_resid",
                                 "dropout_ff", "dropout_embd",
                                 "dropout_lstm"):
            if getattr(args, specific_dropout) == -1:
                setattr(args, specific_dropout, args.dropout)

    # Some checks
    assert not (args.use_ddp and args.device == "cpu"), (
        "Must set --device to a GPU when setting --use_ddp. "
        f"Received --device {args.device}, --use_ddp {args.use_ddp}"
    )

    info(None, logger, f"Arguments provided: {str(sys.argv)}")
    main(args)
    # if n_devices > 1:
    #     mp.spawn(main, args=(n_devices,), nprocs=n_devices)  # type: ignore
    # else:
    #     main(None, n_devices)
