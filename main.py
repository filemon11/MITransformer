import torch
import torch.distributed as dist

from data import (load_dataset, dataset_details_full, DatasetDictTrain,
                  dataset_details_full_memmaped, get_loader, Dataset)
from trainer import (LMTrainer, TrainConfig, MITransformerConfig,
                     Metric, MetricWriter, metric_writer, sum_and_std_metrics)
from model import TransformerDescription, description_builder
from params import Params, dict_info

import optuna
import random
import os
import numpy as np
import sys
from contextlib import contextmanager
from dataclasses import dataclass
import argparse
from ast import literal_eval as make_tuple
from copy import copy

from logmaker import getLogger, info, logging_config, get_timestr

from typing import (
    Literal, Any, Iterable, cast, Iterator, TypeVar, Generic,
    Sequence, Callable, overload)

logger = getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


def seed_everything(seed: int):
    """There might be nondeterministic torch algorithms.
    We're not making them deterministic here."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # according to
    # https://pytorch.org/docs/stable/data.html#data-loading-randomness
    # each dataloader worker will have its PyTorch seed set to
    # base_seed + worker_id. Thus, with the same number
    # of workers, the process is deterministic


T = TypeVar("T")


def str_to_bool(string: str) -> bool:
    try:
        if (string.lower() == "true"
                or int(string) == 1):
            return True
    except ValueError:
        pass
    try:
        if (string.lower() == "false"
                or int(string) == 0):
            return False
    except ValueError:
        pass
    raise Exception((f"argument value {string} cannot"
                     "be parsed as a string!"))


class OptNone(Generic[T]):
    def __init__(self, type: Callable[[Any], T]):
        self.type = type

    def __call__(self, value: str) -> None | T:
        if value.lower() == 'none':
            return None
        try:
            return self.type(value)  # type: ignore
        except TypeError:
            raise Exception(
                f"Constructor for {self.type} does not accept an argument")


class HyperoptSpace(Generic[T]):
    def __init__(self, type: Callable[[Any], T],
                 choices: Sequence[T] | None = None):
        self.type = type
        self.choices = choices

    def __call__(self, value: str) -> tuple[T, T] | list[T] | T:
        # try to split via :
        split = value.split(":")
        if len(split) == 2:
            try:
                h_range = (self.type(split[0]),
                           self.type(split[1]))  # type: ignore
                assert (isinstance(h_range[0], (int, float, complex))
                        and not isinstance(h_range[0], bool)), (
                        f"Non-numeric range detected: {h_range}")
                return h_range
            except TypeError:
                raise Exception(
                    f"Constructor for {self.type} does not accept an argument")

        assert len(split) < 2, (
            f"Range must have one starting and one end point. Given: {value}")

        split = value.split(",")
        if len(split) == 0:
            try:
                return self.type(value)  # type: ignore
            except TypeError:
                raise Exception(
                    f"Constructor for {self.type} does not accept an argument")

        try:
            options = [self.type(v) for v in split]  # type: ignore
            if self.choices is not None:
                for o in options:
                    assert o in self.choices, (
                        f"{o} must be one of {self.choices}")
            return options
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


def _load_dataset(args: "ParserArgs", memmaped: bool = False
                  ) -> DatasetDictTrain:
    if memmaped:
        details = dataset_details_full_memmaped[args.dataset_name]
        details["memmaped"] = details["memmaped"][0:2]  # type: ignore
    else:
        details = dataset_details_full[args.dataset_name]
        details["dirs"] = details["dirs"][0:2]  # type: ignore
    datasets = load_dataset(
        details,
        args.max_len_train,
        args.max_len_eval_test,
        args.vocab_size,
        args.first_k,
        args.first_k_eval_test,
        args.triangulate,
        args.connect_with_dummy,
        args.connect_with_self)
    info(args.rank, logger, (
        "Loaded datasets with "
        + ', '.join([str(len(ds)) for ds  # type: ignore
                     in datasets.values() if isinstance(ds, Dataset)])
        + " sentences."))
    return datasets


def main_dataprep(args: "ParserArgs") -> None:
    _load_dataset(args, memmaped=False)


def main_train(
        args: "TrainParserArgs",
        world_size: int,
        iterate: bool = False,
        datasets: DatasetDictTrain | None = None
        ) -> (Iterator[tuple[Metric, Metric]]
              | Iterator[tuple[Metric, Metric, Metric]]):
    # device: where to execute computation
    if world_size > 1:
        assert args.rank is not None, "Rank cannot be None if word_size > 1."

    train_config = TrainConfig.from_kwargs(
        **args.to_dict())

    if datasets is None:
        # TODO: do this entirely in the load dataset method
        # load memmap
        datasets = _load_dataset(args, memmaped=True)
    assert isinstance(datasets, dict)

    # Model
    # 24 heads, one layer approximately matches CBR-RRN
    # TODO: Make this a proper config
    args.vocab_size = datasets["token_mapper"].vocab_size

    # make proper transformer description
    if args.transformer_description is None:
        args.transformer_description = description_builder(
            args.layer_design,
            args.use_standard,
            args.width,
            args.depth,
            args.unrestricted_before,
            args.unrestricted_after
        )
    # make n_embd divisible by number of heads
    n_heads = len(args.transformer_description[0][0])
    args.n_embd = args.n_embd // n_heads * n_heads

    transformer_config = MITransformerConfig.from_kwargs(
        **args.to_dict(),
        use_input_mask=(args.mode == "input"))

    trainer = LMTrainer.new(transformer_config, train_config)

    # Training setting
    if not iterate:
        metrics = trainer.train(**datasets)
        if args.rank is None or args.rank == 0:
            generated = []
            for _ in range(20):
                generated.append(trainer.generate(datasets["token_mapper"]))
            info(args.rank, logger,
                 f"Generated model output sample: {generated}")
        yield metrics  # type: ignore

    # Hyperopt setting
    else:
        for metrics in trainer.train_iter(**datasets):
            yield metrics

    del trainer
    del datasets


MeanStdDict = dict[str, tuple[float, float]]


def main_train_multiple(
        args: "TrainParserArgs",
        world_size: int,
        datasets: DatasetDictTrain | None = None
        ) -> (tuple[MeanStdDict, MeanStdDict]
              | tuple[MeanStdDict, MeanStdDict, MeanStdDict]):
    """Calculates the mean and standard deviation of several
    runs."""
    # How to keep track of results? We are logging them
    # but shouldn't we also forward them to tensorboard?
    # but we only have the final results or should we compute
    # the mean over the models for each step?
    datasets = _load_dataset(args, memmaped=True)

    assert args.n_runs != 0, "--n_runs cannot be 0"

    metrics_list: (list[tuple[Metric, Metric]]
                   | list[tuple[Metric, Metric, Metric]]) = []
    for n_run in range(args.n_runs):
        run_args = copy(args)
        run_args.model_name = f"{args.name}_{n_run}"
        run_args.seed = args.seed + n_run  # offset seed
        args_logic(run_args)  # also sets seed

        metrics_list.append(
            next(main_train(
                run_args,
                world_size,
                iterate=False,
                datasets=datasets)))  # type: ignore

    means_and_stds = tuple(sum_and_std_metrics(seq)
                           for seq in zip(*metrics_list))

    info(args.rank, logger,
         f"Performed {args.n_runs} training runs.")
    info(args.rank, logger,
         f"Final mean and std train: {dict_info(means_and_stds[0])}")
    info(args.rank, logger,
         f"Final mean and std dev: {dict_info(means_and_stds[1])}")
    if len(means_and_stds) > 2:
        info(args.rank, logger,
             f"Final mean and std test: {dict_info(means_and_stds[2])}")

    return means_and_stds  # type: ignore


def hyperopt_args_sampler(
        name: str,
        arg: T | list[T] | tuple[T, T],
        trial
        ) -> T:
    if isinstance(arg, list):
        assert len(arg) > 0, f"Provided an empty selection for {name}!"
        return trial.suggest_categorical(name, arg)
    elif (isinstance(arg, tuple)
          and len(arg) == 2
          and isinstance(arg[0], (int, float))):
        if isinstance(arg[0], float) and isinstance(arg[1], float):
            return trial.suggest_float(name, arg[0], arg[1])
        elif isinstance(arg[0], int) and isinstance(arg[1], int):
            return trial.suggest_int(name, arg[0], arg[1])
        else:
            raise Exception(
                f"Range {arg} for arg {name} inconsistently typed!")
    else:
        return cast(T, arg)


class Objective:
    def __init__(self, n_devices: int,
                 args: "HyperoptParserArgs",
                 writer: MetricWriter):
        self.n_devices = n_devices
        self.args = args
        self.writer = writer

        self.datasets = _load_dataset(args, memmaped=True)

        # Since pin_memory=True, persistent_workers=True lead to too many files
        # error when creating a lot of dataloaders, we need to construct
        # dataloaders here
        # Remove this if https://github.com/pytorch/pytorch/issues/91252
        # is resolved
        self.datasets["train"] = get_loader(  # type: ignore
                self.datasets["train"],  # type: ignore
                batch_size=self.args.batch_size,
                bucket=False,
                shuffle=True, droplast=True,
                world_size=self.n_devices,
                rank=self.args.rank,
                n_workers=self.args.n_workers)

        self.datasets["eval"] = get_loader(  # type: ignore
                self.datasets["eval"],  # type: ignore
                batch_size=self.args.batch_size,
                bucket=False,
                shuffle=False, droplast=False,
                world_size=self.n_devices,
                rank=self.args.rank,
                n_workers=self.args.n_workers)

    def __call__(self, trial) -> float:
        if self.n_devices > 1:
            trial = optuna.integration.TorchDistributedTrial(
                trial, None)  # type: ignore

        args = TrainParserArgs.from_kwargs(**{
            name: hyperopt_args_sampler(name, arg, trial) for
            name, arg in self.args.to_dict().items()},
            model_name=f"{self.args.name}_{trial.number}",
            n_runs=1)
        args.seed = args.seed + trial.number
        args_logic(args)

        train_iterator = main_train(
            args, self.n_devices,
            iterate=True, datasets=self.datasets)
        assert train_iterator is not None
        for step, metrics in enumerate(train_iterator, start=1):
            # Handle pruning based on the intermediate value.
            trial.report(
                getattr(metrics[1], self.args.optimise),
                args.eval_interval*step)

            if trial.should_prune():
                del train_iterator
                raise optuna.exceptions.TrialPruned()

        if self.writer is not None:
            self.writer.add_params(
                args.to_dict(),
                metrics[1])

        # trial.set_user_attr("metric_dicts", metric_dicts)

        loss: float = getattr(metrics[1], self.args.optimise)
        return loss


def main_hyperopt(args: "HyperoptParserArgs",
                  world_size: int) -> None:
    maximise = {"UAS"}
    direction = "maximize" if args.optimise in maximise else "minimize"

    with metric_writer() as writer:
        objective: Objective = Objective(world_size, args, writer)
        if args.rank == 0 or args.rank is None:
            study = optuna.create_study(
                study_name=args.name,
                direction=direction,
                sampler=optuna.samplers.RandomSampler(
                    seed=args.seed),  # TODO: normal sampler
                pruner=optuna.pruners.MedianPruner(
                    n_warmup_steps=args.n_warmup_steps,
                    n_startup_trials=args.n_startup_trials))
            study.optimize(
                objective, n_trials=args.n_trials)

        else:
            for _ in range(args.n_trials):
                try:
                    objective(None)
                except optuna.TrialPruned:
                    pass

    if args.rank == 0 or args.rank is None:
        assert study is not None
        pruned_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        info(args.rank, logger,
             (f"Pruned {len(pruned_trials)}, "
              f"completed {len(complete_trials)} trials"))

    return None


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
        info(None, logger,
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
        assert not ((n_devices == 1 or not args.use_ddp) and
                    (args.rank is not None and args.rank > 0)), (
            "Rank cannot be larger than 0 if only having one device"
            "/not using ddp. "
            f"Received --local-rank {args.rank} --use_ddp {args.use_ddp} "
            f"and number of recognised CUDA devices is {n_devices}.")
        info(None, logger, f"Running on {n_devices} devices.")
        with ddp(args.rank, n_devices) as ddp_status:
            info(args.rank, logger, f"Using DDP: {ddp_status}")
            mode = args.mode
            if mode == "train":
                assert isinstance(args, TrainParserArgs)
                info(args.rank, logger, "Launching model training.")
                main_train_multiple(args, n_devices)
            else:
                assert isinstance(args, HyperoptParserArgs)
                info(args.rank, logger, "Launching hyperparameter tuning.")
                main_hyperopt(args, n_devices)


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
    vocab_size: int | None
    triangulate: bool
    first_k: int | None
    first_k_eval_test: int | None
    connect_with_dummy: bool
    connect_with_self: bool
    seed: int


@dataclass
class TrainParserArgs(ParserArgs):
    n_runs: int

    model_name: str
    dependency_mode: Literal["supervised", "input", "standard"]
    batch_size: int
    use_steps: bool
    max_steps: int | None
    eval_interval: int
    early_stop_after: int | None
    early_stop_metric: str | None
    epochs: int
    learning_rate: float
    loss_alpha: float | None
    arc_loss_weighted: bool

    transformer_description: TransformerDescription | None
    layer_design: tuple[str, ...]
    use_standard: bool
    width: int
    depth: int
    unrestricted_before: int
    unrestricted_after: int
    d_ff_factor: int
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
    optimise: Literal["perplexity", "uas", "loss", "lm_loss", "arc_loss"]
    n_warmup_steps: int
    n_startup_trials: int
    n_trials: int

    dependency_mode: Literal["supervised", "input", "standard"]
    batch_size: int
    use_steps: bool
    max_steps: int | None
    eval_interval: int
    early_stop_after: int | None
    early_stop_metric: str | None
    epochs: int

    learning_rate: float | tuple[float, float] | list[float]
    loss_alpha: float | tuple[float, float] | list[float | None] | None
    arc_loss_weighted: bool | list[bool]

    block_size: int
    overlay_causal: bool

    transformer_description: (TransformerDescription
                              | list[TransformerDescription])
    layer_design: tuple[str, ...] | list[tuple[str, ...]]
    use_standard: bool | list[bool]
    width: int | tuple[int, int] | list[int]
    depth: int | tuple[int, int] | list[int]
    unrestricted_before: int | tuple[int, int] | list[int]
    unrestricted_after: int | tuple[int, int] | list[int]
    d_ff_factor: int | tuple[int, int] | list[int]
    dropout: float | tuple[int, int] | list[int | None] | None
    dropout_attn: float | tuple[int, int] | list[int | None] | None
    dropout_resid: float | tuple[int, int] | list[int | None] | None
    dropout_ff: float | tuple[int, int] | list[int | None] | None
    dropout_embd: float | tuple[int, int] | list[int | None] | None
    dropout_lstm: float | tuple[int, int] | list[int | None] | None
    n_embd: int | tuple[int, int] | list[int]
    use_dual_fixed: bool | list[bool]
    bias: bool | list[bool]


@dataclass
class DataprepParserArgs(ParserArgs):
    pass

# TODO: make bool args optionally just acccept flag for True


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
        help="device to run models on; must be 'cuda' if --use_ddp is set")
    parser.add_argument(
        '--use_ddp', type=str_to_bool, default=torch.cuda.device_count() > 1,
        help="whether to use distributed GPU training")
    parser.add_argument(
        '--seed', type=int, default=1895,
        help="seed for random processes")
    # TODO: actually set seed

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
        '--triangulate', type=str_to_bool, default=True,
        help='TODO')
    data_group.add_argument(
        '--vocab_size', type=OptNone(int), default=50_000,
        help=('number of most frequent tokens to embed; all other '
              'tokens are replaced with an UNK token;'
              'can be None when loading existing token_mapper'))
    data_group.add_argument(
        '--first_k', type=OptNone(int), default=None,
        help='only load first k sentences of the training set')
    data_group.add_argument(
        '--first_k_eval_test', type=OptNone(int), default=None,
        help='only load first k sentences of the eval and test sets')
    data_group.add_argument(
        '--connect_with_dummy', type=str_to_bool, default=True,
        help=('Establish an arc to a dummy token when there is no '
              'parent/child among the precedents?'))
    data_group.add_argument(
        '--connect_with_self', type=str_to_bool, default=False,
        help=('Establish a recursive arc to the token itself when there '
              'is not parent/child among the precedents?'))

    # Subparsers
    # # Training Parser
    subparsers = parser.add_subparsers(dest="mode")
    train_parser = subparsers.add_parser(
        "train", help="training mode")
    train_parser.add_argument(
        '--n_runs', type=int,
        default=1,
        help="Number of runs; if > 1 computes mean and std of final scores.")

    # # # Trainer parser group
    trainer_group = train_parser.add_argument_group('trainer')
    trainer_group.add_argument(
        '--model_name', type=OptNone(str),
        default=None,
        help="model name. Set to experiment name if None")
    trainer_group.add_argument(
        '--dependency_mode', type=str,
        choices=("supervised", "input", "standard"),
        default="supervised",
        help="how to use dependency information")
    trainer_group.add_argument(
        '--batch_size', type=int, default=32,
        help=("batch size; in case of multiple GPUs it is "
              "chunked across the devices"))
    trainer_group.add_argument(
        '--use_steps', type=str_to_bool, default=False,
        help=("Where the unit of evaluation are steps (i.e. batches) "
              "instead of epochs. If true, --eval_interval "
              "refers to number of batches processed."))
    trainer_group.add_argument(
        '--max_steps', type=OptNone(int), default=100_000,
        help=("maximum number of steps (batches) to process "
              "if --use_steps is set to true"))
    trainer_group.add_argument(
        '--eval_interval', type=int,
        default=1, help="frequency to perform evaluations in")
    trainer_group.add_argument(
        '--early_stop_after', type=OptNone(int), default=None,
        help=("abort training after x evaluations without "
              "improvement on the eval loss; if None, no early stopping"))
    trainer_group.add_argument(
        '--early_stop_metric', type=OptNone(str), default='perplexity',
        help=("metric to chose for early stopping if "
              "--early_stop_after is not none"))
    trainer_group.add_argument(
        '--epochs', type=int, default=100,
        help="how many epochs to train for")
    trainer_group.add_argument(
        '--learning_rate', type=float, default=1e-3,
        help="learning rate for the optimiser")
    trainer_group.add_argument(
        '--loss_alpha', type=OptNone(float), default=0.5,
        help=("loss weight for supervised learning; 1.0 is only "
              "language model training while 0.0 is only arc training"))
    trainer_group.add_argument(
        '--arc_loss_weighted', type=str_to_bool, default=False,
        help="Overrepresent arcs against non-arcs in arc loss calculation")

    # # # Model parser group
    model_group = train_parser.add_argument_group('model')
    model_group.add_argument(
        '--transformer_description',
        type=OptNone(make_tuple), default=None,
        help=("Architecture of the transformer model. Tuple of layers "
              "where each layer is a tuple of a tuple of "
              "head types and a width."
              "The width is applied to every head type in the layer."
              "If provised, overrides --layer_design, --use_standard, "
              "--width, --depth, --unrestricted_before, --unrestricted_after"))
    model_group.add_argument(
        '--layer_design',
        type=make_tuple, default=("head", "child"),
        help=("design of the core transformer layer; tuple of head types "
              "is overriden if --transformer_description is provided"))
    model_group.add_argument(
        '--use_standard',
        type=str_to_bool, default=False,
        help=("whether to add an unrestricted head to the core layer(s)"))
    model_group.add_argument(
        '--width',
        type=int, default=1,
        help=("width of the core transformer; "
              "is overriden if --transformer_description is provided"))
    model_group.add_argument(
        '--depth',
        type=int, default=1,
        help=("depth of the core transformer; "
              "is overriden if --transformer_description is provided"))
    model_group.add_argument(
        '--unrestricted_before',
        type=int, default=0,
        help=("number of unrestricted layers below the core transformer; "
              "is overriden if --transformer_description is provided"))
    model_group.add_argument(
        '--unrestricted_after',
        type=int, default=0,
        help=("number of unrestricted layers above the core transformer; "
              "is overriden if --transformer_description is provided"))
    model_group.add_argument(
        '--d_ff_factor', type=int, default=4,
        help="hidden dimensionality of the feed-forward layers (*n_embd)")
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
        '--overlay_causal', type=str_to_bool, default=True,
        help=("whether to overlay a casual mask if providing"
              "masks as additional inputs"))
    model_group.add_argument(
        '--use_dual_fixed', type=str_to_bool, default=False,
        help=("whether to cross-fix key and query weights of "
              "the head and child attention heads. Only allowed "
              "if both are present in the description with width 1"))
    model_group.add_argument(
        '--bias', type=str_to_bool, default=False,
        help="Whether to use bias in all of the model weights")

    # # Hyperopt Parser
    hyperopt_parser = subparsers.add_parser(
        "hyperopt", help="hyperopt mode")
    hyperopt_parser.add_argument(
        '--optimise', type=str,
        choices=("perplexity", "uas", "loss", "lm_loss", "arc_loss"),
        default="perplexity",
        help="metric to optimise")
    hyperopt_parser.add_argument(
        '--n_warmup_steps', type=int,
        default=1,
        help=("how many epochs to wait before pruning can "
              "happen within a trial"))
    hyperopt_parser.add_argument(
        '--n_startup_trials', type=int,
        default=5,
        help="how many trials to run before pruning can happen at all")
    hyperopt_parser.add_argument(
        '--n_trials', type=int,
        default=25,
        help="how many trials to run")

    # # # Hyperopt Fixed Trainer parser group
    hyperopt_fixed_trainer_group = hyperopt_parser.add_argument_group(
        'trainer_fixed')
    hyperopt_fixed_trainer_group.add_argument(
        '--dependency_mode', type=str,
        choices=("supervised", "input", "standard"),
        default="supervised",
        help="how to use dependency information")
    hyperopt_fixed_trainer_group.add_argument(
        '--batch_size', type=int, default=32,
        help=("batch size; in case of multiple GPUs it is"
              "chunked across the devices"))
    hyperopt_fixed_trainer_group.add_argument(
        '--use_steps', type=str_to_bool, default=False,
        help=("Where the unit of evaluation are steps (i.e. batches) "
              "instead of epochs. If true, --eval_interval "
              "and n_warmup_steps refer to number of batches processed."))
    hyperopt_fixed_trainer_group.add_argument(
        '--max_steps', type=OptNone(int), default=100_000,
        help=("maximum number of steps (batches) to process "
              "if --use_steps is set to true"))
    hyperopt_fixed_trainer_group.add_argument(
        '--eval_interval', type=int,
        default=1, help="frequency to perform evaluations in")
    hyperopt_fixed_trainer_group.add_argument(
        '--early_stop_after', type=OptNone(int), default=None,
        help=("abort training after x evaluations without"
              "improvement on the eval loss"))
    hyperopt_fixed_trainer_group.add_argument(
        '--early_stop_metric', type=str, default='perplexity',
        help=("metric to chose for early stopping if "
              "--early_stop_after is not none"))
    hyperopt_fixed_trainer_group.add_argument(
        '--epochs', type=int, default=100,
        help="how many epochs to train for")

    # # # Hyperopt Flexible Trainer parser group
    hyperopt_flexible_trainer_group = hyperopt_parser.add_argument_group(
        'trainer flexible')
    hyperopt_flexible_trainer_group.add_argument(
        '--learning_rate', type=HyperoptSpace(float), default=1e-3,
        help="learning rate for the optimiser")
    hyperopt_flexible_trainer_group.add_argument(
        '--loss_alpha', type=HyperoptSpace(float), default=0.5,
        help=("loss weight for supervised learning; 1.0 is only"
              "language model training while 0.0 is only arc training"))
    hyperopt_flexible_trainer_group.add_argument(
        '--arc_loss_weighted', type=HyperoptSpace(str_to_bool), default=False,
        help="Overrepresent arcs against non-arcs in arc loss calculation")

    # # # Hyperopt Fixed Model parser group
    hyperopt_fixed_model_group = hyperopt_parser.add_argument_group(
        'model fixed')
    hyperopt_fixed_model_group.add_argument(
        '--block_size', type=int, default=500,
        help="maximum sequence length of the model")
    hyperopt_fixed_model_group.add_argument(
        '--overlay_causal', type=str_to_bool, default=True,
        help=("whether to overlay a casual mask if providing"
              "masks as additional inputs"))

    # # # Hyperopt Flexible Model parser group
    hyperopt_flexbile_model_group = hyperopt_parser.add_argument_group(
        'model flexible')
    hyperopt_flexbile_model_group.add_argument(
        '--transformer_description',
        type=HyperoptSpace(OptNone(make_tuple)),
        default=((('head', 'child'), 1),),
        help=("Architecture of the transformer model. Tuple of layers "
              "where each layer is a tuple of a tuple of "
              "head types and a width."
              "The width is applied to every head type in the layer."))
    hyperopt_flexbile_model_group.add_argument(
        '--layer_design',
        type=HyperoptSpace(make_tuple), default=("head", "child"),
        help=("design of the core transformer layer; tuple of head types "
              "is overriden if --transformer_description is provided"))
    hyperopt_flexbile_model_group.add_argument(
        '--use_standard',
        type=HyperoptSpace(str_to_bool), default=False,
        help=("whether to add an unrestricted head to the core layer(s)"))
    hyperopt_flexbile_model_group.add_argument(
        '--width',
        type=HyperoptSpace(int), default=1,
        help=("width of the core transformer; "
              "is overriden if --transformer_description is provided"))
    hyperopt_flexbile_model_group.add_argument(
        '--depth',
        type=HyperoptSpace(int), default=1,
        help=("depth of the core transformer; "
              "is overriden if --transformer_description is provided"))
    hyperopt_flexbile_model_group.add_argument(
        '--unrestricted_before',
        type=HyperoptSpace(int), default=0,
        help=("number of unrestricted layers below the core transformer; "
              "is overriden if --transformer_description is provided"))
    hyperopt_flexbile_model_group.add_argument(
        '--unrestricted_after',
        type=HyperoptSpace(int), default=0,
        help=("number of unrestricted layers above the core transformer; "
              "is overriden if --transformer_description is provided"))
    hyperopt_flexbile_model_group.add_argument(
        '--d_ff_factor', type=HyperoptSpace(int), default=4,
        help="hidden dimensionality of the feed-forward layers (*n_embd)")
    hyperopt_flexbile_model_group.add_argument(
        '--dropout', type=HyperoptSpace(OptNone(float)), default=0.3,
        help=("hidden dimensionality of the feed-forward layers; "
              "if provided, fixes the search for all specific dropouts;"
              "if provided, specific dropouts establish individual "
              " search spaces"))
    hyperopt_flexbile_model_group.add_argument(
        '--dropout_attn', type=HyperoptSpace(OptNone(float)), default=-1,
        help=("dropout for the attention module; "
              "overriden by --dropout if specified"))
    hyperopt_flexbile_model_group.add_argument(
        '--dropout_resid', type=HyperoptSpace(OptNone(float)), default=-1,
        help=("dropout for the residual connections; "
              "overriden by --dropout if specified"))
    hyperopt_flexbile_model_group.add_argument(
        '--dropout_ff', type=HyperoptSpace(OptNone(float)), default=-1,
        help=("dropout for the feed-forward layers; "
              "overriden by --dropout if specified"))
    hyperopt_flexbile_model_group.add_argument(
        '--dropout_embd', type=HyperoptSpace(OptNone(float)), default=-1,
        help=("dropout for the embedding layer; "
              "overriden by --dropout if specified"))
    hyperopt_flexbile_model_group.add_argument(
        '--dropout_lstm', type=HyperoptSpace(OptNone(float)), default=-1,
        help=("dropout for the LSTM (if existant); "
              "overriden by --dropout if specified"))
    hyperopt_flexbile_model_group.add_argument(
        '--n_embd', type=HyperoptSpace(int), default=500,
        help="model embedding size")
    hyperopt_flexbile_model_group.add_argument(
        '--use_dual_fixed', type=HyperoptSpace(str_to_bool), default=False,
        help=("whether to cross-fix key and query weights of "
              "the head and child attention heads. Only allowed "
              "if both are present in the description with width 1"))
    hyperopt_flexbile_model_group.add_argument(
        '--bias', type=HyperoptSpace(str_to_bool), default=False,
        help="Whether to use bias in all of the model weights")

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


def args_logic(args: TrainParserArgs | HyperoptParserArgs | DataprepParserArgs
               ) -> None:
    seed_everything(args.seed)
    args.device = make_device_str(args.device)
    if isinstance(args, TrainParserArgs):
        # parse transformer description

        # override dropout that was not set specifically
        for specific_dropout in ("dropout_attn", "dropout_resid",
                                 "dropout_ff", "dropout_embd",
                                 "dropout_lstm"):
            if getattr(args, specific_dropout) == -1:
                setattr(args, specific_dropout, args.dropout)

        if args.model_name is None:
            args.model_name = args.name

    if args.rank is None and args.use_ddp:
        try:
            args.rank = int(os.environ["LOCAL_RANK"])
        except AttributeError:
            pass


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

    args_logic(args)

    # For hyperopt we deal with dropout later

    # Some checks
    assert not (args.use_ddp and args.device == "cpu"), (
        "Must set --device to a GPU when setting --use_ddp. "
        f"Received --device {args.device}, --use_ddp {args.use_ddp}")

    info(None, logger, f"Arguments provided: {str(sys.argv)}")
    main(args)
    # if n_devices > 1:
    #     mp.spawn(main, args=(n_devices,), nprocs=n_devices)  # type: ignore
    # else:
    #     main(None, n_devices)
