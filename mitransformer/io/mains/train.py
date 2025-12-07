from ...data import DataProvider
from ...train import (
    LMMetric, Result, LMTrainer, TrainConfig)
from ...models import (description_builder, MITransformerConfig)
from ...train.metrics import sum_and_std_metrics
from ...utils.params import dict_info
from .. import parsing
from . import functions

from tqdm import tqdm
import os
from copy import copy

from mitransformer.utils.logmaker import (
    getLogger, info)

from typing import Iterator, Tuple, overload, Literal

import time

logger = getLogger(__name__)


@overload
def main_train(
        arguments: "parsing.TrainParserArgs",
        world_size: int,
        iterate: Literal[False] = False,
        data_provider: DataProvider | None = None
        ) -> Tuple[LMTrainer, Result]:
    ...


@overload
def main_train(
        arguments: "parsing.TrainParserArgs",
        world_size: int,
        iterate: Literal[True],
        data_provider: DataProvider | None = None
        ) -> Tuple[LMTrainer, Iterator[Result]]:
    ...


def main_train(
        arguments: "parsing.TrainParserArgs",
        world_size: int,
        iterate: bool = False,
        data_provider: DataProvider | None = None
        ) -> (
            Tuple[LMTrainer, Iterator[Result]]
            | Tuple[LMTrainer, Result]):

    # device: where to execute computation
    if world_size > 1:
        assert arguments.rank is not None, (
            "Rank cannot be None if word_size > 1.")

    train_config = TrainConfig.from_kwargs(
        **arguments.to_dict(),
        world_size=world_size)

    if data_provider is None:
        # TODO: do this entirely in the load dataset method
        # load memmap
        data_provider = functions._load_data_provider(arguments, memmaped=True)

    # Model
    # 24 heads, one layer approximately matches CBR-RRN
    # TODO: Make this a proper config
    arguments.vocab_size = data_provider.datasets["token_mapper"].vocab_size

    # make proper transformer description
    if (arguments.transformer_description is None
            and arguments.masks_setting == "both"):
        arguments.transformer_description = (
            (('head_current', 'child_current'), 1),
            (('head_next', 'child_next'), 1))
    elif arguments.transformer_description is None:
        if arguments.layer_design is None:
            if arguments.masks_setting == "current":
                arguments.layer_design = (
                    "head_current", "child_current")
            elif arguments.masks_setting == "next":
                arguments.layer_design = (
                    "head_next", "child_next")
        arguments.transformer_description = description_builder(
            arguments.layer_design,
            arguments.use_standard,
            arguments.width,
            arguments.depth,
            arguments.unrestricted_before,
            arguments.unrestricted_after
        )
    n_heads: list[int] = [
        len(layer[0])*layer[1] for layer in arguments.transformer_description]

    # make n_embd divisible by number of heads in each layer
    for n_h_l in n_heads:
        arguments.n_embd = arguments.n_embd // n_h_l * n_h_l

    transformer_config = MITransformerConfig.from_kwargs(
        **arguments.to_dict(),
        use_input_mask=(arguments.dependency_mode == "input"))

    trainer = LMTrainer.new(transformer_config, train_config)
    if isinstance(data_provider, DataProvider):
        data_provider.save(os.path.join(trainer.model_dir,
                                        train_config.model_name,
                                        "data_config.json"))
    metrics: Result
    # Training setting
    if not iterate:
        metrics = trainer.train(**data_provider.datasets)
        if arguments.rank is None or arguments.rank == 0:
            generated = []
            for _ in range(20):
                generated.append(
                    trainer.generate(data_provider.datasets["token_mapper"]))
            info(
                arguments.rank, logger,
                f"Generated model output sample: {generated}")
        return (trainer, metrics)

    # Hyperopt setting
    else:
        generator = (metrics for metrics in trainer.train_iter(
            **data_provider.datasets
        ))
        return (trainer, generator)


MeanStdDict = dict[str, tuple[float, float]]


def main_train_multiple(
        arguments: "parsing.TrainParserArgs",
        world_size: int,
        data_provider: DataProvider | None = None
        ) -> (
            tuple[MeanStdDict, MeanStdDict]
            | tuple[MeanStdDict, MeanStdDict, MeanStdDict]):
    start = time.time()
    """Calculates the mean and standard deviation of several
    runs."""
    # How to keep track of results? We are logging them
    # but shouldn't we also forward them to tensorboard?
    # but we only have the final results or should we compute
    # the mean over the models for each step?
    if data_provider is None:
        data_provider = functions._load_data_provider(arguments, memmaped=True)

    assert arguments.n_runs != 0, "--n_runs cannot be 0"

    metrics_list: (
        list[tuple[LMMetric, LMMetric]]
        | list[tuple[LMMetric, LMMetric, LMMetric]]) = []
    for n_run in tqdm(range(arguments.n_runs), desc="Runs"):
        run_arguments = copy(arguments)
        run_arguments.model_name = f"{arguments.name}_{n_run}"
        run_arguments.seed = arguments.seed + n_run  # offset seed
        parsing.args_logic(run_arguments)  # also sets seed

        metrics_list.append(
            tuple(main_train(
                run_arguments,
                world_size,
                iterate=False,
                data_provider=data_provider)[1].values()))  # type: ignore

    means_and_stds = tuple(
        sum_and_std_metrics(seq)
        for seq in zip(*metrics_list))

    info(
        arguments.rank, logger,
        f"Performed {arguments.n_runs} training runs.")
    info(
        arguments.rank, logger,
        f"Final mean and std train: {dict_info(means_and_stds[0])}")
    info(
        arguments.rank, logger,
        f"Final mean and std dev: {dict_info(means_and_stds[1])}")
    if len(means_and_stds) > 2:
        info(
            arguments.rank, logger,
            f"Final mean and std test: {dict_info(means_and_stds[2])}")

    end = time.time()
    info(arguments.rank, logger, f"Took {end - start} seconds!")
    return means_and_stds  # type: ignore
