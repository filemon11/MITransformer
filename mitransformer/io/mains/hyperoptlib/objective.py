from ....train.metrics import (
    MetricWriter)
from ....train import LMMetric
from ....data import get_loader, DataProvider
from ... import parsing
from .. import functions, train
from . import sampler

import optuna
import pandas as pd

from mitransformer.utils.logmaker import (
    getLogger)

logger = getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


class Objective:
    def __init__(
            self, n_devices: int,
            arguments: "parsing.HyperoptParserArgs",
            writer: MetricWriter,
            pg):
        self.n_devices = n_devices
        self.arguments = arguments
        self.writer = writer
        self.pg = pg

        self.data_provider: None | DataProvider = None

        # TODO: do not use try but check if any of the relevant arguments are
        # Hyperopt spaces
        try:
            self.data_provider = functions._load_data_provider(
                arguments, memmaped=True)
            # Since pin_memory=True, persistent_workers=True lead
            # to too many files
            # error when creating a lot of dataloaders, we need to construct
            # dataloaders here
            # Remove this if https://github.com/pytorch/pytorch/issues/91252
            # is resolved
            self.data_provider.datasets["train"] = get_loader(  # type: ignore
                    self.data_provider.datasets["train"],  # type: ignore
                    batch_size=self.arguments.batch_size,
                    bucket=True,
                    shuffle=True, droplast=False,
                    world_size=self.n_devices,
                    rank=self.arguments.rank,
                    n_workers=self.arguments.n_workers,
                    seed=arguments.seed)
            self.data_provider.datasets["eval"] = get_loader(  # type: ignore
                    self.data_provider.datasets["eval"],  # type: ignore
                    batch_size=self.arguments.batch_size,
                    bucket=True,
                    shuffle=True, droplast=False,
                    world_size=self.n_devices,
                    rank=self.arguments.rank,
                    n_workers=self.arguments.n_workers,
                    seed=arguments.seed)
        except TypeError:
            self.data_provider = None

    def __call__(self, trial) -> float:
        if self.n_devices > 1:
            trial = optuna.integration.TorchDistributedTrial(
                trial, self.pg)  # type: ignore

        arguments = parsing.TrainParserArgs.from_kwargs(**{
            name: sampler.hyperopt_arguments_sampler(name, arg, trial) for
            name, arg in self.arguments.to_dict().items()},
            model_name=f"{self.arguments.name}_{trial.number}",
            n_runs=1)
        arguments.seed = arguments.seed + trial.number
        parsing.args_logic(arguments)

        _, train_iterator = train.main_train(
            arguments, self.n_devices,
            iterate=True,
            data_provider=self.data_provider)
        assert train_iterator is not None

        should_prune = False
        metrics = None
        step = 0
        best_eval_metric: LMMetric | float | None = None
        for step, metrics in enumerate(train_iterator, start=1):
            # Handle pruning based on the intermediate value.
            if best_eval_metric is None:
                best_eval_metric = metrics["eval"].minval()
            if metrics["eval"] > best_eval_metric:
                best_eval_metric = metrics["eval"]

            opt_metric = getattr(
                metrics["eval"], self.arguments.optimise.lower())
            if isinstance(opt_metric, pd.DataFrame):
                opt_metric = float(opt_metric.to_numpy().sum())
            trial.report(
                opt_metric,
                step)

            if trial.should_prune():
                should_prune = True
                break

        assert metrics is not None, (
            "eval_interval is larger than total number of steps")
        if self.writer is not None:
            self.writer.add_params(
                arguments.to_dict(),
                metrics["eval"],
                run_name=str(trial.number),
                global_step=arguments.eval_interval*step)

        if should_prune:
            raise optuna.exceptions.TrialPruned()

        opt_metric = getattr(best_eval_metric, self.arguments.optimise.lower())
        if isinstance(opt_metric, pd.DataFrame):
            opt_metric = opt_metric.to_numpy().sum()
        loss: float = float(opt_metric)
        return loss
