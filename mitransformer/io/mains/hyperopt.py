from ...train.metrics import (
    MetricWriter, metric_writer, minimise)
from ...data import get_loader
from .. import parsing, ddp
from . import functions, train

import optuna
import os
import pandas as pd

from mitransformer.utils.logmaker import (
    getLogger, info)

from typing import cast

logger = getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


USE_LOG = {"learning_rate"}


def hyperopt_arguments_sampler(
        name: str,
        arg: parsing.T | parsing.Choices[parsing.T] | parsing.Range | dict[
            str, parsing.T | parsing.Choices[parsing.T] | parsing.Range],
        trial
        ) -> parsing.T | dict[str, parsing.T]:
    if isinstance(arg, parsing.Choices):
        assert len(arg) > 0, f"Provided an empty selection for {name}!"
        str_choices: list[str] = [
            f"{num}_{str(choice)}" for num, choice in enumerate(arg)]
        str_arg: str = trial.suggest_categorical(name, str_choices)
        arg = arg[int(str_arg.split("_", 1)[0])]
    elif (
            isinstance(arg, parsing.Range)):
        if arg.is_continuous:
            arg = trial.suggest_float(
                name, arg[0], arg[1],
                log=name in USE_LOG)
        else:
            arg = trial.suggest_int(
                name, arg[0], arg[1],
                log=name in USE_LOG)

    if isinstance(arg, dict):
        arg = {subn: hyperopt_arguments_sampler(  # type: ignore
            f"{name}_{subn}", subv, trial)
            for subn, subv in arg.items()}
    else:
        arg = cast(parsing.T, arg)

    return arg  # type: ignore


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

        self.data_provider = None
        self.datasets = None

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
                    bucket=False,
                    shuffle=True, droplast=True,
                    world_size=self.n_devices,
                    rank=self.arguments.rank,
                    n_workers=self.arguments.n_workers)
            self.data_provider.datasets["eval"] = get_loader(  # type: ignore
                    self.data_provider.datasets["eval"],  # type: ignore
                    batch_size=self.arguments.batch_size,
                    bucket=False,
                    shuffle=False, droplast=False,
                    world_size=self.n_devices,
                    rank=self.arguments.rank,
                    n_workers=self.arguments.n_workers)
        except TypeError:
            self.data_provider = None

    def __call__(self, trial) -> float:
        if self.n_devices > 1:
            trial = optuna.integration.TorchDistributedTrial(
                trial, self.pg)  # type: ignore

        arguments = parsing.TrainParserArgs.from_kwargs(**{
            name: hyperopt_arguments_sampler(name, arg, trial) for
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
        for step, metrics in enumerate(train_iterator, start=1):
            # Handle pruning based on the intermediate value.
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
        # trial.set_user_attr("metric_dicts", metric_dicts)

        opt_metric = getattr(metrics["eval"], self.arguments.optimise.lower())
        if isinstance(opt_metric, pd.DataFrame):
            opt_metric = opt_metric.to_numpy().sum()
        loss: float = float(opt_metric)
        return loss


def main_hyperopt(
        arguments: "parsing.HyperoptParserArgs",
        world_size: int) -> None:
    direction = (
        "minimize" if minimise[arguments.optimise.lower().split(":")[0]]
        else "maximize")

    study: None | optuna.Study = None
    ld = os.path.join("./runs", f"{arguments.name}_hyperopt")
    with ddp.new_pg(
            world_size, "gloo") as pg, metric_writer(log_dir=ld) as writer:
        objective: Objective = Objective(world_size, arguments, writer, pg)
        if arguments.rank == 0 or arguments.rank is None:
            study = optuna.create_study(
                study_name=arguments.name,
                direction=direction,
                sampler=optuna.samplers.RandomSampler(
                    seed=arguments.seed),  # TODO: normal sampler
                pruner=optuna.pruners.MedianPruner(
                    n_warmup_steps=arguments.n_warmup_steps,
                    n_startup_trials=arguments.n_startup_trials))
            study.optimize(
                objective, n_trials=arguments.n_trials)

        else:
            for _ in range(arguments.n_trials):
                try:
                    objective(None)
                except optuna.TrialPruned:
                    pass

    if arguments.rank == 0 or arguments.rank is None:
        assert study is not None
        pruned_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        info(
            arguments.rank, logger,
            (
                f"Pruned {len(pruned_trials)}, "
                f"completed {len(complete_trials)} trials"))

        info(
            arguments.rank, logger,
            f"Best trial: {study.best_trial.number}\n"
            f"with results: {study.best_value}\n"
            f"with params: {study.best_params}")

    return None
