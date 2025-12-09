from ...train.metrics import (
    metric_writer, minimise)
from .. import parsing, ddp
from . import hyperoptlib

import optuna
import os

from mitransformer.utils.logmaker import (
    getLogger, info)

logger = getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


def main_hyperopt(
        arguments: "parsing.HyperoptParserArgs",
        world_size: int) -> None:
    if arguments.optimise == "loglik":
        direction = "maximize"
    else:
        direction = (
            "minimize" if minimise[arguments.optimise.lower().split(":")[0]]
            else "maximize")

    study: None | optuna.Study = None
    ld = os.path.join("./runs", f"{arguments.name}_hyperopt")
    with ddp.new_pg(
            world_size, "gloo") as pg, metric_writer(log_dir=ld) as writer:
        objective: hyperoptlib.Objective
        if arguments.optimise == "loglik":
            objective = hyperoptlib.PsyLingObjective(
                world_size, arguments, writer, pg)
        else:
            objective = hyperoptlib.Objective(
                world_size, arguments, writer, pg)

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
