from ...train.metrics import (
    metric_writer, minimise)
from .. import parsing, ddp
from . import hyperoptlib

import pickle
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
        if arguments.psyling_eval or arguments.optimise == "loglik":
            objective = hyperoptlib.PsyLingObjective(
                world_size, arguments, writer, pg)
        else:
            objective = hyperoptlib.Objective(
                world_size, arguments, writer, pg)

        if arguments.rank == 0 or arguments.rank is None:
            sampler: optuna.samplers.BaseSampler
            pruner: optuna.pruners.BasePruner

            db_path = f"./runs/{arguments.name}_hyperopt/db"
            if not os.path.exists(db_path):
                os.makedirs(db_path)

            storage_name = (
                f"sqlite:///{db_path}/{arguments.name}.db")  # ./models/{arguments.name}/db/
            db_file_path = storage_name.replace('sqlite:///', '')
            sampler_path = f'{db_path}/sampler.pkl'
            pruner_path = f'{db_path}/pruner.pkl'

            if os.path.exists(db_file_path):
                # saving/resuming study with RDB backend
                if os.path.exists(sampler_path):
                    if os.path.exists(pruner_path):
                        with open(sampler_path, 'rb') as fin:
                            sampler = pickle.load(fin)
                        with open(pruner_path, 'rb') as fin:
                            pruner = pickle.load(fin)
                        info(
                            arguments.rank, logger,
                            "Resuming existing optuna study...")
                        study = optuna.create_study(
                            study_name=arguments.name,
                            storage=storage_name,
                            direction=direction,
                            sampler=sampler,
                            pruner=pruner,
                            load_if_exists=True)
                    else:
                        raise Exception("pruner.pkl not found.")
                else:
                    raise Exception("sampler.pkl not found.")
            else:
                info(arguments.rank, logger, "Starting new optuna study...")

                match arguments.sampler:
                    case "random":
                        sampler = optuna.samplers.RandomSampler(
                            seed=arguments.seed
                        )
                    case "tpe":
                        sampler = optuna.samplers.TPESampler(
                            n_startup_trials=arguments.sampler_startup_trials,
                            seed=arguments.seed,
                            multivariate=True, group=True
                        )
                    case _:
                        raise Exception(f"Sampler {arguments.sampler} unknown.")

                match arguments.pruner:
                    case "hyperband":
                        pruner = optuna.pruners.HyperbandPruner(
                        )
                    case "median":
                        pruner = optuna.pruners.MedianPruner(
                            n_startup_trials=arguments.pruner_startup_trials,
                            n_warmup_steps=arguments.n_warmup_steps,
                        )
                    case _:
                        raise Exception(f"Pruner {arguments.pruner} unknown.")

                study = optuna.create_study(
                    study_name=arguments.name,
                    storage=storage_name,
                    direction=direction,
                    sampler=sampler,
                    pruner=pruner)

            for _ in range(arguments.n_trials):
                study.optimize(
                    objective, n_trials=1)
                with open(sampler_path, 'wb') as fout:
                    pickle.dump(study.sampler, fout)
                with open(pruner_path, 'wb') as fout:
                    pickle.dump(study.pruner, fout)

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
