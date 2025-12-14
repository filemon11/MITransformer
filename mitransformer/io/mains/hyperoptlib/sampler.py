from ... import parsing
import numpy as np

import optuna

from mitransformer.utils.logmaker import (
    getLogger)

from typing import cast, Sequence

logger = getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


USE_LOG = {"learning_rate"}


def hyperopt_arguments_sampler(
        name: str,
        arg: parsing.T | parsing.Choices[parsing.T] | parsing.Range | dict[
            str, parsing.T | parsing.Choices[parsing.T] | parsing.Range],
        trial: optuna.Trial
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
            arg = trial.suggest_float(  # type: ignore
                name, arg[0], arg[1],
                log=name in USE_LOG)
        else:
            arg = trial.suggest_int(  # type: ignore
                name, int(arg[0]), int(arg[1]),
                log=name in USE_LOG)

    if isinstance(arg, dict):
        # If values are all None, sample from Dirichlet
        if all([value is None for value in arg.values()]):
            arg = dirichlet(name, arg.keys(), trial)  # type: ignore
        else:
            arg = {subn: hyperopt_arguments_sampler(  # type: ignore
                f"{name}_{subn}", subv, trial)
                for subn, subv in arg.items()}
    else:
        arg = cast(parsing.T, arg)

    return arg  # type: ignore


def dirichlet(
        name: str,
        subnames: Sequence[str],
        trial: optuna.Trial
        ) -> dict[str, float]:
    x: list[float] = []
    for sub in subnames:
        x.append(-np.log(
            trial.suggest_float(f"x_{name}_{sub}", 0, 1)))

    p: dict[str, float] = {}
    for i, sub in enumerate(subnames):
        p[sub] = (x[i] / sum(x))
        trial.set_user_attr(f"{name}_{sub}", p[sub])

    return p
