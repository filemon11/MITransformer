
import subprocess

from . import evaluation

from typing import overload

ANALYSIS_SCRIPT = "RT/analysis_new.R"


@overload
def lme_legacy(
        model_name: str,
        dataset_name: str,
        additional_name: str,
        n_runs: int = 1,
        shift: int = 0,
        log_at: None = None) -> str:
    ...


@overload
def lme_legacy(
        model_name: str,
        dataset_name: str,
        additional_name: str,
        n_runs: int,
        shift: int,
        log_at: str) -> None:
    ...


def lme_legacy(
        model_name: str,
        dataset_name: str,
        additional_name: str,
        n_runs: int = 1,
        shift: int = 0,
        log_at: str | None = None) -> None | str:

    arguments: list[str] = [
            "Rscript", "--vanilla", ANALYSIS_SCRIPT,
            f"{model_name}",
            f"{n_runs}",
            f"{dataset_name}",
            f"{shift}",
            "RT/data",
            f"{additional_name}"]
    if log_at is None:
        s = subprocess.run(
            arguments, encoding='utf-8', stdout=subprocess.PIPE).stdout
        return s
    else:
        with open(log_at, 'w') as f:
            subprocess.run(arguments, stdout=f)
        return None


def lme(
        model_name: str,
        dataset_name: str,
        additional_name: str,
        n_runs: int = 1,
        shift: int = 0) -> None | str:
    evaluation.run(
        model_name, n_runs, dataset_name, shift, "RT/data", additional_name)
    return None
