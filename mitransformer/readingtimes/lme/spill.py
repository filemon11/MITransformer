from typing import Iterable

# ------------------------------------------------------------------
# Spillover helpers
# ------------------------------------------------------------------


def get_spillover(predictors: Iterable[str], spillover: int) -> list[str]:
    return [f"{p}.{spillover}" for p in predictors]


def get_spillover_upto(predictors: Iterable[str], spillover: int) -> list[str]:
    out = list(predictors)
    if spillover > 0:
        for i in range(1, spillover + 1):
            out += get_spillover(predictors, i)
    return out
