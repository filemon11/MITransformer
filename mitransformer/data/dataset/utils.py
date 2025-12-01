from collections import defaultdict

from typing import (Iterable, Sequence,
                    TypeVar, Callable, Hashable, Literal)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)


X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z", bound=Hashable)


MasksSetting = Literal["complete", "both", "next", "current"]


def listmap(func: Callable[[X], Y], seq: Iterable[X]) -> list[Y]:
    return list(map(func, seq))


def filldict(
        keys: Sequence[Z],
        funcs: Sequence[Callable[[X], Y]],
        seq: Iterable[X]) -> dict[Z, list[Y]]:

    out_dict: dict[Z, list[Y]] = defaultdict(list)

    for entry in seq:
        for key, func in zip(keys, funcs):
            out_dict[key].append(func(entry))

    return out_dict
