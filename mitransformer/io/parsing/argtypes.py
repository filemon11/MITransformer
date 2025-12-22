from itertools import cycle

from . import hyperopt

from types import EllipsisType
from typing import (
    Any, cast, TypeVar, Generic,
    Sequence, Callable, Iterable)


T = TypeVar("T")
S = TypeVar("S")


class StrToLiteral(Generic[T]):
    def __init__(self, *selection: T):
        self.selection: tuple[T, ...] = selection

    def __call__(self, string: str) -> T:
        if string in self.selection:
            return cast(T, string)
        else:
            raise Exception(
                f"Argument value not allowed."
                f"Given: {string}, allowed: {self.selection}")


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
    raise Exception((
        f"argument value {string} cannot"
        " be parsed as a string!"))


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
    def __init__(
            self, type: Callable[[Any], T],
            choices: Sequence[T] | None = None):
        self.type = type
        self.choices = choices

    def __call__(
            self, value: str
            ) -> hyperopt.Range | hyperopt.Choices[T] | T:
        # try to split via :
        split = split_nested(value, ";")
        if len(split) == 2:
            try:
                h_range = (
                    self.type(split[0]),
                    self.type(split[1]))  # type: ignore
            except TypeError:
                raise Exception(
                    f"Constructor for {self.type} does not accept an argument")
            assert (isinstance(h_range[0], (int, float, complex))
                    and not isinstance(h_range[0], bool)), (
                    f"Non-numeric range detected: {h_range}")
            assert (isinstance(h_range[1], (int, float, complex))
                    and not isinstance(h_range[1], bool)), (
                    f"Non-numeric range detected: {h_range}")
            return hyperopt.Range(*h_range)  # type: ignore

        assert len(split) < 2, (
            f"Range must have one starting and one end point. Given: {value}")

        split = split_nested(value, "|")
        if len(split) == 1:
            try:
                return self.type(value)  # type: ignore
            except TypeError:
                raise Exception(
                    f"Constructor for {self.type} does not accept an argument")

        try:
            options = [self.type(v) for v in split]  # type: ignore
        except TypeError:
            raise Exception(
                f"Constructor for {self.type} does not accept an argument")
        if self.choices is not None:
            for o in options:
                assert o in self.choices, (
                    f"{o} must be one of {self.choices}")
        return hyperopt.Choices(options)


class StrToTuple(Generic[T]):
    def __init__(
            self,
            type1: Callable[[Any], T],
            type2: Callable[[Any], T] | EllipsisType | None = None,
            *types: Callable[[Any], T]):
        self.types: Iterable[Callable[[Any], T]]
        if type2 is not None:
            if type2 == Ellipsis:
                self.types = cycle([type1])
            else:
                self.types = [type1, type2, *types]  # type: ignore
        else:
            self.types = [type1]

    def __call__(self, string: str) -> tuple[T, ...]:
        string = string.strip()
        if string[0] == "(" and string[-1] == ")":
            string = string[1:-1]
        components: Iterable[str] = split_nested(string)
        components = [c.strip() for c in components]

        out_list = []
        for c, t in zip(components, self.types):
            out_list.append(t(c))
        return tuple(out_list)


def split_nested(
        string: str, split_at: str = ",",
        l_brackets: str = "({[",
        r_brackets: str = ")}]"
        ) -> tuple[str, ...]:
    """ATTENTION: do not cross brackets"""
    assert len(split_at) == 1

    components: list[str] = []
    level = 0
    c_incomplete = ""
    for c in string:
        if c == split_at and level == 0:
            components.append(c_incomplete)
            c_incomplete = ""
            continue
        elif c in l_brackets:
            level += 1
        elif c in r_brackets:
            level -= 1
        c_incomplete += c
    if len(c_incomplete) > 0:
        components.append(c_incomplete)

    return tuple(components)


class StrToDict(Generic[S, T]):
    def __init__(
            self,
            key_type: Callable[[Any], S],
            value_type: Callable[[Any], T],):
        self.key_type = key_type
        self.value_type = value_type

    def __call__(self, string: str) -> dict[S, T]:
        string = string.strip()
        assert string[0] == "{" and string[-1] == "}"
        string = string[1:-1]
        components: Iterable[str] = split_nested(string)
        components = [c.strip() for c in components]
        # components are of form key:value

        out_dict: dict[S, T] = {}
        for c in components:
            key, value = c.split(":")
            key = key.strip()
            value = value.strip()
            out_dict[self.key_type(key)] = self.value_type(value)

        return out_dict
