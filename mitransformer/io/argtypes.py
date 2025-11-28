from typing import (
    Any, cast, TypeVar, Generic,
    Sequence, Callable)


T = TypeVar("T")


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
        "be parsed as a string!"))


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

    def __call__(self, value: str) -> tuple[T, T] | list[T] | T:
        # try to split via :
        split = value.split(":")
        if len(split) == 2:
            try:
                h_range = (
                    self.type(split[0]),
                    self.type(split[1]))  # type: ignore
                assert (isinstance(h_range[0], (int, float, complex))
                        and not isinstance(h_range[0], bool)), (
                        f"Non-numeric range detected: {h_range}")
                return h_range
            except TypeError:
                raise Exception(
                    f"Constructor for {self.type} does not accept an argument")

        assert len(split) < 2, (
            f"Range must have one starting and one end point. Given: {value}")

        split = value.split(";")
        if len(split) == 1:
            try:
                return self.type(value)  # type: ignore
            except TypeError:
                raise Exception(
                    f"Constructor for {self.type} does not accept an argument")

        try:
            options = [self.type(v) for v in split]  # type: ignore
            if self.choices is not None:
                for o in options:
                    assert o in self.choices, (
                        f"{o} must be one of {self.choices}")
            return options
        except TypeError:
            raise Exception(
                f"Constructor for {self.type} does not accept an argument")
