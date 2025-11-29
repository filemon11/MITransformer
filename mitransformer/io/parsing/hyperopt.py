from typing import TypeVar, Sequence

T = TypeVar("T")


class Range(tuple[int | float, int | float]):
    def __new__(cls, x: int | float, y: int | float):
        assert type(x) is type(y), (
            f"Range with values {x} and {y} inconsistently typed!")
        return tuple.__new__(Range, (x, y))

    @property
    def is_continuous(self) -> bool:
        if isinstance(self[0], int):
            return False
        return True


class Choices(list[T]):
    def __init__(self, xs: Sequence[T]):
        super().__init__(xs)
        assert len(xs) > 0, "Specified empty choices parameter."
