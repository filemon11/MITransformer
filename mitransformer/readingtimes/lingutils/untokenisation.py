from .linguistics import TAGSET, PUNCTUATION

from itertools import cycle
import pandas as pd

from abc import ABC, abstractmethod

from typing import (
    Iterable, Literal, Callable,
    overload, Protocol, runtime_checkable, Sequence, Optional,
    NamedTuple, Any)


@runtime_checkable
class Addable(Protocol):
    def __add__(self, other: "Addable") -> "Addable": ...


@runtime_checkable
class Multipliable(Protocol):
    def __mul__(self, other: "Multipliable") -> "Multipliable": ...


# --- NamedTuples ---

class HeadInfo(NamedTuple):
    pos_tag: str
    head: int


class PosInfo(NamedTuple):
    pos_tag: str


# --- Typing helpers ---

T = Any
Memory = Optional[Any]

FullFunc = Callable[
    [tuple[T, Any, Memory, Memory, int], tuple[T, Any]],
    tuple[T, Any, Memory, Memory],
]

SimpleFunc = Callable[[T, T], T]


def promote_func(func: SimpleFunc) -> FullFunc:
    """Wrap a simple (x, y) -> z function into full
    (x, extra, short, long), (y, extra) -> (z, None, None, None)."""
    def wrapper(
        current: tuple[T, Any, Memory, Memory, int],
        new: tuple[T, Any],
    ) -> tuple[T, Any, Memory, Memory]:
        return func(current[0], new[0]), None, None, None
    return wrapper


# --- Core helper ---

def _untokense(
    items: Iterable[T],
    space_after: Iterable[bool],
    func: FullFunc,
    additional: Optional[Iterable[Any]] = None,
) -> Iterable[T]:
    extra = additional if additional is not None else cycle([None])

    current_item = current_extra = short_memory = long_memory = None

    for i, (item, space, extra_item) in enumerate(
            zip(items, space_after, extra)):
        if current_item is None:
            current_item, current_extra = item, extra_item
        else:
            current_item, current_extra, short_memory, long_memory = func(
                (current_item, current_extra, short_memory, long_memory, i),
                (item, extra_item),
            )
        if space:
            yield current_item
            current_item = current_extra = short_memory = long_memory = None

    if current_item is not None:
        yield current_item


# --- Overloads ---

@overload
def untokenise(
    items: Iterable[Addable],
    space_after: Iterable[bool],
    mode: Literal["add"],
    additional: None = None,
    punctuation: set[str] = ...,
) -> Iterable[Addable]: ...


@overload
def untokenise(
    items: Iterable[Multipliable],
    space_after: Iterable[bool],
    mode: Literal["mult"] = "mult",
    additional: None = None,
    punctuation: set[str] = ...,
) -> Iterable[Multipliable]: ...


@overload
def untokenise(
    items: Iterable[T],
    space_after: Iterable[bool],
    mode: Literal["first", "last"],
    additional: None = None,
    punctuation: set[str] = ...,
) -> Iterable[T]: ...


@overload
def untokenise(
    items: Iterable[T],
    space_after: Iterable[bool],
    mode: Literal["pos"],
    additional: Optional[Iterable[PosInfo]] = None,
    punctuation: set[str] = ...,
) -> Iterable[T]: ...


@overload
def untokenise(
    items: Iterable[T],
    space_after: Iterable[bool],
    mode: Literal["head"],
    additional: Optional[Iterable[HeadInfo]] = None,
    punctuation: set[str] = ...,
) -> Iterable[T]: ...


# --- Real Implementation ---

def untokenise(
    items: Iterable[T],
    space_after: Iterable[bool],
    mode: Literal["mult", "add", "last", "first", "pos", "head"] = "mult",
    additional: Optional[Iterable[Any]] = None,
    punctuation: set[str] = set(),
) -> Iterable[T]:

    # -- Simple (x, y) -> merged

    def add_simple(x: Addable, y: Addable) -> Addable:
        return x + y

    def mult_simple(x: Multipliable, y: Multipliable) -> Multipliable:
        return x * y

    def first_simple(x: T, y: T) -> T:
        return x

    def last_simple(x: T, y: T) -> T:
        return y

    # -- Complex (x, extra, short, long) -> merged

    def pos_full(
        current: tuple[T, PosInfo, Memory, Memory],
        new: tuple[T, PosInfo],
    ) -> tuple[T, PosInfo, Any, Any]:
        curr_item, curr_info, *_ = current
        new_item, new_info = new
        if (curr_info.pos_tag in punctuation
                and new_info.pos_tag not in punctuation):
            return new_item, new_info, None, None
        return curr_item, curr_info, None, None

    # TODO: fix headlist alignment
    def head_full(
        current: tuple[T, HeadInfo, Memory, Memory, int],
        new: tuple[T, HeadInfo],
    ) -> tuple[T, HeadInfo, Any, Any]:
        curr_item, curr_info, history, _, idx = current
        new_item, new_info = new

        num_in = 1 if history is None else history[0] + 1

        if history is None:
            left_elem = right_elem = (curr_item, curr_info)
        else:
            left_elem, right_elem = (
                history[3], history[4])

        left_border, right_border = idx - num_in, idx

        head = curr_info.head - 2
        if curr_info.head < 0:
            head = idx

        left_dist = max(0, idx-head) if history is None else history[1]
        right_dist = max(
            0, head-idx) if history is None else max(0, history[2] - 1)

        if (head < left_border
                and (left_border - head > left_dist)
                and not (
                    left_elem[1].pos_tag not in punctuation
                    and curr_info.pos_tag in punctuation)):
            left_dist, left_elem = left_border - head, (
                new_item, new_info)
        elif (head > right_border
                and (head - right_border > right_dist)
                and not (
                    right_elem[1].pos_tag not in punctuation
                    and curr_info.pos_tag in punctuation)):
            right_dist, right_elem = head - right_border, (
                new_item, new_info)

        selected = left_elem if left_dist > right_dist else right_elem
        return selected[0], selected[1], (
            num_in, left_dist, right_dist, left_elem, right_elem), None

    match mode:
        case "add":
            func: FullFunc = promote_func(add_simple)
        case "mult":
            func = promote_func(mult_simple)
        case "last":
            func = promote_func(last_simple)
        case "first":
            func = promote_func(first_simple)
        case "pos":
            func = pos_full  # type: ignore
        case "head":
            func = head_full  # type: ignore
        case _:
            raise ValueError(f"Unsupported mode: {mode}")

    yield from _untokense(items, space_after, func, additional)


def untokenise_by_POS(
        items: Iterable[T],
        space_after: Iterable[bool],
        pos_tags: Iterable[str],
        punctuation: set[str] = PUNCTUATION[TAGSET]
        ) -> Iterable[T]:
    additional = (PosInfo(p) for p in pos_tags)
    return untokenise(
        mode="pos", space_after=space_after,
        items=items,
        punctuation=punctuation, additional=additional)


def untokenise_by_head(
        items: Iterable[T],
        space_after: Iterable[bool],
        heads: Iterable[int],
        pos_tags: Iterable[str],
        punctuation: set[str] = PUNCTUATION[TAGSET]
        ) -> Iterable[T]:
    additional = (HeadInfo(p, h) for p, h in zip(pos_tags, heads))
    return untokenise(
        mode="head", space_after=space_after,
        items=items,
        punctuation=punctuation, additional=additional)


def untokenise_split_df_primitive(
        df: pd.DataFrame,
        untok_funcs: dict[str, "UntokSplitFunc"]
        ) -> pd.DataFrame:
    untok_df = df.copy()
    for col, func in untok_funcs.items():
        untok_df[col] = func(df)
    return untok_df


# TODO: extend to unsplit

class UntokSplitFunc(ABC):
    def __init__(self, col: str, space_after_col: str, *args, **kwargs):
        self.col = col
        self.space_after_col = space_after_col

    @abstractmethod
    def __call__(self, untok_df: pd.DataFrame) -> Sequence:
        ...


class UntokSplitPOS(UntokSplitFunc):
    def __init__(
            self, col: str, space_after_col: str, pos_col: str,
            punctuation: set[str] = PUNCTUATION[TAGSET], *args, **kwargs):
        super().__init__(col, space_after_col)

        self.pos_col = pos_col
        self.punctuation = punctuation

    def __call__(self, untok_df: pd.DataFrame) -> Sequence:
        it = [list(untokenise_by_POS(
            untok_df[self.col][i],
            untok_df[self.space_after_col][i],
            untok_df[self.pos_col][i],
            self.punctuation
            )) for i in range(len(untok_df))]
        return it


class UntokSplitHead(UntokSplitFunc):
    def __init__(
            self, col: str, space_after_col: str, pos_col: str,
            head_col: str,
            punctuation: set[str] = PUNCTUATION[TAGSET], *args, **kwargs):
        super().__init__(col, space_after_col)

        self.pos_col = pos_col
        self.head_col = head_col
        self.punctuation = punctuation

    def __call__(self, untok_df: pd.DataFrame) -> Sequence:
        it = [list(untokenise_by_head(
            untok_df[self.col][i],
            untok_df[self.space_after_col][i],
            untok_df[self.head_col][i],
            untok_df[self.pos_col][i],
            self.punctuation
            ))
                for i in range(len(untok_df))]

        return it


class UntokSplitAdd(UntokSplitFunc):
    def __init__(
            self, col: str, space_after_col: str,
            *args, **kwargs):
        super().__init__(col, space_after_col)

    def __call__(self, untok_df: pd.DataFrame) -> Sequence:
        it = [list(untokenise(
            untok_df[self.col][i],
            untok_df[self.space_after_col][i], "add"))
                for i in range(len(untok_df))]
        return it


class UntokSplitLast(UntokSplitFunc):
    def __init__(
            self, col: str, space_after_col: str,
            *args, **kwargs):
        super().__init__(col, space_after_col)

    def __call__(self, untok_df: pd.DataFrame) -> Sequence:
        it = [list(untokenise(
            untok_df[self.col][i],
            untok_df[self.space_after_col][i], "last"))
                for i in range(len(untok_df))]
        return it


def untokenise_split_df(
        df: pd.DataFrame,
        untok_funcs: Sequence[UntokSplitFunc]
        ) -> pd.DataFrame:
    return untokenise_split_df_primitive(
        df, {f.col: f for f in untok_funcs})
