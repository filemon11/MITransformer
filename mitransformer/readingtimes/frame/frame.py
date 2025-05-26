import pandas as pd

from .raw.metrics import gen_without_tok
from .split.metrics import gen_and_untok
from ..lingutils import (
    UntokSplitFunc, untokenise, untokenise_split_df)

from typing import (
    Sequence, Any, Iterable, TypeVar, Collection,
    Self)


T = TypeVar("T")
F = TypeVar("F", bound="Frame")


def unsplit_add_column_(
        df: pd.DataFrame, colname: str,
        content: Sequence[Any] | pd.Series) -> None:

    # Assert that number of rows matches
    assert len(df) == len(content), (
        f"Number of dataframe rows ({len(df)}) and "
        f"number of new content elements ({len(content)}) do"
        "not match.")
    df[colname] = content


class Frame():
    def __init__(
            self, df: pd.DataFrame = pd.DataFrame(),
            colnames: dict[str, str] = dict(),
            tokenised: bool = False,
            untok_funcs: list[UntokSplitFunc] = [],
            additional: dict[str, Any] = {}):
        self.df = df
        self.colnames = colnames
        self.tokenised = tokenised
        self.untok_funcs = untok_funcs
        self.additional = additional

    def add_override_additional(self, **kwargs) -> None:
        for key, item in kwargs.items():
            self.additional[key] = item

    def copy(self) -> Self:
        return self.__class__(
            df=self.df.copy(), colnames=self.colnames.copy(),
            tokenised=self.tokenised,
            untok_funcs=self.untok_funcs.copy(),
            additional=self.additional.copy()
        )

    def clone_settings(self, old_colname: str, new_colname: str) -> None:
        # TODO: untok_funcs etc.
        pass

    @classmethod
    def conservative_add_col(
            cls,
            df: pd.DataFrame, colname: str, column: pd.Series) -> str:
        name = colname
        if colname in df.columns:
            numbers_list = colname.split(".")
            try:
                if len(numbers_list) > 0:
                    num = int(numbers_list[-1])
                    name = ".".join(numbers_list[:-1] + [str(num+1)])
                    Frame.conservative_add_col(
                        df, name, column)
                else:
                    raise ValueError()
            except ValueError:
                name = f"{colname}.1"
                Frame.conservative_add_col(
                    df, name, column)
        else:
            df[name] = column
        return name

    @classmethod
    def merge_dataframes(
            cls,
            df1: pd.DataFrame, df2: pd.DataFrame
            ) -> tuple[pd.DataFrame, dict[str, str]]:
        new_df = df1.copy()
        clone_dict: dict[str, str] = {}
        for col in df2.columns:
            clone_dict[col] = cls.conservative_add_col(new_df, col, df2[col])
        return new_df, clone_dict

    @classmethod
    def merge_frames(
            cls,
            new_df: pd.DataFrame, frame1: Self, frame2: Self) -> Self:
        new_colnames = frame1.colnames | frame2.colnames
        assert frame1.tokenised == frame2.tokenised
        new_untok_funcs = frame1.untok_funcs + frame2.untok_funcs
        new_additional = frame1.additional | frame2.additional
        return frame1.__class__(
            df=new_df, colnames=new_colnames, tokenised=frame1.tokenised,
            untok_funcs=new_untok_funcs, additional=new_additional
        )

    def __or__(self, other: Self) -> Self:
        new_df, clone_dict = self.merge_dataframes(self.df, other.df)
        new_frame = self.merge_frames(new_df, self, other)
        for old, new in clone_dict.items():
            new_frame.clone_settings(old, new)
        return new_frame


class UnsplitFrame(Frame):
    generators = gen_without_tok

    def copy(self) -> Self:
        new_frame = super().copy()
        new_frame.generators = self.generators.copy()
        return new_frame

    def add_column_(
            self,
            colkey: str,
            colname: str,
            content: Sequence[Any] | pd.Series,
            additional: dict[str, Any]) -> None:
        unsplit_add_column_(self.df, colname, content)
        self.colnames[colkey] = colname
        self.add_override_additional(**additional)

    def add_(self, coltype: str, alt_name: str | None = None,
             *args, **kwargs) -> None:
        self.add_override_additional(**kwargs)

        colname = alt_name if alt_name is not None else coltype
        colkey = f"{colname}_col"

        self.add_column_(
            colkey, colname,
            *self.generators[coltype](**self.colnames)(
                self.df, *args, **self.additional))

    def split(self, lengths: Iterable[int]) -> "SplitFrame":
        ends = [item for le in lengths for item in [False]*(le-1)+[True]]
        new_df = split_df(self.df, ends)
        return SplitFrame(
            new_df, self.colnames,
            dict(),
            dict(),
            self.tokenised,
            self.untok_funcs,
            self.additional)

    def __or__(self, other: Self) -> Self:
        new_frame = super().__or__(other)
        new_frame.generators = self.generators | other.generators
        return new_frame


def shift_(
        list_frame: pd.DataFrame, amount: int,
        cols_to_shift: Collection[str],
        cols_to_ignore: Collection[str]) -> None:

    for colname in list_frame.columns:
        if colname not in cols_to_ignore:
            if colname in cols_to_shift:
                sign = -1
            else:
                sign = +1
            obj = []
            for sentence in list_frame[colname]:
                try:
                    obj.append(sentence[
                        max(0, sign*amount):min(
                            len(sentence),
                            len(sentence)+(sign*amount))])
                except TypeError:
                    raise Exception(
                        f"Does sentence ({sentence}) have the right format?"
                        " Likey there is a misalignment, "
                        "maybe caused by spacy?"
                        " This can happen when two punctuation "
                        "symbols follow a"
                        " word. Suggestion: Remove one.")
            list_frame[colname] = obj


def split_to_sentence_list(
        items: Iterable[T], ends: Iterable[bool]) -> Iterable[list[T]]:
    current_sen: list[T] = []
    for item, end in zip(items, ends):
        current_sen.append(item)
        if end:
            yield current_sen
            current_sen = []


def split_df(df: pd.DataFrame, ends: Iterable[bool]) -> pd.DataFrame:
    column_names = df.columns
    list_frame = pd.DataFrame({
        column_names[0]: split_to_sentence_list(
            df[column_names[0]], ends
        )})
    for colname in column_names[1:]:
        content = split_to_sentence_list(
            df[colname], ends)
        add_column_(list_frame, colname, list(content))
    return list_frame


def add_column_(
        list_frame: pd.DataFrame, colname: str,
        content: Sequence[Sequence[Any]] | pd.Series,
        ) -> None:

    if len(list_frame) > 0:
        # Assert that number of rows matches
        assert len(list_frame) == len(content), (
            f"Number of dataframe rows ({len(list_frame)}) and "
            f"number of new content elements ({len(content)}) do"
            "not match.")

        # # Assert that number elements in each sentence matches
        # series = list_frame.iloc[:, -1]
        # for i, (series_l, content_l) in enumerate(zip(series, content)):
        #     assert len(series_l) == len(content_l), (
        #         f"Number of items is not equal at position {i}"
        #         f"({len(series_l)} vs {len(content_l)})")

    list_frame[colname] = content


def generate_sentence_end_list(
        tok_sentences: Iterable[Sequence[Any]],
        untok_space_after: Iterable[bool]
        ) -> Iterable[bool]:
    lengths = [len(sen) for sen in tok_sentences]
    tok_end_list = [end for s_length in lengths
                    for end in [False]*(s_length-1)+[True]]

    return untokenise(tok_end_list, untok_space_after, "last")


def unsplit_df(
        df: pd.DataFrame, not_to_unsplit: Collection[str]
        ) -> pd.DataFrame:
    unsplit_df = pd.DataFrame()
    for colname in df.columns:
        if colname in not_to_unsplit:
            continue
        try:
            values = [item for sen in df[colname] for item in sen]
            unsplit_df[colname] = values
        except ValueError:
            print(
                "Warning: "
                f"Length of values ({len(values)}) does not match "
                f"length of index ({len(unsplit_df)}) for "
                f"column {colname}. Omitted this column.")
    return unsplit_df


# TODO change structure wrt. to the sets (shift, to_unsplit, etc.)
# new: maintain two dicts:
# 1. <type>_col: <colname>
# 2. <colname>: (<type>, shift, to_unsplit, untok_func,...)
class SplitFrame(Frame):
    generators = gen_and_untok

    def __init__(
            self, df: pd.DataFrame = pd.DataFrame(),
            colnames: dict[str, str] = dict(),
            shift: dict[str, bool | None] = dict(),
            to_unsplit: dict[str, bool] = dict(),
            tokenised: bool = False,
            untok_funcs: list[UntokSplitFunc] = [],
            additional: dict[str, Any] = {},
            trunc_left: int = 0,
            trunc_right: int = 0):
        super().__init__(
            df=df, colnames=colnames, tokenised=tokenised,
            untok_funcs=untok_funcs, additional=additional)
        self.shift: dict[str, bool | None] = shift
        self.to_unsplit = to_unsplit
        self.trunc_left: int = trunc_left
        self.trunc_right: int = trunc_right

    def copy(self) -> Self:
        new_frame = super().copy()
        new_frame.generators = self.generators.copy()
        new_frame.shift = self.shift.copy()
        new_frame.to_unsplit = self.to_unsplit.copy()
        new_frame.trunc_left = self.trunc_left
        new_frame.trunc_right = self.trunc_right
        return new_frame

    def clone_settings(self, old_colname: str, new_colname: str) -> None:
        # TODO: untok_funcs etc.
        super().clone_settings(old_colname, new_colname)
        if old_colname in self.shift.keys():
            self.shift[new_colname] = self.shift[old_colname]
        if old_colname in self.to_unsplit.keys():
            self.to_unsplit[new_colname] = self.to_unsplit[old_colname]

    def add_column_(
            self,
            colkey: str,
            colname: str,
            content: Sequence[Sequence[Any]] | pd.Series,
            additional: dict[str, Any],
            shift: bool | None,
            to_unsplit: bool,
            untokenise_func: None | UntokSplitFunc = None,
            ) -> None:
        add_column_(
            self.df, colname, content)
        self.colnames[colkey] = colname
        self.add_override_additional(**additional)

        if untokenise_func is not None:
            self.untok_funcs.append(untokenise_func)

        self.shift[colname] = shift
        self.to_unsplit[colname] = to_unsplit

    def add_(self, coltype: str, alt_name: str | None = None,
             *args, **kwargs) -> None:
        self.add_override_additional(**kwargs)

        colname = alt_name if alt_name is not None else coltype
        colkey = f"{colname}_col"

        untok_type = self.generators[coltype][2]
        untok: None | UntokSplitFunc
        if untok_type is not None:
            untok = untok_type(
                coltype, **(self.colnames | {colkey: colname}))
        else:
            untok = None

        self.add_column_(
            colkey, colname,
            *self.generators[coltype][0](**self.colnames)(
                self.df, *args, **self.additional),
            self.generators[coltype][1],
            self.generators[coltype][3],
            untok)

    def shift_(
            self,
            amount: int) -> None:
        assert not self.tokenised, "Cannot shift tokenised dataframe."
        if amount == 0:
            return
        names_to_shift = {
            colname for colname, shift in self.shift.items() if shift}
        cols_to_ignore = {
            colname for colname, shift in self.shift.items() if shift is None}
        shift_(self.df, amount, names_to_shift, cols_to_ignore)
        self.trunc_left += max(0, amount)
        self.trunc_right += max(0, -amount)

    def untokenise_(self) -> None:
        assert self.tokenised
        self.df = untokenise_split_df(
            self.df,
            self.untok_funcs
            )
        self.tokenised = False

    def unsplit(self) -> UnsplitFrame:
        names_to_not_unsplit = {
            colname for colname, unsplit in self.to_unsplit.items()
            if not unsplit}
        df = unsplit_df(self.df, names_to_not_unsplit)

        return UnsplitFrame(
            df, self.colnames, self.tokenised, self.untok_funcs,
            self.additional)

    def __or__(self, other: Self) -> Self:
        trunc_diff_left = other.trunc_left - self.trunc_left
        trunc_diff_right = other.trunc_right - self.trunc_right

        trunc_self_left = max(0, trunc_diff_left)
        trunc_self_right = max(0, trunc_diff_right)

        trunc_other_left = max(0, -trunc_diff_left)
        trunc_other_right = max(0, -trunc_diff_right)

        df1 = self.df.copy()
        df2 = other.df.copy()

        cols_to_ignore1 = {
            colname for colname, shift in self.shift.items()
            if shift is None}

        cols_to_ignore2 = {
            colname for colname, shift in other.shift.items()
            if shift is None}

        self.truncate_df_(
            df1, cols_to_ignore1, trunc_self_left, trunc_self_right)
        self.truncate_df_(
            df2, cols_to_ignore2, trunc_other_left, trunc_other_right)

        new_df, clone_dict = self.merge_dataframes(
            df1, df2)

        new_frame = self.merge_frames(new_df, self, other)

        for old, new in clone_dict.items():
            new_frame.clone_settings(old, new)

        new_frame.generators = self.generators | other.generators
        new_frame.shift = self.shift | other.shift
        new_frame.to_unsplit = self.to_unsplit | other.to_unsplit
        new_frame.trunc_left = max(self.trunc_left, other.trunc_left)
        new_frame.trunc_right = max(self.trunc_right, other.trunc_right)

        for old, new in clone_dict.items():
            new_frame.clone_settings(old, new)
        return new_frame

    @staticmethod
    def truncate_df_(
            df: pd.DataFrame,
            cols_to_ignore: Collection[str],
            left: int, right: int) -> None:
        for colname in df.columns:
            if colname not in cols_to_ignore:
                new_series = []
                for sentence in df[colname]:
                    length = len(sentence)
                    new_series.append(
                        sentence[left:length-right]
                    )
                df[colname] = new_series
