import pandas as pd

from ... import lingutils

from abc import ABC, abstractmethod

from typing import Type, Any


# Classes
# # Unsplit untokenised

class UnsplitUntokMetricMaker(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(
            self, unsplit_untok_df: pd.DataFrame,
            *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        ...


class UnsplitUntokWordMetricMaker(UnsplitUntokMetricMaker):
    def __init__(self, word_col: str, *args, **kwargs):
        self.word_col = word_col


class UnsplitUntokFrequency(UnsplitUntokWordMetricMaker):
    def __call__(
            self, unsplit_untok_df: pd.DataFrame,
            *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        return unsplit_untok_df.apply(
            lambda r: lingutils.get_frequency(
                r[self.word_col]), axis=1), dict()


class UnsplitUntokLength(UnsplitUntokWordMetricMaker):
    def __call__(
            self, unsplit_untok_df: pd.DataFrame,
            *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        return unsplit_untok_df.apply(
            lambda r: len(r[self.word_col]), axis=1), dict()


class UnsplitUntokChunkSentence(UnsplitUntokMetricMaker):
    def __init__(self, item_col: str, position_col: str, *args, **kwargs):
        self.item_col = item_col
        self.position_col = position_col

    def __call__(
            self, unsplit_untok_df: pd.DataFrame,
            *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        counter: int = 1
        positions = unsplit_untok_df[self.position_col]
        items = unsplit_untok_df[self.item_col]

        prev_item: str | None = None
        prev_position: int | None = None

        sentence_nums: list[int] = list()        # sentence num inside item
        for item, position in zip(items, positions):
            if item != prev_item:
                counter = 1
            elif position <= prev_position:
                counter += 1

            sentence_nums.append(counter)

            prev_item = item
            prev_position = position

        return pd.Series(sentence_nums), dict()


class UnsplitUntokGlobalSentence(UnsplitUntokMetricMaker):
    def __init__(self, position_col: str, *args, **kwargs):
        self.position_col = position_col

    def __call__(
            self, unsplit_untok_df: pd.DataFrame,
            *args, **kwargs
            ) -> tuple[pd.Series, dict[str, Any]]:
        counter: int = 1
        positions = unsplit_untok_df[self.position_col]

        prev_position: int | None = None

        sentence_nums: list[int] = list()        # sentence num inside item
        for position in positions:
            if prev_position is not None and position <= prev_position:
                counter += 1

            sentence_nums.append(counter)

            prev_position = position

        return pd.Series(sentence_nums), dict()


gen_without_tok: dict[
    str, Type[UnsplitUntokMetricMaker]] = {
        "frequency": UnsplitUntokFrequency,
        "length": UnsplitUntokLength,
        "chunksentence": UnsplitUntokChunkSentence,
        "globalsentence": UnsplitUntokGlobalSentence,
}
