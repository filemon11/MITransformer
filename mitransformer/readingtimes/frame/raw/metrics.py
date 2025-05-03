import pandas as pd
import wordfreq

from abc import ABC, abstractmethod

from typing import Type, Any


LANG = "en"

# Methods


def get_frequency(token: str, language: str = LANG):
    return wordfreq.zipf_frequency(token, language)


# Classes
# # Unsplit untokenised

class UnsplitUntokMetricMaker(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(
            self, unsplit_untok_df: pd.DataFrame
            ) -> tuple[pd.Series, dict[str, Any]]:
        ...


class UnsplitUntokWordMetricMaker(UnsplitUntokMetricMaker):
    def __init__(self, word_col: str, *args, **kwargs):
        self.word_col = word_col


class UnsplitUntokFrequency(UnsplitUntokWordMetricMaker):
    def __call__(
            self, unsplit_untok_df: pd.DataFrame
            ) -> tuple[pd.Series, dict[str, Any]]:
        return unsplit_untok_df.apply(
            lambda r: get_frequency(r[self.word_col]), axis=1), dict()


class UnsplitUntokLength(UnsplitUntokWordMetricMaker):
    def __call__(
            self, unsplit_untok_df: pd.DataFrame
            ) -> tuple[pd.Series, dict[str, Any]]:
        return unsplit_untok_df.apply(
            lambda r: len(r[self.word_col]), axis=1), dict()


gen_without_tok: dict[
    str, Type[UnsplitUntokMetricMaker]] = {
        "frequency": UnsplitUntokFrequency,
        "length": UnsplitUntokLength
}
