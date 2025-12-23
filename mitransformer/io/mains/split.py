from .. import parsing
from ... import data

from typing import cast


def main_split(
        arguments: "parsing.SplitParserArgs") -> None:

    assert arguments.dataset_name in data.RTCORPORA
    arguments.dataset_name = cast(data.RTCorpus, arguments.dataset_name)

    data.split_RT_text(
        arguments.dataset_name, arguments.proportion,
        verbose=True)
