from .. import parsing
from ... import readingtimes
from ... import data

from typing import cast


def main_split(
        arguments: "parsing.SplitParserArgs") -> None:

    corpus_to_func: dict[readingtimes.Corpus, data.CorpusSplitter] = {
        "naturalstories": data.split_naturalstories,
        "naturalstories_train": data.split_naturalstories,
        "naturalstories_test": data.split_naturalstories,
        "frank_ET": data.split_frank,
        "frank_ET_train": data.split_frank,
        "frank_ET_test": data.split_frank,
        "frank_SP": data.split_frank,
        "frank_SP_train": data.split_frank,
        "frank_SP_test": data.split_frank,
    }

    assert arguments.dataset_name in readingtimes.CORPORA
    arguments.dataset_name = cast(readingtimes.Corpus, arguments.dataset_name)

    corpus_to_func[arguments.dataset_name](
        data.rt_corpus_to_text_file[arguments.dataset_name],
        arguments.proportion
    )
