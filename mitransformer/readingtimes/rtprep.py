import pandas as pd

from typing import Literal, overload

Corpus = Literal["naturalstories", "zuco", "frank_ET", "frank_SP"]
CorpusTypes = Literal["ET", "SP"]
CORPORA = {"naturalstories", "zuco", "frank_ET", "frank_SP"}
ET_CORPORA = {"frank_ET", "zuco"}
SP_CORPORA = {"naturalstories", "frank_SP"}


@overload
def prepare_RTs_naturalstories(
        input_file: str, output_file: str
        ) -> None:
    ...


@overload
def prepare_RTs_naturalstories(
        input_file: str, output_file: None
        ) -> pd.DataFrame:
    ...


def prepare_RTs_naturalstories(
        input_file: str, output_file: str | None
        ) -> None | pd.DataFrame:
    # TODO simply copy the file
    df = pd.read_csv(input_file, sep='\t', header=0)
    if output_file is None:
        return df
    df.to_csv(output_file)
    return None


@overload
def prepare_RTs_zuco(
        input_file: str, output_file: str
        ) -> None:
    ...


@overload
def prepare_RTs_zuco(
        input_file: str, output_file: None
        ) -> pd.DataFrame:
    ...


def prepare_RTs_zuco(
        input_file: str, output_file: str | None = None
        ) -> None | pd.DataFrame:
    df = pd.read_csv(input_file)
    df = df.rename(columns={"sentence_id": "item", "word_id": "zone"})
    # TODO: Load correct corpus data and not aggregated over participants
    df["WorkerId"] = 1
    if output_file is None:
        return df
    df.to_csv(output_file)
    return None


@overload
def prepare_RTs_frank_ET(
        input_file: str, output_file: str
        ) -> None:
    ...


@overload
def prepare_RTs_frank_ET(
        input_file: str, output_file: None
        ) -> pd.DataFrame:
    ...


def prepare_RTs_frank_ET(
        input_file: str, output_file: str | None = None
        ) -> None | pd.DataFrame:
    df = pd.read_csv(input_file, sep='\t', header=0)
    df = df.rename(columns={
        "subj_nr": "WorkerId",
        "sent_nr": "item",
        "word_pos": "zone",
        "RTfirstfix": "FFD",
        "RTgopast": "GPT",
        "RTfirstpass": "GD",
        "RTrightbound": "RBT"})
    if output_file is None:
        return df
    df.to_csv(output_file)
    return None


@overload
def prepare_RTs_frank_SP(
        input_file: str, output_file: str
        ) -> None:
    ...


@overload
def prepare_RTs_frank_SP(
        input_file: str, output_file: None
        ) -> pd.DataFrame:
    ...


def prepare_RTs_frank_SP(
        input_file: str, output_file: str | None = None
        ) -> None | pd.DataFrame:
    df = pd.read_csv(input_file, sep='\t', header=0)
    df = df.rename(columns={
        "subj_nr": "WorkerId",
        "sent_nr": "item",
        "word_pos": "zone"})
    if output_file is None:
        return df
    df.to_csv(output_file)
    return None


@overload
def prepare_RTs(
        input_file: str, output_file: str,
        corpus: Corpus = "naturalstories"
        ) -> None:
    ...


@overload
def prepare_RTs(
        input_file: str, output_file: None = None,
        corpus: Corpus = "naturalstories"
        ) -> pd.DataFrame:
    ...


def prepare_RTs(
        input_file: str, output_file: str | None = None,
        corpus: Corpus = "naturalstories"
        ) -> None | pd.DataFrame:
    corpus_to_func = {
        "naturalstories": prepare_RTs_naturalstories,
        "zuco": prepare_RTs_zuco,
        "frank_ET": prepare_RTs_frank_ET,
        "frank_SP": prepare_RTs_frank_SP
    }
    return corpus_to_func[corpus](input_file, output_file)
