"""Module for loading natural stories corpus
from a .tsv file.
"""

import random

import pandas as pd
import pyreadr  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from . import utils
from ... import tokeniser

from typing import Tuple, overload, Literal


def line_to_components(line: str) -> Tuple[str, str, str, str]:
    token_id, token = line.split("\t")
    token = token[:-1]

    story_id, word_id, token_num = token_id.split(".")

    return story_id, word_id, token_num, token


def split_meco(
        input_file: str,
        proportion: float,
        out_path1: str | None = None,
        out_path2: str | None = None,
        verbose: bool = False,
        lang: str = "en"
        ) -> None:
    """
    This will result in all sentences of a story belonging to the same split.
    Guarantees a story split according to 'proportion' argument in order to not
    risk a huge deviation for corpora with a very small number of stories."""

    if out_path1 is None:
        out_path1 = utils.create_suffixed_filepath(input_file, "train")
    if out_path2 is None:
        out_path2 = utils.create_suffixed_filepath(input_file, "test")

    df: pd.DataFrame = pyreadr.read_r(input_file)["joint.data"]

    # Filter language
    df = df[df["lang"] == lang]

    # only these are necessary
    df = df[[
        "word", "trialid", "wordnum", "sentnum",
        "firstrun.gopast", "firstfix.dur", "firstrun.dur"]]

    story_ids_list = df["trialid"].unique().tolist()

    random.shuffle(story_ids_list)
    print(story_ids_list)

    to_train = story_ids_list[:int(len(story_ids_list)*proportion)]

    mask = df["trialid"].isin(to_train)
    df1 = df[mask]
    df2 = df[~mask]

    pyreadr.write_rdata(
        out_path1, df1, "joint.data")
    pyreadr.write_rdata(
        out_path2, df2, "joint.data")


def split_meco1(
        input_file: str,
        proportion: float,
        out_path1: str | None = None,
        out_path2: str | None = None,
        verbose: bool = False,
        lang: str = "en"
        ) -> None:
    split_meco(
        input_file, proportion,
        out_path1, out_path2,
        verbose, lang
    )


def split_meco2(
        input_file: str,
        proportion: float,
        out_path1: str | None = None,
        out_path2: str | None = None,
        verbose: bool = False,
        lang: str = "en_uk"
        ) -> None:
    split_meco(
        input_file, proportion,
        out_path1, out_path2,
        verbose, lang
    )


def load_meco(
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None,
        verbose: bool = False,
        lang: str = "en"
        ) -> tuple[list[str], list[str], list[int]]:
    """Load meco corpus from .rda file.

    Parameters
    ----------
    input_file : str
        .rda file that contains the corpus.
        ('joint_l1_data_trimmed_version2.0.rda')
    make_lower : bool, default=True
        Whether to convert the tokens to lowercase.
    token_mapper_dir : str | None, default=None
        Path to a saved `tokeniser.TokenMapper`. If provided,
        every token in the .tsv file gets encoded and
        then decoded by the `tokeniser.TokenMapper` so that
        unknown tokens get replaced by th
        `tokeniser.TokenMapper.unk_token`.

    Returns
    -------
    list[str]
        The list of all tokens.
    list[str]
        For every token the story ID it appears in.
    list[int]
        For every token, its word ID.
    """

    # TODO: add first item in id (e.g. 1.3.whole -> 1) to set for eachr row.
    # Then make list from that, make dict from list item to index,
    # create random mask and iterate through corpus to append the stories
    # This will result in all sentences of a story belonging to the same split.

    pretokeniser = AutoTokenizer.from_pretrained(
        "bert-base-uncased").backend_tokenizer.pre_tokenizer  # type: ignore

    token_mapper = None
    if token_mapper_dir is not None:
        token_mapper = tokeniser.TokenMapper.load(token_mapper_dir)

    df: pd.DataFrame = pyreadr.read_r(input_file)["joint.data"]

    # Filter language
    df = df[df["lang"] == lang]

    # Remove duplicates because we are only interested in
    # the text here
    df = df[["word", "trialid", "wordnum", "sentnum"]].drop_duplicates()
    df["trialid"] = df["trialid"].astype(int)
    df["sentnum"] = df["sentnum"].astype(int)
    df["wordnum"] = df["wordnum"].astype(int)

    # Sort to be sure the order is right
    df.sort_values(by=["trialid", "wordnum"], inplace=True)

    # Make words lowercase
    if make_lower:
        df["word"] = df["word"].str.lower()

    # Apply pretokenisation and token mapper (to map to UNK)
    if token_mapper is not None:
        df["word"] = df["word"].apply(
            lambda t: [
                tup[0] for tup in
                pretokeniser.pre_tokenize_str(
                    t)])
        df["word"] = df["word"].apply(
            lambda t: token_mapper.decode(
                token_mapper.encode([t]),
                to_string=True, join_with="")[0])

    # This corpus comes with story ids, sentences numbers (per story)
    # and word numbers (also per story, i.e. zone in story).
    # Therefore, the sentence numbers are not important for identification
    # but useful for the parsing process we perform. Thus,
    # we merge story ids and sentence numbers to retain unique
    # identifiability of zone in corpus and be able to demark
    # every sentence.

    # In the long run, if we want to pass larger contexts that extend
    # sentence boundaries into the language model, we might want to
    # return three columns: story id, sentence num, word num

    df["trialid"] = df["trialid"].astype(str) + "_" + df["sentnum"].astype(str)
    return (
        df["word"].to_list(),
        df["trialid"].to_list(),
        df["wordnum"].to_list())


@overload
def prepare_RTs_meco(
        meco_wave: Literal[1, 2],
        input_file: str, output_file: str,
        lang: str = "en"
        ) -> None:
    ...


@overload
def prepare_RTs_meco(
        meco_wave: Literal[1, 2],
        input_file: str, output_file: None = None,
        lang: str = "en"
        ) -> pd.DataFrame:
    ...


def prepare_RTs_meco(
        meco_wave: Literal[1, 2],
        input_file: str, output_file: str | None = None,
        lang: str = "en"
        ) -> None | pd.DataFrame:

    df: pd.DataFrame = pyreadr.read_r(input_file)["joint.data"]

    # Filter language
    df = df[df["lang"] == lang]

    # This corpus comes with story ids, sentences numbers (per story)
    # and word numbers (also per story, i.e. zone in story).
    # Therefore, the sentence numbers are not important for identification
    # but useful for the parsing process we perform. Thus,
    # we merge story ids and sentence numbers to retain unique
    # identifiability of zone in corpus and be able to demark
    # every sentence.

    # In the long run, if we want to pass larger contexts that extend
    # sentence boundaries into the language model, we might want to
    # return three columns: story id, sentence num, word num

    df["trialid"] = df["trialid"].astype(int)
    df["sentnum"] = df["sentnum"].astype(int)
    df["wordnum"] = df["wordnum"].astype(int)
    df["trialid"] = df["trialid"].astype(str) + "_" + df["sentnum"].astype(str)

    df.rename(columns={
        "trialid": "item", "wordnum": "zone", "uniform_id": "WorkerId",
        "firstrun.gopast": "GPT",
        "firstfix.dur": "FFD",
        "firstrun.dur": "GD"},
        inplace=True)
    df["Corpus"] = f"meco{meco_wave}"

    if output_file is None:
        return df
    df.to_csv(output_file)
    return None


@overload
def prepare_RTs_meco1(
        input_file: str, output_file: str,
        lang: str = "en"
        ) -> None:
    ...


@overload
def prepare_RTs_meco1(
        input_file: str, output_file: None = None,
        lang: str = "en"
        ) -> pd.DataFrame:
    ...


def prepare_RTs_meco1(
        input_file: str, output_file: str | None = None,
        lang: str = "en"
        ) -> None | pd.DataFrame:
    return prepare_RTs_meco(
        1, input_file, output_file, lang)


@overload
def prepare_RTs_meco2(
        input_file: str, output_file: str,
        lang: str = "en_uk"
        ) -> None:
    ...


@overload
def prepare_RTs_meco2(
        input_file: str, output_file: None = None,
        lang: str = "en_uk"
        ) -> pd.DataFrame:
    ...


def prepare_RTs_meco2(
        input_file: str, output_file: str | None = None,
        lang: str = "en_uk"
        ) -> None | pd.DataFrame:
    return prepare_RTs_meco(
        2, input_file, output_file, lang)
