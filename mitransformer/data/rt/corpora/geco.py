"""Module for loading natural stories corpus
from a .tsv file.
"""

import random

import pandas as pd
import numpy as np
from transformers import AutoTokenizer  # type: ignore

from . import utils
from ... import tokeniser

from typing import Tuple, overload


def line_to_components(line: str) -> Tuple[str, str, str, str]:
    token_id, token = line.split("\t")
    token = token[:-1]

    story_id, word_id, token_num = token_id.split(".")

    return story_id, word_id, token_num, token


def split_geco(
        input_file: str,
        proportion: float,
        out_path1: str | None = None,
        out_path2: str | None = None,
        verbose: bool = False
        ) -> None:
    """
    This will result in all sentences of a story belonging to the same split.
    Guarantees a story split according to 'proportion' argument in order to not
    risk a huge deviation for corpora with a very small number of stories."""

    if out_path1 is None:
        out_path1 = utils.create_suffixed_filepath(input_file, "train")
    if out_path2 is None:
        out_path2 = utils.create_suffixed_filepath(input_file, "test")

    df: pd.DataFrame = pd.read_excel(input_file)

    # only these are necessary
    df = df[[
        "PP_NR", "PART", "TRIAL", "WORD_ID_WITHIN_TRIAL",
        "WORD", "WORD_FIRST_FIXATION_DURATION", "WORD_GO_PAST_TIME",
        "WORD_GAZE_DURATION", "WORD_ID"]]

    story_ids_list = df["PART"].unique().tolist()

    random.shuffle(story_ids_list)

    to_train = story_ids_list[:int(len(story_ids_list)*proportion)]

    mask = df["PART"].isin(to_train)
    df1 = df[mask]
    df2 = df[~mask]

    df1.to_excel(out_path1)
    df2.to_excel(out_path2)


def load_geco(
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None,
        verbose: bool = False,
        ) -> tuple[list[str], list[str], list[int]]:
    """Load geco corpus from .xlsx file.

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
        For every token the part ID it appears in.
    list[int]
        For every token, its word ID.
    """

    # TODO: add first item in id (e.g. 1.3.whole -> 1) to set for eachr row.
    # Then make list from that, make dict from list item to index,
    # create random mask and iterate through corpus to append the stories
    # This will result in all sentences of a story belonging to the same split.

    pretokeniser = AutoTokenizer.from_pretrained(
        "./bert-base-uncased",
        local_files_only=True).backend_tokenizer.pre_tokenizer  # type: ignore

    token_mapper = None
    if token_mapper_dir is not None:
        token_mapper = tokeniser.TokenMapper.load(token_mapper_dir)

    df: pd.DataFrame = pd.read_excel(
        input_file, keep_default_na=False, na_values=None)
    df = df[~(df["WORD"].isna())]
    df = df[~(df["WORD"] == "")]
    df["WORD"] = df["WORD"].astype(str)

    # Remove duplicates because we are only interested in
    # the text here
    df = df.drop_duplicates(["WORD_ID"])

    # Create new zone entries
    lens_of_parts = df[["PART"]].groupby(by="PART").size()
    zones = [np.arange(1, length+1) for length in lens_of_parts]

    df["zone"] = np.concat(zones)

    # Make words lowercase
    if make_lower:
        df["WORD"] = df["WORD"].str.lower()

    # Apply pretokenisation and token mapper (to map to UNK)
    if token_mapper is not None:
        df["WORD"] = df["WORD"].apply(
            lambda t: [
                tup[0] for tup in
                pretokeniser.pre_tokenize_str(
                    t)])
        df["WORD"] = df["WORD"].apply(
            lambda t: token_mapper.decode(
                token_mapper.encode([t]),
                to_string=True, join_with="")[0])

    df = df[~(df["WORD"] == " ")]
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

    return (
        df["WORD"].to_list(),
        df["PART"].to_list(),
        df["zone"].to_list())


@overload
def prepare_RTs_geco(
        input_file: str, output_file: str
        ) -> None:
    ...


@overload
def prepare_RTs_geco(
        input_file: str, output_file: None = None
        ) -> pd.DataFrame:
    ...


def prepare_RTs_geco(
        input_file: str, output_file: str | None = None
        ) -> None | pd.DataFrame:
    # NOTE: Removes all trials containing words containing
    # ...<letter> because these are split by spacy which creates
    # alignment problems

    df: pd.DataFrame = pd.read_excel(input_file)
    df = df[~(df["WORD"].isna())]
    df = df[~(df["WORD"] == "")]

    # Create new zone entries
    df_by_worker_by_item = df.groupby(["PP_NR", "PART"])

    lens_of_parts = df_by_worker_by_item.size()
    zones = np.concat([np.arange(1, length+1) for length in lens_of_parts])

    df["zone"] = zones
    df["Corpus"] = "geco"

    # rename columns
    df.rename(
        columns={
            "PP_NR": "WorkerId",
            "PART": "item",
            "WORD_FIRST_FIXATION_DURATION": "FFD",
            "WORD_GO_PAST_TIME": "GPT",
            "WORD_GAZE_DURATION": "GD",
            "WORD": "word",
        },
        inplace=True
    )

    for col in ("FFD", "GPT", "GD"):
        df[col] = df[col].replace(".", 0)

    df = df[[
        "Corpus", "item", "zone", "WorkerId",
        "word", "GPT", "FFD", "GD"]]
    if output_file is None:
        return df
    df.to_csv(output_file)
    return None
