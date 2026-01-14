"""Module for loading natural stories corpus
from a .tsv file.
"""

import random

import pandas as pd
from transformers import AutoTokenizer  # type: ignore

from . import utils
from ... import tokeniser

from typing import Tuple, overload


def line_to_components(line: str) -> Tuple[str, str, str, str]:
    token_id, token = line.split("\t")
    token = token[:-1]

    story_id, word_id, token_num = token_id.split(".")

    return story_id, word_id, token_num, token


def split_provo(
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

    df: pd.DataFrame = pd.read_csv(input_file, encoding="ISO-8859-1")

    # only these are necessary
    df = df[[
        "Word", "Text_ID", "Sentence_Number", "Word_In_Sentence_Number",]]

    story_ids_list = df["Text_ID"].unique().tolist()

    random.shuffle(story_ids_list)

    to_train = story_ids_list[:int(len(story_ids_list)*proportion)]

    mask = df["Text_ID"].isin(to_train)
    df1 = df[mask]
    df2 = df[~mask]

    df1.to_csv(
        out_path1, encoding="ISO-8859-1")
    df2.to_csv(
        out_path2, encoding="ISO-8859-1")


def load_provo(
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None,
        verbose: bool = False,
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

    df: pd.DataFrame = pd.read_csv(input_file, encoding="ISO-8859-1")

    df = df[[
        "Word", "Text_ID", "Sentence_Number",
        "Word_In_Sentence_Number"]]

    # Sort to be sure the order is right
    df.sort_values(by=[
        "Text_ID", "Sentence_Number", "Word_In_Sentence_Number"], inplace=True)

    # Make words lowercase
    if make_lower:
        df["Word"] = df["Word"].str.lower()

    # Apply pretokenisation and token mapper (to map to UNK)
    if token_mapper is not None:
        df["Word"] = df["Word"].apply(
            lambda t: [
                tup[0] for tup in
                pretokeniser.pre_tokenize_str(
                    t)])
        df["Word"] = df["Word"].apply(
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

    df["Sentence_ID"] = df[
        "Text_ID"].astype(str) + "_" + df["Sentence_Number"].astype(str)

    df["Word_In_Sentence_Number"] = df["Word_In_Sentence_Number"].astype(int)

    return (
        df["Word"].to_list(),
        df["Sentence_ID"].to_list(),
        df["Word_In_Sentence_Number"].to_list())


@overload
def prepare_RTs_provo(
        input_file: str, output_file: str,
        ) -> None:
    ...


@overload
def prepare_RTs_provo(
        input_file: str, output_file: None = None,
        ) -> pd.DataFrame:
    ...


def prepare_RTs_provo(
        input_file: str, output_file: str | None = None,
        ) -> None | pd.DataFrame:

    df: pd.DataFrame = pd.read_csv(input_file, encoding="ISO-8859-1")

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

    df["Sentence_ID"] = df[
        "Text_ID"].astype(str) + "_" + df["Sentence_Number"].astype(str)

    # # Remove sentences that contain NA values
    # remove_sentence_ids = df[df["Word"].isna()]["Sentence_ID"].unique()
    # df = df[~df["Sentence_ID"].isin(remove_sentence_ids)]
    # Removes 80.42 percent of sentences

    df["Word_In_Sentence_Number"] = df["Word_In_Sentence_Number"].astype(int)

    df.rename(columns={
        "Sentence_ID": "item",
        "Word_In_Sentence_Number": "zone",
        "Participant_ID": "WorkerId",
        "IA_REGRESSION_PATH_DURATION": "GPT",
        "IA_FIRST_FIXATION_DURATION": "FFD",
        "IA_FIRST_RUN_DWELL_TIME": "GD",
        "Word": "word"},
        inplace=True)

    df["Corpus"] = "provo"

    df = df[[
        "Corpus", "item", "zone", "WorkerId",
        "word", "GPT", "FFD", "GD"]]
    if output_file is None:
        return df
    df.to_csv(output_file)
    return None
