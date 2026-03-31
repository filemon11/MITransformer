"""Module for loading natural stories corpus
from a .tsv file.
"""

import conllu
import pandas as pd

from . import utils
from ... import tokeniser

from typing import Tuple


def line_to_components(line: str) -> Tuple[str, str, str, str]:
    token_id, token = line.split("\t")
    token = token[:-1]

    story_id, word_id, token_num = token_id.split(".")

    return story_id, word_id, token_num, token


def load_ud(
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None,
        verbose: bool = False,
        ) -> tuple[list[str], list[str], list[str]]:
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
    list[str]
        For every token, its word ID.
    """

    pretokeniser = utils.load_pretokeniser()

    token_mapper = None
    if token_mapper_dir is not None:
        token_mapper = tokeniser.TokenMapper.load(token_mapper_dir)

    df = pd.DataFrame(columns=["item", "zone", "word"])
    # with open(input_file, "r") as src:
    #     current = 0
    #     for i, sentence in enumerate(conllu.parse_incr(src)):
    #         for j, token in enumerate(sentence):
    #             df.loc[current] = pd.Series({
    #                 "item": i,
    #                 "zone": j,
    #                 "word": token["form"]})
    #             current += 1

    with open(input_file, "r") as src:
        current_row = 0
        current_item = 0
        current_zone = 0
        for i, line in enumerate(src):
            if line == "\n":
                current_item += 1
                current_zone = 0
                continue
            for j, token in enumerate(line.split()):
                df.loc[current_row] = pd.Series({
                    "item": current_item,
                    "zone": current_zone,
                    "word": token})
                current_zone += 1
                current_row += 1

    # Sort to be sure the order is right
    df.sort_values(by=[
        "item", "zone"], inplace=True)
    df.to_csv("testcsv.csv")

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

    return (
        df["word"].to_list(),
        df["item"].to_list(),
        df["zone"].to_list())
