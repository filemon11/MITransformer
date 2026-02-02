"""Module for loading natural stories corpus
from a .tsv file.
"""

import tqdm
import random
import pandas as pd

from . import utils
from ... import tokeniser

from typing import Tuple, overload


def line_to_components(line: str) -> Tuple[str, str, str, str]:
    token_id, token = line.split("\t")
    token = token[:-1]

    story_id, word_id, token_num = token_id.split(".")

    return story_id, word_id, token_num, token


def split_naturalstories(
        input_file: str,
        proportion: float,
        out_path1: str | None = None,
        out_path2: str | None = None,
        verbose: bool = False) -> None:
    """
    This will result in all sentences of a story belonging to the same split.
    Guarantees a story split according to 'proportion' argument in order to not
    risk a huge deviation for corpora with a very small number of stories."""

    if out_path1 is None:
        out_path1 = utils.create_suffixed_filepath(input_file, "train")
    if out_path2 is None:
        out_path2 = utils.create_suffixed_filepath(input_file, "test")

    with open(input_file, mode="r") as file:
        story_ids: set[str] = set()

        for line in tqdm.tqdm(
                file,
                "Splitting naturalstories corpus (1/2)",
                disable=not verbose):
            story_ids.add(line_to_components(line)[0])
        file.seek(0)

        story_ids_list = list(story_ids)
        random.shuffle(story_ids_list)

        to_train = story_ids_list[:int(len(story_ids)*proportion)]
        story2train: dict[str, bool] = {
            story_id: (story_id in to_train)
            for story_id in story_ids}

        with (
                open(out_path1, mode="w") as out1,
                open(out_path2, mode="w") as out2):

            for line in tqdm.tqdm(
                    file,
                    "Splitting naturalstories corpus (2/2)",
                    disable=not verbose):
                if story2train[line_to_components(line)[0]]:
                    out1.write(line)
                else:
                    out2.write(line)


def load_natural_stories(
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None,
        verbose: bool = False
        ) -> tuple[list[str], list[str], list[int]]:
    """Load natural stories corpus from tsv file.

    Parameters
    ----------
    input_file : str
        .tsv file that contains the corpus.
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

    # TODO: add first item in id (e.g. 1.3.whole -> 1) to set for each row.
    # Then make list from that, make dict from list item to index,
    # create random mask and iterate through corpus to append the stories
    # This will result in all sentences of a story belonging to the same split.

    token_mapper = None
    if token_mapper_dir is not None:
        token_mapper = tokeniser.TokenMapper.load(token_mapper_dir)

    words: list[str] = []
    story_ids: list[str] = []
    word_ids: list[int] = []
    with open(input_file, "r") as file:
        for line in tqdm.tqdm(
                file,
                "Loading naturalstories corpus",
                disable=not verbose):

            story_id, word_id, token_num, token = line_to_components(line)

            if token_num == "whole":
                if make_lower:
                    token = token.lower()

                if token_mapper is not None:
                    tokens = token.split(" ")
                    token = token_mapper.decode(
                        token_mapper.encode([tokens]),
                        to_string=True)[0]

                words.append(token.replace(" ", ""))
                story_ids.append(story_id)
                word_ids.append(int(word_id))
    return words, story_ids, word_ids


@overload
def prepare_RTs_naturalstories(
        input_file: str, output_file: str
        ) -> None:
    ...


@overload
def prepare_RTs_naturalstories(
        input_file: str, output_file: None = None
        ) -> pd.DataFrame:
    ...


def prepare_RTs_naturalstories(
        input_file: str, output_file: str | None = None
        ) -> None | pd.DataFrame:
    # TODO simply copy the file
    df = pd.read_csv(input_file, sep='\t', header=0)
    df["Corpus"] = "naturalstories"
    df["item"] = df["item"].astype(str)
    df = df[["Corpus", "item", "zone", "WorkerId", "word", "RT"]]
    df["element"] = df["Corpus"] + df["zone"] + df["item"].astype(str)

    if output_file is None:
        return df
    df.to_csv(output_file)
    return None
