"""Module for loading natural stories corpus
from a .tsv file.
"""

import pandas as pd
from transformers import AutoTokenizer  # type: ignore

from ... import tokeniser

from typing import overload


def load_zuco(
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None,
        verbose: bool = False
        ) -> tuple[list[str], list[str], list[int]]:
    """Load natural stories corpus from csv file.

    Parameters
    ----------
    input_file : str
        .csv file that contains the corpus.
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

    pretokeniser = AutoTokenizer.from_pretrained(
        "bert-base-uncased").backend_tokenizer.pre_tokenizer  # type: ignore

    token_mapper = None
    if token_mapper_dir is not None:
        token_mapper = tokeniser.TokenMapper.load(token_mapper_dir)

    file = pd.read_csv(input_file, keep_default_na=False, na_values=['NaN'])

    file["word"] = file["word"].str.replace("<EOS>", "")
    if make_lower:
        file["word"] = file["word"].str.lower()

    if token_mapper is not None:
        # TODO split at boundaries (punctuation, parantheses, ...)
        file["word"] = file["word"].apply(
            lambda t: [
                tup[0] for tup in
                pretokeniser.pre_tokenize_str(
                    t)])
        file["word"] = file["word"].apply(
            lambda t: token_mapper.decode(
                token_mapper.encode([t]),
                to_string=True, join_with="")[0])

    return (
        file["word"].to_list(),
        file["sentence_id"].astype(str).to_list(),
        file["word_id"].astype(int).to_list())


@overload
def prepare_RTs_zuco(
        input_file: str, output_file: str
        ) -> None:
    ...


@overload
def prepare_RTs_zuco(
        input_file: str, output_file: None = None
        ) -> pd.DataFrame:
    ...


def prepare_RTs_zuco(
        input_file: str, output_file: str | None = None
        ) -> None | pd.DataFrame:
    df = pd.read_csv(input_file)
    df.rename(
        columns={"sentence_id": "item", "word_id": "zone"},
        inplace=True)
    # TODO: Load correct corpus data and not aggregated over participants
    df["WorkerId"] = "1"
    df["item"] = df["item"].astype(str)
    if output_file is None:
        return df
    df.to_csv(output_file)
    return None
