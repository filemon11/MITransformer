"""Module for loading natural stories corpus
from a .tsv file.
"""

import pandas as pd
import random
import string
import difflib

from . import utils
from ... import tokeniser

from typing import overload, Literal


def split_zuco(
        input_file: str,
        proportion: float,
        out_path1: str | None = None,
        out_path2: str | None = None,
        verbose: bool = False
        ) -> None:

    if out_path1 is None:
        out_path1 = utils.create_suffixed_filepath(input_file, "train")
    if out_path2 is None:
        out_path2 = utils.create_suffixed_filepath(input_file, "test")

    df: pd.DataFrame = pd.read_csv(input_file, encoding="ISO-8859-1")

    # only these are necessary
    df = df[[
        "subject", "Sent_ID", "Word_ID", "Word", "FFD", "GPT", "GD"]]

    sent_ids_list = df["Sent_ID"].unique().tolist()

    random.shuffle(sent_ids_list)

    to_train = sent_ids_list[:int(len(sent_ids_list)*proportion)]

    mask = df["Sent_ID"].isin(to_train)
    df1 = df[mask]
    df2 = df[~mask]

    df1.to_csv(out_path1, index=False, encoding="ISO-8859-1")
    df2.to_csv(out_path2, index=False, encoding="ISO-8859-1")


def split_zuco2_1(
        input_file_zuco1_2_a: str,
        input_file_zuco1_2_b: str,
        input_file: str,
        proportion: float,
        out_path1: str | None = None,
        out_path2: str | None = None,
        verbose: bool = False
        ) -> None:
    """Split zuco1_2 first to account for sentences that appear in both.
    Specify the same proportion for both split action.
    Needs to apply heuristic string similarity function.
    It is not clear why many matches are not found through
    exact comparison. Different formatting?"""

    if out_path1 is None:
        out_path1 = utils.create_suffixed_filepath(input_file, "train")
    if out_path2 is None:
        out_path2 = utils.create_suffixed_filepath(input_file, "test")

    df: pd.DataFrame = pd.read_csv(input_file, encoding="ISO-8859-1")
    df_a: pd.DataFrame = pd.read_csv(
        input_file_zuco1_2_a, encoding="ISO-8859-1")
    df_b: pd.DataFrame = pd.read_csv(
        input_file_zuco1_2_b, encoding="ISO-8859-1")

    def remove_punctuation(s: str) -> str:
        # Necessary due to different formatting in the two
        # waves
        s = s.translate(
            str.maketrans("", "", string.punctuation))
        for h in ("‐", "-", "—",  "–", " "):
            s = s.replace(h, "")
        return s

    def get_text(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Word"] = df["Word"].astype(str)
        grouped = df.drop_duplicates(
            ["Sent_ID", "Word_ID"]).groupby(
                "Sent_ID")
        return grouped["Word"].apply(
            lambda x: remove_punctuation("".join(x)).lower()).reset_index()

    df_text = get_text(df)
    df_a_text = get_text(df_a)
    df_b_text = get_text(df_b)
    del df_a_text["Sent_ID"]
    del df_b_text["Sent_ID"]

    def mask_create(
            df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
        mask = df1["Word"].isin(df2["Word"])
        mask = mask.__or__(df1["Word"].apply(
            lambda x: any([y in x for y in df2["Word"]])))

        # Overly sensitive
        mask = mask.__or__(df1["Word"].apply(
            lambda x: any([difflib.SequenceMatcher(
                None, x, s2).ratio() > 0.4 for s2 in df2["Word"]])))
        return mask

    # necessary because for some sentences words are missing in
    # one of the two corpora
    df_a_merged = df_text[mask_create(df_text, df_a_text)]
    df_b_merged = df_text[mask_create(df_text, df_b_text)]

    a_sent_ids = df_a_merged["Sent_ID"].tolist()
    b_sent_ids = df_b_merged["Sent_ID"].tolist()

    # only these are necessary
    df = df[[
        "subject", "Sent_ID", "Word_ID", "Word", "FFD", "GPT", "GD"]]

    sent_ids_list = list(
        set(df["Sent_ID"].unique().tolist())
        - set(a_sent_ids)
        - set(b_sent_ids))

    random.shuffle(sent_ids_list)

    to_train = sent_ids_list[:int(len(sent_ids_list)*proportion)]

    mask = df["Sent_ID"].isin(to_train + a_sent_ids)
    df1 = df[mask]
    df2 = df[~mask]

    df1.to_csv(out_path1, index=False, encoding="ISO-8859-1")
    df2.to_csv(out_path2, index=False, encoding="ISO-8859-1")


def load_zuco(
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None,
        verbose: bool = False,
        ) -> tuple[list[str], list[str], list[str]]:
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
    list[str]
        For every token, its word ID.
    """

    # TODO: add first item in id (e.g. 1.3.whole -> 1) to set for eachr row.
    # Then make list from that, make dict from list item to index,
    # create random mask and iterate through corpus to append the stories
    # This will result in all sentences of a story belonging to the same split.

    pretokeniser = utils.load_pretokeniser()

    token_mapper = None
    if token_mapper_dir is not None:
        token_mapper = tokeniser.TokenMapper.load(token_mapper_dir)

    df: pd.DataFrame = pd.read_csv(     # type: ignore
        input_file, keep_default_na=False, na_values=None,
        encoding="ISO-8859-1")

    df["Word"] = df["Word"].astype(str)

    df = df[~(df["Word"].isna())]
    df = df[~(df["Word"] == "")]

    # Remove duplicates because we are only interested in
    # the text here
    df = df.drop_duplicates(["Sent_ID", "Word_ID"])

    # Sort to be sure the order is right
    df.sort_values(by=[
        "Sent_ID", "Word_ID"], inplace=True)

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

    return (
        df["Word"].to_list(),
        df["Sent_ID"].to_list(),
        df["Word_ID"].astype(str).to_list())


@overload
def prepare_RTs_zuco(
        wave: Literal[1, 2], task:  Literal[1, 2],
        input_file: str, output_file: str,
        only_interest: bool = True,
        ) -> None:
    ...


@overload
def prepare_RTs_zuco(
        wave: Literal[1, 2], task:  Literal[1, 2, 3],
        input_file: str, output_file: None = None,
        only_interest: bool = True,
        ) -> pd.DataFrame:
    ...


def prepare_RTs_zuco(
        wave: Literal[1, 2], task:  Literal[1, 2, 3],
        input_file: str, output_file: str | None = None,
        only_interest: bool = True,
        ) -> None | pd.DataFrame:
    if wave == 2:
        assert task != 3, "ZuCo 2.0 has only two tasks."

    df: pd.DataFrame = pd.read_csv(input_file, encoding="ISO-8859-1")

    df = df[~(df["Word"].isna())]
    df = df[~(df["Word"] == "")]

    df["Corpus"] = f"zuco{wave}_{task}"

    # rename columns
    df.rename(
        columns={
            "Sent_ID": "item",
            "Word_ID": "zone",
            "Word": "word",
            "subject": "WorkerId"
        },
        inplace=True
    )

    df["WorkerId"] = df["WorkerId"].astype(str)

    if only_interest:
        df = df[[
            "Corpus", "item", "zone", "WorkerId",
            "word", "GPT", "FFD", "GD"]]

    df["zone"] = df["zone"].astype(str)
    df["element"] = df[
        "Corpus"] + "_" + df["zone"] + "_" + df["item"].astype(str)

    if output_file is None:
        return df
    df.to_csv(output_file)
    return None


@overload
def prepare_RTs_zuco1_1(
        input_file: str, output_file: str,
        only_interest: bool = True,
        ) -> None:
    ...


@overload
def prepare_RTs_zuco1_1(
        input_file: str, output_file: None = None,
        only_interest: bool = True,
        ) -> pd.DataFrame:
    ...


def prepare_RTs_zuco1_1(
        input_file: str, output_file: str | None = None,
        only_interest: bool = True,
        ) -> None | pd.DataFrame:
    return prepare_RTs_zuco(1, 1, input_file, output_file)


@overload
def prepare_RTs_zuco1_2(
        input_file: str, output_file: str,
        only_interest: bool = True,
        ) -> None:
    ...


@overload
def prepare_RTs_zuco1_2(
        input_file: str, output_file: None = None,
        only_interest: bool = True,
        ) -> pd.DataFrame:
    ...


def prepare_RTs_zuco1_2(
        input_file: str, output_file: str | None = None,
        only_interest: bool = True,
        ) -> None | pd.DataFrame:
    return prepare_RTs_zuco(
        1, 2, input_file, output_file, only_interest=only_interest)


@overload
def prepare_RTs_zuco2_1(
        input_file: str, output_file: str,
        only_interest: bool = True,
        ) -> None:
    ...


@overload
def prepare_RTs_zuco2_1(
        input_file: str, output_file: None = None,
        only_interest: bool = True,
        ) -> pd.DataFrame:
    ...


def prepare_RTs_zuco2_1(
        input_file: str, output_file: str | None = None,
        only_interest: bool = True,
        ) -> None | pd.DataFrame:
    return prepare_RTs_zuco(
        2, 1, input_file, output_file, only_interest=only_interest)
