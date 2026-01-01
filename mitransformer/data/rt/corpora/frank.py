from transformers import AutoTokenizer  # type: ignore
import numpy as np
import tqdm
import pandas as pd

from . import utils
from ... import tokeniser

from typing import overload


def split_frank(
        input_file: str,
        proportion: float,
        out_path1: str | None = None,
        out_path2: str | None = None,
        verbose: bool = False) -> None:

    if out_path1 is None:
        out_path1 = utils.create_suffixed_filepath(input_file, "train")
    if out_path2 is None:
        out_path2 = utils.create_suffixed_filepath(input_file, "test")

    with open(input_file, mode="r", encoding='cp1252') as file:
        num_lines = sum(1 for _ in file)
        file.seek(0)

        include_in_train = np.random.rand(num_lines-1) < proportion

        file_iter = iter(file)
        header = next(file_iter)

        with open(out_path1, mode="w") as out1:
            out1.write(header)
            with open(out_path2, mode="w") as out2:
                out2.write(header)

                for line, iit in tqdm.tqdm(
                        zip(file_iter, include_in_train),
                        "Splitting UCL corpus", disable=not verbose):
                    if iit:
                        out1.write(line)
                    else:
                        out2.write(line)


def load_frank(
        input_file: str,
        make_lower: bool = True,
        token_mapper_dir: str | None = None,
        verbose: bool = False
        ) -> tuple[list[str], list[str], list[int]]:
    """Load natural stories corpus from tsv file.

    Parameters
    ----------
    input_file : str
        .txt file that contains the corpus.
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

    words: list[str] = []
    sentence_ids: list[str] = []
    word_ids: list[int] = []

    with open(input_file, mode="r", encoding='cp1252') as file:
        file_iter = iter(file)
        next(file_iter)
        for sentence_id, line in tqdm.tqdm(
                enumerate(file_iter, start=1),
                "Loading frank corpus",
                disable=not verbose):
            sentence = line.split("\t")[1]

            for word_id, word in enumerate(sentence.split(), start=1):
                if make_lower:
                    word = word.lower()

                if token_mapper is not None:
                    components = [
                        tup[0] for tup in
                        pretokeniser.pre_tokenize_str(
                            word)]
                    word = token_mapper.decode(
                        token_mapper.encode([components]),
                        to_string=True, join_with="")[0]

                words.append(word)
                sentence_ids.append(str(sentence_id))
                word_ids.append(word_id)
    return words, sentence_ids, word_ids


@overload
def prepare_RTs_frank_ET(
        input_file: str, output_file: str
        ) -> None:
    ...


@overload
def prepare_RTs_frank_ET(
        input_file: str, output_file: None = None
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
        "RTfirstpass": "GD"})
    df["WorkerId"] = df["WorkerId"].astype(str)
    df["item"] = df["item"].astype(str)
    df["Corpus"] = "frank_ET"
    df = df[[
        "Corpus", "item", "zone", "WorkerId",
        "word", "GPT", "FFD", "GD"]]
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
        input_file: str, output_file: None = None
        ) -> pd.DataFrame:
    ...


def prepare_RTs_frank_SP(
        input_file: str, output_file: str | None = None
        ) -> None | pd.DataFrame:
    df = pd.read_csv(input_file, sep='\t', header=0)
    df.rename(columns={
        "subj_nr": "WorkerId",
        "sent_nr": "item",
        "word_pos": "zone"},
        inplace=True)
    df["WorkerId"] = df["WorkerId"].astype(str)
    df["item"] = df["item"].astype(str)
    df["Corpus"] = "frank_SP"
    df = df[["Corpus", "item", "zone", "WorkerId", "word", "RT"]]
    if output_file is None:
        return df
    df.to_csv(output_file)
    return None
