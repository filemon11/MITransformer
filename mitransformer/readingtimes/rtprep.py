import pandas as pd

from typing import Literal

Corpus = Literal["naturalstories", "zuco", "frank_ET", "frank_SP"]


def prepare_RTs_naturalstories(
        input_file: str, output_file: str
        ) -> None:
    # TODO simply copy the file
    df = pd.read_csv(input_file, sep='\t', header=0)
    df.to_csv(output_file)


def prepare_RTs_zuco(
        input_file: str, output_file: str
        ) -> None:
    df = pd.read_csv(input_file)
    df = df.rename(columns={"sentence_id": "item", "word_id": "zone"})
    df["WorkerId"] = 1
    df.to_csv(output_file)


def prepare_RTs_frank_ET(
        input_file: str, output_file: str
        ) -> None:
    df = pd.read_csv(input_file, sep='\t', header=0)
    df = df.rename(columns={
        "subj_nr": "WorkerId",
        "sent_nr": "item",
        "word_pos": "zone",
        "RTfirstfix": "FFD",
        "RTgopast": "GPT",
        "RTfirstpass": "GD",
        "RTrightbound": "RBT"})
    df.to_csv(output_file)


def prepare_RTs_frank_SP(
        input_file: str, output_file: str
        ) -> None:
    df = pd.read_csv(input_file, sep='\t', header=0)
    df = df.rename(columns={
        "subj_nr": "WorkerId",
        "sent_nr": "item",
        "word_pos": "zone"})
    df.to_csv(output_file)


def prepare_RTs(
        input_file: str, output_file: str,
        corpus: Corpus = "naturalstories"
        ) -> None:
    corpus_to_func = {
        "naturalstories": prepare_RTs_naturalstories,
        "zuco": prepare_RTs_zuco,
        "frank_ET": prepare_RTs_frank_ET,
        "frank_SP": prepare_RTs_frank_SP
    }
    corpus_to_func[corpus](input_file, output_file)
