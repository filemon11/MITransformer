import pandas as pd
import spacy

from . import corpora

from typing import Literal, overload

from ...utils.logmaker import (
    getLogger, info)

logger = getLogger(__name__)

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


RTCorpus = Literal[
    "naturalstories", "naturalstories_train", "naturalstories_test",
    "zuco1_1", "zuco1_1_train", "zuco1_1_test",
    "zuco1_2", "zuco1_2_train", "zuco1_2_test",
    "zuco2_1", "zuco2_1_train", "zuco2_1_test",
    "frank_ET", "frank_ET_train", "frank_ET_test",
    "frank_SP", "frank_SP_train", "frank_SP_test",
    "meco1", "meco1_train", "meco1_test",
    "meco2", "meco2_train", "meco2_test",
    "geco", "geco_train", "geco_test",
    "provo", "provo_train", "provo_test",
    "EWT"
]

RTCorpusTypes = Literal["ET", "SP"]

ET_CORPORA = {
    "frank_ET", "zuco1_1", "zuco1_2",
    "zuco2_1", "meco1", "meco2", "geco",
    "provo"}
SP_CORPORA = {
    "naturalstories", "frank_SP"}
NO_SENTENCE_NUM_CORPORA = {"naturalstories", "geco", "EWT"}

NO_MEASUREMENT_CORPORA = {"EWT", }

CORE_CORPORA = set(ET_CORPORA | SP_CORPORA | NO_MEASUREMENT_CORPORA)

for constant in (ET_CORPORA, SP_CORPORA, NO_SENTENCE_NUM_CORPORA):
    for corpus in constant.copy():
        for variant in ("train", "test"):
            constant.add(f"{corpus}_{variant}")


RTCORPORA = ET_CORPORA | SP_CORPORA | NO_MEASUREMENT_CORPORA

rt_corpus_to_measurements_file: dict[RTCorpus, str] = {
    "naturalstories": "RT/data/processed_RTs.tsv",
    "zuco1_1": "zuco/zuco1/task1.csv",
    "zuco1_2": "zuco/zuco1/task2.csv",
    "zuco2_1": "zuco/zuco2/task1.csv",
    "frank_ET": "frank/eyetracking.RT.txt",
    "frank_SP": "frank/selfpacedreading.RT.txt",
    "meco1": "meco/joint_l1_data_trimmed_version2.0.rda",
    "meco2": "meco/joint_data_trimmed_wave2_version2.0.rda",
    "geco": "geco/MonolingualReadingData.xlsx",
    "provo": "provo/Provo_Corpus-Eyetracking_Data.csv"
}


rt_corpus_to_prepare_measurements_func: dict[
    RTCorpus, corpora.CorpusPreparer] = {
        "naturalstories": corpora.prepare_RTs_naturalstories,
        "zuco1_1": corpora.prepare_RTs_zuco1_1,
        "zuco1_2": corpora.prepare_RTs_zuco1_2,
        "zuco2_1": corpora.prepare_RTs_zuco2_1,
        "frank_ET": corpora.prepare_RTs_frank_ET,
        "frank_SP": corpora.prepare_RTs_frank_SP,
        "meco1": corpora.prepare_RTs_meco1,
        "meco2": corpora.prepare_RTs_meco2,
        "geco": corpora.prepare_RTs_geco,
        "provo": corpora.prepare_RTs_provo
    }


rt_corpus_to_prepare_text_func: dict[RTCorpus, corpora.CorpusLoader] = {
        "naturalstories": corpora.load_natural_stories,
        "zuco1_1": corpora.load_zuco,
        "zuco1_2": corpora.load_zuco,
        "zuco2_1": corpora.load_zuco,
        "frank_ET": corpora.load_frank,
        "frank_SP": corpora.load_frank,
        "meco1": corpora.load_meco1,
        "meco2": corpora.load_meco2,
        "geco": corpora.load_geco,
        "provo": corpora.load_provo,
        "EWT": corpora.load_ud,
    }


rt_corpus_to_split_func: dict[RTCorpus, corpora.CorpusSplitter] = {
        "naturalstories": corpora.split_naturalstories,
        "zuco1_1": corpora.split_zuco,
        "zuco1_2": corpora.split_zuco,
        "zuco2_1": lambda *args, **kwargs: corpora.split_zuco2_1(
            rt_corpus_to_text_file["zuco1_2_train"],
            rt_corpus_to_text_file["zuco1_2_test"],
            *args, **kwargs),
        "frank_ET": corpora.split_frank,
        "frank_SP": corpora.split_frank,
        "meco1": corpora.split_meco1,
        "meco2": corpora.split_meco2,
        "geco": corpora.split_geco,
        "provo": corpora.split_provo,
    }


for mapping in (
        rt_corpus_to_prepare_measurements_func,
        rt_corpus_to_prepare_text_func,
        rt_corpus_to_split_func,
        rt_corpus_to_measurements_file):
    for corpus, value in mapping.copy().items():
        for variant in ("train", "test"):
            mapping[f"{corpus}_{variant}"] = value  # type: ignore


rt_corpus_to_text_file: dict[RTCorpus, str] = {
        "naturalstories": "naturalstories-master/words.tsv",
        "zuco1_1": "zuco/zuco1/task1.csv",
        "zuco1_2": "zuco/zuco1/task2.csv",
        "zuco2_1": "zuco/zuco2/task1.csv",
        "frank_ET": "frank/stimuli.txt",
        "frank_SP": "frank/stimuli.txt",
        "meco1": "meco/joint_l1_data_trimmed_version2.0.rda",
        "meco2": "meco/joint_data_trimmed_wave2_version2.0.rda",
        "geco": "geco/MonolingualReadingData.xlsx",
        "provo": "provo/Provo_Corpus-Eyetracking_Data_Words.csv",
        "EWT": "UD/UD_English-EWT/en_ewt-ud-test.txt",
    }


for corpus, path in rt_corpus_to_text_file.copy().items():
    for variant in ("train", "test"):
        rt_corpus_to_text_file[
            f"{corpus}_{variant}"] = (  # type: ignore
                corpora.create_suffixed_filepath(
                    path, variant))


@overload
def prepare_RT_measurements(
        input_file: str, output_file: str,
        corpus: RTCorpus = "naturalstories",
        only_interest: bool = True,
        ) -> None:
    ...


@overload
def prepare_RT_measurements(
        input_file: str, output_file: None = None,
        corpus: RTCorpus = "naturalstories",
        only_interest: bool = True,
        ) -> pd.DataFrame:
    ...


def prepare_RT_measurements(
        input_file: str, output_file: str | None = None,
        corpus: RTCorpus = "naturalstories",
        only_interest: bool = True,
        ) -> None | pd.DataFrame:
    return rt_corpus_to_prepare_measurements_func[
        corpus](
            input_file, output_file,
            only_interest=only_interest)


TOKEN_COL = "word"
TEXT_ID_COL = "item"
WNUM_COL = "zone"
BASELINE_METRICS = ("frequency", "length")


def prepare_RT_text(
        corpus: RTCorpus,
        input_file: str,
        token_col: str = TOKEN_COL,
        text_id_col: str = TEXT_ID_COL,
        wnum_col: str = WNUM_COL,
        token_mapper_dir: str | None = None,
        make_lower: bool = True,
        verbose: bool = False,
        min_len: None | int = None,
        max_len: None | int = None,
        remove_unk: bool = False,
        rank: None | int = None,
        ) -> pd.DataFrame:

    func = rt_corpus_to_prepare_text_func[corpus]

    tokens, text_ids, wnums = func(
        input_file, token_mapper_dir=token_mapper_dir,
        make_lower=make_lower,
        verbose=verbose)
    # making lowercase makes no difference

    df = pd.DataFrame({
        token_col: tokens,
        text_id_col: text_ids,
        wnum_col: wnums})

    length1 = len(df)

    if min_len is not None or max_len is not None:
        df = filter_sentences_by_length(  # type: ignore
            df, token_col=token_col,
            text_id_col=text_id_col,
            min_len=min_len, max_len=max_len,  # type: ignore
            remove_unk=remove_unk
        )
    length2 = len(df)

    info(
        rank, logger,
        f"Removed {length1-length2}/{length1} tokens at {corpus} import.")

    # Don't allow concatenation of different splits of
    # the same corpus.
    for c in CORE_CORPORA:
        if c in corpus:
            df["Corpus"] = c
            break
    else:
        raise Exception("Corpus unknown.")

    return df


@overload
def filter_sentences_by_length(
        df: pd.DataFrame,
        token_col: str,
        text_id_col: str,
        min_len: int,
        max_len: int | None = None,
        remove_unk: bool = False
        ) -> pd.DataFrame:
    ...


@overload
def filter_sentences_by_length(
        df: pd.DataFrame,
        token_col: str,
        text_id_col: str,
        min_len: int | None,
        max_len: int,
        remove_unk: bool = False
        ) -> pd.DataFrame:
    ...


def filter_sentences_by_length(
        df: pd.DataFrame,
        token_col: str,
        text_id_col: str,
        min_len: int | None = None,
        max_len: int | None = None,
        remove_unk: bool = False
        ) -> pd.DataFrame:
    """
    Filter dataframe to only keep tokens belonging to sentences
    with length in [min_len, max_len], preserving token ↔ wnum mapping.
    """

    keep_indices: set[int] = set()

    # Process each text_id independently
    for _, group in df.groupby(text_id_col, sort=False):
        tokens = group[token_col].tolist()
        row_indices = group.index.tolist()

        # Reconstruct text
        text = " ".join(tokens)

        # Compute character spans of original tokens
        token_spans: list[tuple[int, int]] = []
        cursor = 0
        for tok in tokens:
            start = cursor
            end = start + len(tok)
            token_spans.append((start, end))
            cursor = end + 1  # space

        # Sentence segmentation with preserved offsets
        doc = nlp(text)

        for sent in doc.sents:
            if remove_unk and any(["<unk>" in word.text for word in sent]):
                continue

            s_start = sent.start_char
            s_end = sent.end_char

            token_ids = [
                i for i, (t_start, t_end) in enumerate(token_spans)
                if t_start >= s_start and t_end <= s_end
            ]

            sent_len = len(token_ids)
            if min_len is None or min_len <= sent_len:
                if max_len is None or sent_len <= max_len:
                    for i in token_ids:
                        keep_indices.add(row_indices[i])

    return df.loc[sorted(keep_indices)].reset_index(drop=True)


def split_RT_text(
        corpus: RTCorpus,
        proportion: float = 0.5,
        verbose: bool = False) -> None:

    rt_corpus_to_split_func[corpus](
        rt_corpus_to_text_file[corpus],
        proportion,
        verbose=verbose
    )
