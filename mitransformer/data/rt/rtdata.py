import pandas as pd

from . import corpora

from typing import Literal, overload

RTCorpus = Literal[
    "naturalstories", "naturalstories_train", "naturalstories_test",
    "zuco",
    "frank_ET", "frank_ET_train", "frank_ET_test",
    "frank_SP", "frank_SP_train", "frank_SP_test",
    "meco1", "meco1_train", "meco1_test",
    "meco2", "meco2_train", "meco2_test",
    "geco", "geco_train", "geco_test"]

RTCorpusTypes = Literal["ET", "SP"]

ET_CORPORA = {
    "frank_ET", "zuco", "meco1", "meco2", "geco"}
SP_CORPORA = {
    "naturalstories", "frank_SP"}
NO_SENTENCE_NUM_CORPORA = {"naturalstories", "geco"}

for constant in (ET_CORPORA, SP_CORPORA, NO_SENTENCE_NUM_CORPORA):
    for corpus in constant.copy():
        for variant in ("train", "test"):
            constant.add(f"{corpus}_{variant}")

RTCORPORA = ET_CORPORA | SP_CORPORA


rt_corpus_to_measurements_file: dict[RTCorpus, str] = {
        "naturalstories": "RT/data/processed_RTs.tsv",
        "zuco": "zuco/training_data.csv",
        "frank_ET": "frank/eyetracking.RT.txt",
        "frank_SP": "frank/selfpacedreading.RT.txt",
        "meco1": "meco/joint_l1_data_trimmed_version2.0.rda",
        "meco2": "meco/joint_data_trimmed_wave2_version2.0.rda",
        "geco": "geco/MonolingualReadingData.xlsx"
    }


rt_corpus_to_prepare_measurements_func: dict[
    RTCorpus, corpora.CorpusPreparer] = {
        "naturalstories": corpora.prepare_RTs_naturalstories,
        "zuco": corpora.prepare_RTs_zuco,
        "frank_ET": corpora.prepare_RTs_frank_ET,
        "frank_SP": corpora.prepare_RTs_frank_SP,
        "meco1": corpora.prepare_RTs_meco1,
        "meco2": corpora.prepare_RTs_meco2,
        "geco": corpora.prepare_RTs_geco,
    }


rt_corpus_to_prepare_text_func: dict[RTCorpus, corpora.CorpusLoader] = {
        "naturalstories": corpora.load_natural_stories,
        "zuco": corpora.load_zuco,
        "frank_ET": corpora.load_frank,
        "frank_SP": corpora.load_frank,
        "meco1": corpora.load_meco,
        "meco2": corpora.load_meco,
        "geco": corpora.load_geco,
    }


rt_corpus_to_split_func: dict[RTCorpus, corpora.CorpusSplitter] = {
        "naturalstories": corpora.split_naturalstories,
        "frank_ET": corpora.split_frank,
        "frank_SP": corpora.split_frank,
        "meco1": corpora.split_meco1,
        "meco2": corpora.split_meco2,
        "geco": corpora.split_geco,
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
        "zuco": "zuco/training_data.csv",
        "frank_ET": "frank/stimuli.txt",
        "frank_SP": "frank/stimuli.txt",
        "meco1": "meco/joint_l1_data_trimmed_version2.0.rda",
        "meco2": "meco/joint_data_trimmed_wave2_version2.0.rda",
        "geco": "geco/MonolingualReadingData.xlsx"
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
        corpus: RTCorpus = "naturalstories"
        ) -> None:
    ...


@overload
def prepare_RT_measurements(
        input_file: str, output_file: None = None,
        corpus: RTCorpus = "naturalstories"
        ) -> pd.DataFrame:
    ...


def prepare_RT_measurements(
        input_file: str, output_file: str | None = None,
        corpus: RTCorpus = "naturalstories"
        ) -> None | pd.DataFrame:
    return rt_corpus_to_prepare_measurements_func[
        corpus](input_file, output_file)


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

    # Don't allow concatenation of different splits of
    # the same corpus.
    if "naturalstories" in corpus:
        df["Corpus"] = "naturalstories"
    elif "frank_SP" in corpus:
        df["Corpus"] = "frank_SP"
    elif "frank_ET" in corpus:
        df["Corpus"] = "frank_ET"
    elif "meco1" in corpus:
        df["Corpus"] = "meco1"
    elif "meco2" in corpus:
        df["Corpus"] = "meco2"
    elif "geco" in corpus:
        df["Corpus"] = "geco"
    else:
        raise Exception("Corpus unknown.")

    return df


def split_RT_text(
        corpus: RTCorpus,
        proportion: float = 0.5,
        verbose: bool = False) -> None:

    rt_corpus_to_split_func[corpus](
        rt_corpus_to_text_file[corpus],
        proportion,
        verbose=verbose
    )
