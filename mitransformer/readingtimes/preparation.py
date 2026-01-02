"""Snippets taken from
https://github.com/weijiexu-charlie/
Linearity-of-surprisal-on-RT/blob/main/Preparing%20Corpora/get_frequency.py"""


import pandas as pd
import os

from ..train import LMTrainer
from .. import data
from .frame import SplitFrame, UnsplitFrame
from ..utils.params import Params

from typing import (
    Iterable, overload)

'''
The input files are the meta data (text without RT) of the corpus
that is already parsed and contains logp information. The output
files include one additional column of log-scaled frequency retrieved
from the package wordfreq (https://zenodo.org/records/7199437)
'''


pd.set_option('mode.chained_assignment', None)

LANG = "en"

TOKEN_COL = "word"
TEXT_ID_COL = "item"
WNUM_COL = "zone"
BASELINE_METRICS = ("frequency", "length")


@overload
def io_corpus_convert(
        model_dir: str,
        corpus: data.RTCorpus,
        input_file: str,
        output_file: None = None,
        token_mapper_dir: str | None = None,
        verbose: bool = False,
        min_len: None | int = None,
        max_len: None | int = None,
        remove_unk: bool = False,
        rank: int | None = None,
        ) -> pd.DataFrame:
    ...


@overload
def io_corpus_convert(
        model_dir: str,
        corpus: data.RTCorpus,
        input_file: str,
        output_file: str,
        token_mapper_dir: str | None = None,
        verbose: bool = False,
        min_len: None | int = None,
        max_len: None | int = None,
        remove_unk: bool = False,
        rank: int | None = None,
        ) -> None:
    ...


def io_corpus_convert(
        model_dir: str,
        corpus: data.RTCorpus,
        input_file: str,
        output_file: str | None = None,
        token_mapper_dir: str | None = None,
        verbose: bool = False,
        min_len: None | int = None,
        max_len: None | int = None,
        remove_unk: bool = False,
        rank: int | None = None,
        ) -> pd.DataFrame | None:
    if os.path.split(model_dir)[1][:4] == "hug:":
        df = data.prepare_RT_text(
            corpus,
            input_file,
            token_mapper_dir=token_mapper_dir,
            make_lower=False,
            verbose=verbose,
            min_len=min_len,
            max_len=max_len,
            rank=rank,
            remove_unk=remove_unk)
    else:
        # Convert original format to sensible csv
        df = data.prepare_RT_text(
            corpus,
            input_file,
            token_mapper_dir=token_mapper_dir,
            verbose=verbose,
            min_len=min_len,
            max_len=max_len,
            rank=rank,
            remove_unk=remove_unk)

    if output_file is None:
        return df
    df.to_csv(output_file, index=False)
    return None


def get_conllu_frame(
        df: pd.DataFrame,
        word_col: str = "word",
        sentence_col: str = "item",
        corpus_col: str = "Corpus") -> SplitFrame:
    words = df[word_col]
    sentence_ids = df[sentence_col]
    corpus_names = df[corpus_col]

    # Add surprisal
    frame = SplitFrame(tokenised=True)
    frame.add_(
        "conllu",
        words=words, sentence_ids=sentence_ids,
        corpus_names=corpus_names)  # dataset attribute missing
    frame.add_(
        "space_after", "word",
        "position", "head",
        "pos", "deprel")
    # TODO: subsume all above under conllu

    return frame


@overload
def process(
        input_file: str | pd.DataFrame,
        output_file: str,
        model_dir: str, token_mapper_dir: str,
        world_size: int = 1,
        token_col: str = TOKEN_COL,
        baseline_metrics: Iterable[str] = BASELINE_METRICS,
        only_content_words_left: bool = False,
        only_content_words_cost: bool = False,
        masks_setting: data.MasksSetting = "current",
        shift: int = 0,
        corpus: data.RTCorpus = "naturalstories",
        trainer_args: Params | None = None
        ) -> None:
    ...


@overload
def process(
        input_file: str | pd.DataFrame,
        output_file: None,
        model_dir: str, token_mapper_dir: str,
        world_size: int = 1,
        token_col: str = TOKEN_COL,
        baseline_metrics: Iterable[str] = BASELINE_METRICS,
        only_content_words_left: bool = False,
        only_content_words_cost: bool = False,
        masks_setting: data.MasksSetting = "current",
        shift: int = 0,
        corpus: data.RTCorpus = "naturalstories",
        trainer_args: Params | None = None
        ) -> pd.DataFrame:
    ...


def process(
        input_file: str | pd.DataFrame,
        output_file: str | None,
        model_dir: str, token_mapper_dir: str,
        world_size: int = 1,
        token_col: str = TOKEN_COL,
        baseline_metrics: Iterable[str] = BASELINE_METRICS,
        only_content_words_left: bool = False,
        only_content_words_cost: bool = False,
        masks_setting: data.MasksSetting = "current",
        shift: int = 0,
        corpus: data.RTCorpus = "naturalstories",
        trainer_args: Params | None = None
        ) -> pd.DataFrame | None:

    if isinstance(input_file, str):
        # Add baseline predictors
        input_file = pd.read_csv(
            input_file, keep_default_na=False, na_values=[''])
    else:
        input_file.fillna("NaN")

    orig_frame = UnsplitFrame(
        input_file, {"word_col": token_col}, tokenised=False)

    for metric in baseline_metrics:
        orig_frame.add_(metric)

    # Create conllu frame
    frame = get_conllu_frame(
        input_file, word_col=token_col)

    # Surprisal
    transform = data.TransformMaskHeadChild(
        keys_for_head={"head"},
        keys_for_child={"child"})
    # TODO: load these params from somewhere

    if model_dir[:4] == "hug:":
        model_dir = model_dir[:model_dir.rfind("_")]  # remove model number
        frame.add_(
            "surprisal", token_mapper_dir=token_mapper_dir,
            transform=transform, trainer=model_dir)
        frame.add_(
            "mask", masks_setting="both",
            gov_name="head_current", dep_name="child_current")
        frame.add_(
            "head_distance",
            "first_dependent_distance",
            "first_dependent_deprel",
            "left_dependents_distance_sum",
            "left_dependents_count",
            "demberg",
            only_content_words_cost=only_content_words_cost,
            only_content_words_left=only_content_words_left,
            only_left=True)

        frame.untokenise_()

        split_frame = orig_frame.split([
            len(sentence) for sentence in frame.df[token_col]])

        frame = split_frame | frame
        frame = frame.include_spillover(shift)

        frame.truncate_(right=1)
        if shift == 0:
            frame.truncate_(left=1)
            # truncate first word for which we do not have a probability

        unsplit_frame = frame.unsplit()
        if output_file is None:
            return unsplit_frame.df
        unsplit_frame.df.to_csv(output_file, index=False)
        return None

    else:
        additional = {} if trainer_args is None else trainer_args.to_dict()
        additional["model_name"] = model_dir
        trainer = LMTrainer.load(
            world_size=world_size,
            **additional)
        # omits undefined args

        frame.add_(
            "surprisal", token_mapper_dir=token_mapper_dir,
            transform=transform, trainer=trainer, masks_setting=masks_setting)

        # Other metrics
        frame.add_(
            "mask", masks_setting="both",
            gov_name="head_current", dep_name="child_current")
        frame.add_(
            "head_distance",
            "first_dependent_distance",
            "first_dependent_deprel",
            "left_dependents_distance_sum",
            "left_dependents_count",
            "demberg",
            only_content_words_cost=only_content_words_cost,
            only_content_words_left=only_content_words_left,
            only_left=True)

        candidates = (
                "first_dependent_distance_weight",
                "first_dependent_correct",
                "expected_distance",
                "kl_divergence",
                "predicted_first_dependent_distance",
                # "attention_entropy",  # why does this throw an exception?
            )
        if not masks_setting == "next":
            # Dependent on dependency prediction
            frame.add_(
                *candidates,
                only_past=True)

        if not masks_setting == "current":
            candidate_tuples = [
                cand + "_next_col" for cand in candidates]

            frame.add_(
                *candidate_tuples,
                gov_name="head_next",
                child_name="child_next",
                masks_setting="next",
                only_past=True)

        # TODO: Make it possible to provide a second argument to add_
        # to save the content in a new column
        # so we can compute the last for metrics for the succeeding
        # mask prediction too

        frame.untokenise_()

        split_frame = orig_frame.split([
            len(sentence) for sentence in frame.df[token_col]])

        # # for debugging
        # for sen1, sen2 in zip(frame.df["word"], split_frame.df["word"]):
        #     print(sen1, sen2)
        #     assert sen1[0] == sen2[0]

        frame = split_frame | frame
        frame = frame.include_spillover(shift)
        # (frame_forward := frame.copy()).shift_(1)
        # (frame_backward := frame.copy()).shift_(-1)

        frame.truncate_(right=1)

        unsplit_frame = frame.unsplit()
        if output_file is None:
            return unsplit_frame.df
        unsplit_frame.df.to_csv(output_file, index=False)
        return None
