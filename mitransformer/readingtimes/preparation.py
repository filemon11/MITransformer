"""Snippets taken from
https://github.com/weijiexu-charlie/
Linearity-of-surprisal-on-RT/blob/main/Preparing%20Corpora/get_frequency.py"""


import pandas as pd

from ..train import LMTrainer
from ..data import (
    load_natural_stories, load_zuco, load_frank, CorpusLoader,
    TransformMaskHeadChild)
from .frame import SplitFrame, UnsplitFrame

from typing import (
    Iterable, Literal)

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
DEVICE = "cpu"
BATCH_SIZE = 8


Corpus = Literal["naturalstories", "zuco", "frank_SP", "frank_ET"]


def corpus_to_csv(
        corpus: Corpus,
        input_file: str, output_file: str,
        token_col: str = TOKEN_COL,
        text_id_col: str = TEXT_ID_COL,
        wnum_col: str = WNUM_COL,
        token_mapper_dir: str | None = None
        ) -> None:
    corpus_to_func: dict[Corpus, CorpusLoader] = {
        "naturalstories": load_natural_stories,
        "zuco": load_zuco,
        "frank_ET": load_frank,
        "frank_SP": load_frank
    }

    func = corpus_to_func[corpus]

    tokens, text_ids, wnums = func(
        input_file, token_mapper_dir=token_mapper_dir)
    # making lowercase makes no difference

    df = pd.DataFrame({
        token_col: tokens,
        text_id_col: text_ids,
        wnum_col: wnums})

    df.to_csv(output_file, index=False)


def process(
        input_file: str, output_file: str,
        model_dir: str, token_mapper_dir: str,
        raw: bool = True,
        token_col: str = TOKEN_COL,
        text_id_col: str = TEXT_ID_COL,
        wnum_col: str = WNUM_COL,
        baseline_metrics: Iterable[str] = ("frequency", "length"),
        device: str = DEVICE,
        batch_size: int = BATCH_SIZE,
        masks_setting: Literal[
            "current", "next"] = "current",
        only_content_words_left: bool = False,
        only_content_words_cost: bool = False,
        shift: int = -1,
        corpus: Corpus = "naturalstories"
        ) -> None:

    # Convert original format to sensible csv
    corpus_to_csv(
        corpus,
        input_file, output_file,
        token_col, text_id_col,
        wnum_col, None if raw else token_mapper_dir)

    # Add baseline predictors
    orig_frame = UnsplitFrame(
        pd.read_csv(output_file), {"word_col": token_col}, tokenised=False)
    # print(orig_frame.df["word"].to_list()); raise Exception

    for metric in baseline_metrics:
        orig_frame.add_(metric)

    df = pd.read_csv(output_file)
    words = df["word"]

    sentence_ids: None | pd.Series = None
    if corpus != "naturalstories":
        sentence_ids = df["item"]

    # Add surprisal
    frame = SplitFrame(tokenised=True)
    frame.add_(
        "conllu",
        words=words, sentence_ids=sentence_ids)  # dataset attribute missing
    frame.add_("space_after")
    frame.add_("word")
    frame.add_("position")
    frame.add_("head")
    frame.add_("pos")
    frame.add_("deprel")
    # TODO: subsume all above under conllu

    # Surprisal
    transform = TransformMaskHeadChild(
        keys_for_head={"head"},
        keys_for_child={"child"})
    # TODO: load these params from somewhere

    trainer = LMTrainer.load(
        model_dir,
        batch_size=batch_size,
        device=device,
        use_ddp=False,
        world_size=1)

    frame.add_(
        "surprisal", token_mapper_dir=token_mapper_dir,
        transform=transform, trainer=trainer, masks_setting=masks_setting)

    # Other metrics
    frame.add_(
        "mask", masks_setting="both",
        gov_name="head_current", dep_name="child_current")
    frame.add_(
        "head_distance",
        only_content_words_cost=only_content_words_cost,
        only_content_words_left=only_content_words_left)
    frame.add_("first_dependent_distance")
    frame.add_("first_dependent_deprel")
    frame.add_("left_dependents_distance_sum")
    frame.add_("left_dependents_count")
    frame.add_("demberg")

    if not masks_setting == "next":
        # Dependent on dependency prediction
        frame.add_("first_dependent_distance_weight")
        frame.add_("first_dependent_correct")
        frame.add_("expected_distance")
        frame.add_("kl_divergence")
        frame.add_("predicted_first_dependent_distance")

    if not masks_setting == "current":
        next_args = dict(
            gov_name="head_next",
            child_name="child_next",
            masks_setting="next"
        )
        frame.add_(
            "first_dependent_distance_weight",
            "first_dependent_distance_weight_next_col",
            **next_args)
        frame.add_(
            "first_dependent_correct",
            "first_dependent_correct_next_col",
            **next_args)
        frame.add_(
            "expected_distance",
            "expected_distance_next_col",
            **next_args)
        frame.add_(
            "kl_divergence",
            "kl_divergence_next_col",
            **next_args)
        frame.add_(
            "predicted_first_dependent_distance",
            "predicted_first_dependent_distance_next_col",
            **next_args)

    # TODO: Make it possible to provide a second argument to add_
    # to save the content in a new column
    # so we can compute the last for metrics for the succeeding
    # mask prediction too

    frame.untokenise_()

    split_frame = orig_frame.split([
        len(sentence) for sentence in frame.df["word"]])

    # # for debugging
    # for sen1, sen2 in zip(frame.df["word"], split_frame.df["word"]):
    #     print(sen1, sen2)
    #     assert sen1[0] == sen2[0]

    frame = frame.include_spillover(shift)
    # (frame_forward := frame.copy()).shift_(1)
    # (frame_backward := frame.copy()).shift_(-1)

    frame.truncate_(right=1)

    frame = split_frame | frame

    unsplit_frame = frame.unsplit()
    unsplit_frame.df.to_csv(output_file, index=False)
