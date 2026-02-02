"""Snippets taken from
https://github.com/weijiexu-charlie/
Linearity-of-surprisal-on-RT/blob/main/Preparing%20Corpora/get_frequency.py"""


import pandas as pd
import os
from conllu.models import TokenList
import pathlib
import torch.distributed as dist

from ..train import LMTrainer
from .. import data
from .frame import SplitFrame, UnsplitFrame
from ..utils.params import Params, TypeUndefined, Undefined, is_undef

from typing import (
    Iterable, overload, Sequence, Tuple, Literal)

from mitransformer.utils.logmaker import (
    getLogger, info)

logger = getLogger(__name__)


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
        "space_after", "word", "head",
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


@overload
def new_process(
        input_file: str | pd.DataFrame,
        output_file: str,
        model_dir: str,
        token_mapper_dir: str,
        to_add: Sequence[str],
        batch_size: int | TypeUndefined = Undefined,
        world_size: int = 1,
        use_ddp: bool = False,
        rank: int | None = None,
        token_col: str = TOKEN_COL,
        baseline_metrics: Iterable[str] = BASELINE_METRICS,
        only_content_words_left: bool = False,
        only_content_words_cost: bool = False,
        masks_setting: data.MasksSetting = "current",
        masked: bool = False,
        shift: int = 0,
        trainer_args: Params | None = None,
        transform_mask: (
            None | data.TransformFunc) = None,
        distr_mode: Literal[
            "att", "att-n"] | TypeUndefined = Undefined,
        length_weighted: bool | TypeUndefined = Undefined,
        include_current: bool | TypeUndefined = Undefined,
        global_distr: bool | TypeUndefined = Undefined,
        ) -> None:
    ...


@overload
def new_process(
        input_file: str | pd.DataFrame,
        output_file: None,
        model_dir: str,
        token_mapper_dir: str,
        to_add: Sequence[str],
        batch_size: int | TypeUndefined = Undefined,
        world_size: int = 1,
        use_ddp: bool = False,
        rank: int | None = None,
        token_col: str = TOKEN_COL,
        baseline_metrics: Iterable[str] = BASELINE_METRICS,
        only_content_words_left: bool = False,
        only_content_words_cost: bool = False,
        masks_setting: data.MasksSetting = "current",
        masked: bool = False,
        shift: int = 0,
        trainer_args: Params | None = None,
        transform_mask: (
            None | data.TransformFunc) = None,
        distr_mode: Literal[
            "att", "att-n"] | TypeUndefined = Undefined,
        length_weighted: bool | TypeUndefined = Undefined,
        include_current: bool | TypeUndefined = Undefined,
        global_distr: bool | TypeUndefined = Undefined,
        ) -> pd.DataFrame:
    ...


def new_process(
        input_file: str | pd.DataFrame,
        output_file: str | None,
        model_dir: str,
        token_mapper_dir: str,
        to_add: Sequence[str],
        batch_size: int | TypeUndefined = Undefined,
        world_size: int = 1,
        use_ddp: bool = False,
        rank: int | None = None,
        token_col: str = TOKEN_COL,
        baseline_metrics: Iterable[str] = BASELINE_METRICS,
        only_content_words_left: bool = False,
        only_content_words_cost: bool = False,
        masks_setting: data.MasksSetting = "current",
        masked: bool = False,
        shift: int = 0,
        trainer_args: Params | None = None,
        transform_mask: (
            None | data.TransformFunc) = None,
        distr_mode: Literal[
            "att", "att-n"] | TypeUndefined = Undefined,
        length_weighted: bool | TypeUndefined = Undefined,
        include_current: bool | TypeUndefined = Undefined,
        global_distr: bool | TypeUndefined = Undefined,
        ) -> pd.DataFrame | None:

    # Add baseline predictors
    frame, untok_frame = get_frames(
        input_file, 0, token_col=token_col,
        baseline_metrics=baseline_metrics
    )

    if masked and transform_mask is None:
        transform_mask = data.TransformMaskHeadChild(
            keys_for_head={"head"},
            keys_for_child={"child"})

    legacy_candidates = [
        "first_dependent_distance_weight",
        "first_dependent_correct",
        "expected_distance",
        "kl_divergence",
        "predicted_first_dependent_distance",
    ]

    legacy_candidates = [
        predictor for predictor in to_add if predictor in legacy_candidates
    ]

    to_add = [
        predictor for predictor in to_add if predictor not in legacy_candidates
    ]

    dataset = create_dataset(
            frame.df["conllu"].tolist(),
            masked, masks_setting,
            transform_mask, data.TokenMapper.load(token_mapper_dir),
            use_ddp=use_ddp,
            rank=rank
        )

    # Predictors
    if model_dir[:4] == "hug:":
        model_dir = model_dir[:model_dir.rfind("_")]  # remove model number
        assert not is_undef(batch_size)
        frame.add_batched_(
            batch_size,  # type: ignore
            *to_add,
            dataset=dataset,
            masks_setting="both",
            trainer=model_dir,
            gov_name="head_current",
            dep_name="child_current",
            only_content_words_cost=only_content_words_cost,
            only_content_words_left=only_content_words_left,
            only_left=True,
            arc_distr_mode=distr_mode,
            include_current=include_current,
            length_weighted=length_weighted,
            global_distr=global_distr,
            return_arc_logits=False,
            use_ddp=False,
            rank=None)

        unsplit_frame = combine_frames(
            frame, untok_frame,
            spillover=shift,
            truncate_first=True)

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

        if is_undef(distr_mode):
            distr_mode = trainer.config.distr_mode
        if is_undef(length_weighted):
            length_weighted = trainer.config.length_weighted
        if is_undef(include_current):
            include_current = trainer.config.include_current
        if is_undef(global_distr):
            global_distr = trainer.config.global_distr
        if is_undef(batch_size):
            batch_size = trainer.config.batch_size

        frame.add_batched_(
            batch_size,  # type: ignore
            *to_add,
            dataset=dataset,
            trainer=trainer,
            masks_setting="both",
            gov_name="head_current",
            dep_name="child_current",
            only_content_words_cost=only_content_words_cost,
            only_content_words_left=only_content_words_left,
            return_att=True,
            return_proj_states=True,
            only_left=True,
            arc_distr_mode=distr_mode,
            include_current=include_current,
            length_weighted=length_weighted,
            global_distr=global_distr,
            return_arc_logits=False,
            use_ddp=False,
            rank=None)

        if len(legacy_candidates) > 0:
            if not masks_setting == "next":
                # Dependent on dependency prediction
                frame.add_(
                    *legacy_candidates,
                    only_past=True)

            if not masks_setting == "current":
                candidate_tuples = [
                    cand + "_next_col" for cand in legacy_candidates]

                frame.add_(
                    *candidate_tuples,
                    gov_name="head_next",
                    child_name="child_next",
                    masks_setting="next",
                    only_past=True)

        unsplit_frame = combine_frames(
            frame, untok_frame,
            spillover=shift
        )

        if output_file is None:
            return unsplit_frame.df
        unsplit_frame.df.to_csv(output_file, index=False)
        return None


def get_frames(
        psyling_df: pd.DataFrame | str,
        rank: int | None = None,
        token_col: str = TOKEN_COL,
        baseline_metrics: Iterable[str] = BASELINE_METRICS,
        ) -> Tuple[SplitFrame, SplitFrame]:

    if isinstance(psyling_df, str):
        psyling_df = pd.read_csv(
            psyling_df, keep_default_na=False, na_values=[''])
    else:
        psyling_df.fillna("NaN")

    orig_frame = UnsplitFrame(
            psyling_df, {"word_col": token_col}, tokenised=False)

    for metric in baseline_metrics:
        orig_frame.add_(
            metric)

    # Create conllu frame
    info(rank, logger, "Getting conllu frame...")
    tok_frame = get_conllu_frame(
        psyling_df)

    info(rank, logger, f"Identified {len(tok_frame.df)} psyling sentences.")
    # omits undefined args

    info(rank, logger, "Adjusting tokenisation (1/2)...")
    tok_untok_frame = tok_frame.untokenise()
    info(rank, logger, "Adjusting tokenisation (2/2)...")
    tok_untok_frame.adjust_untokenise_(orig_frame.df["word"])  # type: ignore
    info(rank, logger, "Adjusted tokenisation.")

    assert len(orig_frame.df) == sum(
        len(sentence) for sentence in tok_untok_frame.df["word"])

    untok_frame = orig_frame.split([
        len(sentence) for sentence in tok_untok_frame.df[
            token_col]])

    return tok_frame, untok_frame


def create_dataset(
        tokenlists: list[TokenList],
        masked: bool, masks_setting: data.MasksSetting,
        transform: data.TransformFunc | None,
        token_mapper: data.TokenMapper,
        tempdir: str = ".temp",
        temp_dataset_filename: str = "temp_dataset",
        temp_mmap_filename: str = "temp_mmap",
        only_load_mmap: bool = False,
        use_ddp: bool = False,
        rank: int | None = None,
        ) -> data.MemMapDataset | data.MemMapDepDataset:

    if not only_load_mmap:
        pathlib.Path(tempdir).mkdir(parents=True, exist_ok=True)

        # Prevent memory writes by different processes
        if not use_ddp or rank == 0:
            dataset: data.MemMapDataset | data.MemMapDepDataset

            with open(os.path.join(tempdir, "temp_dataset"), "w") as temp:
                for sentence in tokenlists:
                    temp.write(sentence.serialize())

            if masked:
                assert masks_setting is not None
                assert transform is not None
                dataset = data.MemMapDepDataset.from_file(
                    os.path.join(tempdir, temp_dataset_filename),
                    transform_masks=transform,
                    masks_setting=masks_setting,
                    max_len=None,
                    min_len=None)
            else:
                dataset = data.MemMapDataset.from_file(
                    os.path.join(tempdir, temp_dataset_filename),
                    max_len=None,
                    min_len=None)
            dataset.map_to_ids(
                token_mapper,
                os.path.join(tempdir, temp_mmap_filename))
        if use_ddp:
            dist.barrier()

    if masked:
        dataset = data.MemMapDepDataset.from_memmap(
            os.path.join(tempdir, temp_mmap_filename),
            transform_masks=transform,
            masks_setting=masks_setting,
            max_len=None,
            min_len=None)
    else:
        dataset = data.MemMapDataset.from_memmap(
            os.path.join(tempdir, temp_mmap_filename),
            max_len=None,
            min_len=None)
    return dataset


def combine_frames(
        tok_frame: SplitFrame,
        untok_frame: SplitFrame,
        spillover: int = 0,
        truncate_first: bool = False,
        truncate_right: int = 1,
        ) -> UnsplitFrame:

    # Untokenisation
    # We cannot omit this because surprisal can be a sum
    # of token surprisals.
    temp_untok_frame = tok_frame.untokenise()
    temp_untok_frame.adjust_untokenise_([
        word for sentence in untok_frame.df["word"]
        for word in sentence])  # type: ignore
    temp_untok_frame.add_("position")

    # Do this separately because in untok_frame numbers are replaced
    # with <num>. Therefore, frequency and length would not be
    # correct.
    frame = untok_frame | temp_untok_frame

    frame = frame.include_spillover(spillover)
    frame.truncate_(right=truncate_right)

    if truncate_first and spillover == 0:
        frame.truncate_(left=1)

    return frame.unsplit()
