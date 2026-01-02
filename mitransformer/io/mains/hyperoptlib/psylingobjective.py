from ....train.metrics import (
    MetricWriter)
from .... import readingtimes, data
from ... import parsing
from .. import train
from . import objective, sampler

from conllu.models import TokenList
import pathlib
import os

import optuna
import pandas as pd
import torch.distributed as dist

from typing import Tuple, Iterable, Sequence, Collection

from mitransformer.utils.logmaker import (
    getLogger, info)

logger = getLogger(__name__)

USE_LOG = {"learning_rate"}


class PsyLingObjective(objective.Objective):
    def __init__(
            self, n_devices: int,
            arguments: "parsing.HyperoptParserArgs",
            writer: MetricWriter,
            pg):
        super().__init__(n_devices, arguments, writer, pg)
        # TODO: load and join psycholinguistic data
        # Question: save and load later or keep in memory?
        info(
            arguments.rank, logger,
            "Initialised objective with lme formula "
            f"{arguments.lme_formula['formula']}")

        # Load candidates
        # TODO: check whether tokenisation at surprisal step is correct
        assert all(
            [ds in data.RTCORPORA for ds in arguments.psyling_dataset])
        psyling_df = pd.concat([
            readingtimes.io_corpus_convert(
                "custom", dataset,
                data.rt_corpus_to_text_file[dataset],
                verbose=True,
                token_mapper_dir=None,
                min_len=self.arguments.min_len_eval_test,
                max_len=self.arguments.max_len_eval_test,
                rank=self.arguments.rank)
            for dataset in arguments.psyling_dataset])
        info(self.arguments.rank, logger, "Retreaved psyling corpora.")

        self.tok_frame, self.untok_frame = get_frames(
            psyling_df, rank=self.arguments.rank)
        info(self.arguments.rank, logger, "Generated psyling dataframes.")

        transform = None
        assert self.data_provider is not None
        if isinstance(
                self.data_provider.datasets[
                    "train"].dataset,  # type: ignore
                data.MemMapDepDataset):
            transform = self.data_provider.datasets[
                "train"].dataset.transform_mask  # type: ignore

        # Create memmaped dataset for psyling data
        load_kwargs: dict[str, bool | str] = {}
        if arguments.load_psyling_mmap is not None:
            load_kwargs["only_load_mmap"] = True
            load_kwargs["temp_mmap_filename"] = arguments.load_psyling_mmap
        self.dataset = create_dataset(
            self.tok_frame.df["conllu"].tolist(),
            self.arguments.masked, self.arguments.masks_setting,
            transform, self.data_provider.datasets["token_mapper"],
            **load_kwargs  # type: ignore
        )
        info(
            self.arguments.rank, logger,
            "Loaded psyling dataset for generating surprisal.")

        # Load measurements
        is_et_corpus = [
            ds in data.ET_CORPORA for ds in arguments.psyling_dataset]
        assert all(is_et_corpus) or not any(is_et_corpus), (
            "Psyling corpora must be all of the same type (either ET or SP)."
        )
        self.is_et_corpus: bool = is_et_corpus[0]
        self.measurements = get_measurements(
            self.is_et_corpus, arguments.psyling_dataset, psyling_df
        )
        info(
            self.arguments.rank, logger,
            "Loaded psyling measurements.")

        # # Is this necessary?
        # self.measurements[self.arguments.lme_formula["to_predict"]] = np.log(
        #     self.measurements[self.arguments.lme_formula["to_predict"]])
        # # Is this necessary to account for per-corpus differences?
        # self.measurements[self.arguments.lme_formula["to_predict"]] = (
        #     self.measurements
        #     .groupby("Corpus")[self.arguments.lme_formula["to_predict"]]
        #     .transform(lambda x: (x - x.mean()) / x.std()))

    def __call__(self, trial) -> float:
        if self.n_devices > 1:
            trial = optuna.integration.TorchDistributedTrial(
                trial, self.pg)  # type: ignore

        arguments = parsing.TrainParserArgs.from_kwargs(**{
            name: sampler.hyperopt_arguments_sampler(name, arg, trial) for
            name, arg in self.arguments.to_dict().items()},
            model_name=f"{self.arguments.name}_{trial.number}",
            n_runs=1)
        arguments.seed = arguments.seed + trial.number
        parsing.args_logic(arguments)

        trainer, train_iterator = train.main_train(
            arguments, self.n_devices,
            iterate=True,
            data_provider=self.data_provider)
        assert train_iterator is not None

        should_prune = False
        metrics = None
        step = 0

        add_method = self.tok_frame.add_batched_
        loglik: None | float = None
        for step, metrics in enumerate(train_iterator, start=1):

            # Handle pruning based on the intermediate value.

            to_add = ["surprisal", *set(self.arguments.lme_formula[
                "covariates"]) - set(
                    ["surprisal", *readingtimes.BASELINE_METRICS])]
            to_add = [ta for ta in to_add if "." not in ta]
            # no spillover versions

            add_method(
                self.arguments.batch_size,
                *to_add,
                dataset=self.dataset,
                trainer=trainer,
                arc_distr_mode=self.arguments.distr_mode,
                include_current=self.arguments.include_current,
                length_weighted=self.arguments.length_weighted,
                return_arc_logits=False,)
            # TODO: allow unmasked dataset to be used

            # Untokenisation
            # We cannot omit this because surprisal can be a sum
            # of token surprisals.
            frame: readingtimes.SplitFrame | readingtimes.UnsplitFrame
            untok_frame = self.tok_frame.untokenise()
            untok_frame.adjust_untokenise_([
                word for sentence in self.untok_frame.df["word"]
                for word in sentence])  # type: ignore

            # Do this separately because in untok_frame numbers are replaced
            # with <num>. Therefore, frequency and length would not be
            # correct.
            frame = self.untok_frame | untok_frame

            frame = frame.include_spillover(self.arguments.shift)
            frame.truncate_(right=1)
            frame = frame.unsplit()

            print(frame.df[["surprisal", "item", "zone"]].tail(n=10))

            # Joining
            # This may take some time. Should we precompute this,
            # keep track of the indices and then simply insert
            # the (duplicate) values in columns surprisal, ...?
            # TODO: check how much time this takes

            joined = readingtimes.join(
                frame.df,
                self.measurements,
                "ET" if self.is_et_corpus else "SP")
            del frame

            # In concatenated setting, there can be several corpora per
            # item and several corpora per WorkerId. However, we want these
            # to be unique because they come from different corpora. Therefore:
            joined["item"] = joined["Corpus"] + joined["item"].astype(str)
            joined["WorkerId"] = joined["Corpus"] + joined["WorkerId"]

            # # Remove outliers
            joined = remove_outliers(
                joined, self.arguments.lme_formula["covariates"],
                self.arguments.rank
            )
            # # Scale predictors
            z_score_(
                joined, self.arguments.lme_formula["covariates"]
            )

            relevant = [
                self.arguments.lme_formula["to_predict"],
                *self.arguments.lme_formula["covariates"],
                *self.arguments.lme_formula["random_effects"],]
            joined = joined[relevant]
            joined.dropna(inplace=True)

            # TODO: fir separately for each corpus and
            # take mean per row loglik
            # Fit lme
            lme, d0 = readingtimes.fit_gpboost(
                joined,
                y_col=self.arguments.lme_formula["to_predict"],
                predictors=self.arguments.lme_formula["covariates"],
                random_effects=self.arguments.lme_formula["random_effects"]
            )
            del joined

            model_props = readingtimes.get_model_props(lme, len(d0))
            info(
                arguments.rank, logger,
                readingtimes.model_props_to_str(
                    model_props,
                    ["Intercept"] + list(
                        self.arguments.lme_formula["covariates"]),
                    ["Error_term"]
                    + [
                        group for group, coefs in
                        self.arguments.lme_formula["random_effects"].items()
                        if 1 in coefs]
                    + [
                        f"{group}_{c}"
                        for group, coefs
                        in self.arguments.lme_formula["random_effects"].items()
                        for c in coefs if c not in (1, 0)]
                )
            )
            del lme

            # TODO: decide plausible lme structure

            # Get optimisation metric
            loglik = -model_props["negloglik"]

            # Add to metric writer
            trainer.writer.custom_add_scalar(
                "loglik", loglik, step, "psyling_eval")

            info(
                arguments.rank, logger,
                f"Psyling eval loglik: {loglik}")
            trial.report(
                loglik,
                step)

            if trial.should_prune():
                should_prune = True
                break

            add_method = self.tok_frame.reload_batched_

        assert loglik is not None and metrics is not None, (
            "eval_interval is larger than total number of steps")

        arg_dict = arguments.to_dict()
        additional_dict = {}
        for key, value in arg_dict.items():
            if isinstance(value, dict):
                for inner_key, inner_val in value.items():
                    additional_dict[f"{key}_{inner_key}"] = inner_val
        arg_dict |= additional_dict

        if self.writer is not None:
            self.writer.add_params(
                arg_dict,
                {
                    "loglik": loglik,
                    **metrics["eval"].to_dict()},
                run_name=str(trial.number),
                global_step=arguments.eval_interval*step)

        if should_prune:
            raise optuna.exceptions.TrialPruned()

        return loglik


def get_frames(
        psyling_df: pd.DataFrame,
        rank: int | None = None
        ) -> Tuple[readingtimes.SplitFrame, readingtimes.SplitFrame]:

    orig_frame = readingtimes.UnsplitFrame(
        psyling_df, {"word_col": readingtimes.TOKEN_COL}, tokenised=False)

    for metric in readingtimes.BASELINE_METRICS:
        orig_frame.add_(
            metric)

    # Create conllu frame
    info(rank, logger, "Getting conllu frame...")
    tok_frame = readingtimes.get_conllu_frame(
        psyling_df)
    # omits undefined args

    info(rank, logger, "Adjusting tokenisation (1/2)...")
    tok_untok_frame = tok_frame.untokenise()
    info(rank, logger, "Adjusting tokenisation (2/2)...")
    tok_untok_frame.adjust_untokenise_(orig_frame.df["word"])  # type: ignore
    info(rank, logger, "Adjusted tokenisation.")

    assert len(orig_frame.df) == sum(
        len(sentence) for sentence in tok_untok_frame.df["word"])

    # TODO: is this necessary?
    # include = [
    # len(sentence) > 0
    # for sentence in tok_untok_frame.df[readingtimes.TOKEN_COL]]
    # tok_untok_frame.df = tok_untok_frame.df[include].reset_index(drop=True)

    untok_frame = orig_frame.split([
        len(sentence) for sentence in tok_untok_frame.df[
            readingtimes.TOKEN_COL]])

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
        rank: int = 0,
        ) -> data.MemMapDataset | data.MemMapDepDataset:

    if not only_load_mmap:
        pathlib.Path(tempdir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(tempdir, "temp_dataset"), "w") as temp:
            for sentence in tokenlists:
                temp.write(sentence.serialize())

        # Prevent memory writes by different processes
        if not use_ddp or rank == 0:
            dataset: data.MemMapDataset | data.MemMapDepDataset
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


def get_measurements(
        is_et_corpus: bool,
        psyling_datasets: Iterable[data.RTCorpus],
        psyling_df: pd.DataFrame | None = None,
        merge_on: Sequence[str] = ["Corpus", "item", "zone"]
        ) -> pd.DataFrame:
    measurement_keys = [
        "FFD", "GPT", "GD"] if is_et_corpus else ["RT"]
    measurement_keys.extend(["word", "item", "zone", "WorkerId", "Corpus"])
    measurements = [
        data.prepare_RT_measurements(
            data.rt_corpus_to_measurements_file[dataset],
            corpus=dataset)[measurement_keys]
        for dataset in psyling_datasets
    ]
    measurements_df = pd.concat(measurements)
    # remove items that are not needed (i.e. that won't be joined on later)
    if psyling_df is not None:
        measurements_df = measurements_df.merge(
            psyling_df[merge_on],
            on=merge_on, how="inner")
    return measurements_df


def z_score_(
        df: pd.DataFrame, cols: Collection[str]) -> None:
    cols = list(cols)
    df[cols] = (
        df[cols]
        - df[cols].mean()) / df[cols].std()


def z_score_numerical_(
        df: pd.DataFrame, exception: Iterable[str]) -> None:
    num_cols = list(
        set(df.select_dtypes(include="number").columns)
        - set(exception))
    z_score_(df, num_cols)


def remove_outliers(
        df: pd.DataFrame,
        cols: Collection[str], rank: int | None = None
        ) -> pd.DataFrame:
    cols = list(cols)
    q1 = df[cols].quantile(0.25)
    q3 = df[cols].quantile(0.75)
    iqr = q3 - q1
    mask = (
        (df[cols] < (q1 - 1.5 * iqr))
        | (df[cols] > (q3 + 1.5 * iqr))).any(axis=1)
    info(rank, logger, f"Removed {sum(mask)} out of {len(df)} rows.")
    return df[~mask]


def remove_outliers_numerical(
        df: pd.DataFrame,
        exception: Iterable[str], rank: int | None = None
        ) -> pd.DataFrame:
    num_cols = list(
        set(df.select_dtypes(include="number").columns)
        - set(exception))
    return remove_outliers(df, num_cols, rank)
