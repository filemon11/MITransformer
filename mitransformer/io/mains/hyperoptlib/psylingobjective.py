from ....train.metrics import (
    MetricWriter)
from .... import readingtimes, data
from ... import parsing
from .. import train
from . import objective, sampler

import optuna  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from collections import abc
from typing import Iterable, Sequence, Collection, Any, Literal

# from torch.profiler import profile, ProfilerActivity, record_function

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
        formulas = [f["formula"] for f in arguments.lme_formula]
        info(
            arguments.rank, logger,
            "Initialised objective with lme formulas "
            f"{' '.join(formulas)}")

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
                rank=self.arguments.rank,
                remove_unk=True)
            for dataset in arguments.psyling_dataset])
        info(self.arguments.rank, logger, "Retreaved psyling corpora.")

        self.tok_frame, self.untok_frame = readingtimes.get_frames(
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
        self.dataset = readingtimes.create_dataset(
            self.tok_frame.df["conllu"].tolist(),
            self.arguments.masked, self.arguments.masks_setting,
            transform, self.data_provider.datasets["token_mapper"],
            use_ddp=self.arguments.use_ddp,
            rank=self.arguments.rank,
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
        tracker = LengthTracker(self.measurements, rank=arguments.rank)

        # remove nans
        self.measurements = self.measurements[
            ~self.measurements[
                self.arguments.lme_formula[0]["to_predict"]].isna()]
        tracker.update(self.measurements, "Removed na observations:")

        # Apply cutoffs
        self.measurements = apply_cutoff(
            self.measurements, self.arguments.lme_formula[0]["to_predict"])
        tracker.update(self.measurements, "Cut off observations:")

        # Apply outlier removal
        self.measurements = remove_outliers_per_group(
            self.measurements, (self.arguments.lme_formula[0]["to_predict"],),
            "WorkerId")
        tracker.update(self.measurements, "Applied outlier removal:")

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
        best_loglik: float = -np.inf

        best_eval_metric: train.LMMetric | float | None = None
        # gen = iter(enumerate(train_iterator, start=1))
        for step, metrics in enumerate(train_iterator, start=1):
            # for step, metrics in enumerate(train_iterator, start=1):
            # with profile(activities=[ProfilerActivity.CUDA],
            #   record_shapes=True) as prof:
            #     with record_function("training"):
            #        step, metrics = next(gen)
            if best_eval_metric is None:
                best_eval_metric = metrics["eval"].minval()
            if metrics["eval"] > best_eval_metric:
                best_eval_metric = metrics["eval"]

            # Handle pruning based on the intermediate value.

            to_add = [
                "unknown", "surprisal", *set(self.arguments.lme_formula[0][
                    "covariates"]) - set(
                    ["0", "surprisal", *readingtimes.BASELINE_METRICS])]
            to_add = [ta for ta in to_add if "." not in ta]
            # no spillover versions

            add_method(
                self.arguments.batch_size,
                *to_add,
                dataset=self.dataset,
                trainer=trainer,
                arc_distr_mode=self.arguments.distr_mode,
                include_current=self.arguments.include_current,
                global_distr=self.arguments.global_distr,
                length_weighted=self.arguments.length_weighted,
                return_arc_logits=False,
                use_ddp=False,
                rank=None,
                unk_id=self.data_provider.datasets[  # type: ignore
                    "token_mapper"].unk_id)
            # TODO: allow unmasked dataset to be used

            # Untokenisation
            frame: readingtimes.UnsplitFrame
            frame = readingtimes.combine_frames(
                self.tok_frame, self.untok_frame,
                spillover=self.arguments.shift
            )

            # if self.arguments.rank is None or self.arguments.rank == 0:
            #     print(frame.df.head(n=10))

            # Joining
            # This may take some time. Should we precompute this,
            # keep track of the indices and then simply insert
            # the (duplicate) values in columns surprisal, ...?
            # TODO: check how much time this takes

            joined = readingtimes.join(
                frame.df,
                self.measurements,
                "ET" if self.is_et_corpus else "SP",
                rank=self.arguments.rank,
                only_interest=False)
            del frame

            # In concatenated setting, there can be several corpora per
            # item and several corpora per WorkerId. However, we want these
            # to be unique because they come from different corpora. Therefore:
            joined["item"] = joined["Corpus"] + joined["item"].astype(str)
            joined["WorkerId"] = joined["Corpus"] + joined["WorkerId"]

            tracker = LengthTracker(joined, rank=arguments.rank)
            joined = apply_cutoff(
                joined, "frequency",
                lowerOther=2, upperOther=np.inf)
            tracker.update(joined, "Removed observations with freq<2:")

            joined = drop_sentences(joined)
            tracker.update(
                joined,
                "Removed observations (sentences with words "
                "unknown to tokeniser):")

            if self.arguments.average_psyling:
                z_score_per_group_(
                    joined, self.arguments.lme_formula[0]["covariates"],
                    "Corpus"
                )
            else:
                z_score_(
                    joined, self.arguments.lme_formula[0]["covariates"],
                )

            # Fit lme
            if self.arguments.average_psyling:
                measures: list[float] = []
                for corpus in joined["Corpus"].unique():
                    joined_corpus = joined[joined["Corpus"] == corpus]
                    info(
                        arguments.rank, logger,
                        f"Number of observations: {len(joined_corpus)}")
                    lme, d0 = readingtimes.fit_gpboost(
                        joined_corpus,
                        y_col=self.arguments.lme_formula[0]["to_predict"],
                        predictors=self.arguments.lme_formula[0]["covariates"],
                        random_effects=self.arguments.lme_formula[0][
                            "random_effects"]
                    )

                    model_props = readingtimes.get_model_props(lme, len(d0))
                    info(self.arguments.rank, logger, f"---{corpus}---")
                    print_lme_info(
                        model_props, self.arguments.lme_formula[0],
                        self.arguments.rank
                    )
                    del lme
                    loglik = -model_props["negloglik_per_row"]
                    measures.append(loglik)

                    trainer.writer.custom_add_scalar(
                        f"loglik_{corpus}", loglik, step, "psyling_eval")

                    info(
                        arguments.rank, logger,
                        f"Psyling eval loglik {corpus}: {loglik}")

                # TODO: decide plausible lme structure

                # Get optimisation metric
                assert len(measures) > 0
                loglik = sum(measures) / len(measures)

            else:
                info(
                    arguments.rank, logger,
                    f"Number of observations: {len(joined)}")
                lme, d0 = readingtimes.fit_gpboost(
                    joined,
                    y_col=self.arguments.lme_formula[0]["to_predict"],
                    predictors=self.arguments.lme_formula[0]["covariates"],
                    random_effects=self.arguments.lme_formula[
                        0]["random_effects"]
                )

                model_props = readingtimes.get_model_props(lme, len(d0))
                print_lme_info(
                    model_props, self.arguments.lme_formula[0],
                    self.arguments.rank
                )
                del lme
                loglik = -model_props["negloglik_per_row"]

            if loglik > best_loglik:
                best_loglik = loglik
            del joined

            # Add to metric writer
            trainer.writer.custom_add_scalar(
                "loglik", loglik, step, "psyling_eval")

            info(
                arguments.rank, logger,
                f"Psyling eval loglik: {loglik}")

            if self.arguments.optimise == "loglik":
                trial.report(
                    loglik,
                    step)
            else:
                opt_metric = getattr(
                    best_eval_metric, self.arguments.optimise.lower())
                if isinstance(opt_metric, pd.DataFrame):
                    opt_metric = float(opt_metric.to_numpy().sum())
                trial.report(
                    opt_metric,
                    step)

            if trial.should_prune():
                should_prune = True
                break

            add_method = self.tok_frame.reload_batched_

            # print(prof.key_averages().table(row_limit=1000))

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
            if self.arguments.optimise == "loglik":
                self.writer.add_params(
                    arg_dict,
                    {
                        "loglik": loglik,
                        **metrics["eval"].to_dict()},
                    run_name=str(trial.number),
                    global_step=arguments.eval_interval*step)
            else:
                self.writer.add_params(
                    arguments.to_dict(),
                    metrics["eval"],
                    run_name=str(trial.number),
                    global_step=arguments.eval_interval*step)

        if should_prune:
            raise optuna.exceptions.TrialPruned()

        if self.arguments.optimise == "loglik":
            return best_loglik

        else:
            opt_metric = getattr(
                best_eval_metric, self.arguments.optimise.lower())
            if isinstance(opt_metric, pd.DataFrame):
                opt_metric = opt_metric.to_numpy().sum()
            return float(opt_metric)


def get_measurements(
        is_et_corpus: bool,
        psyling_datasets: Iterable[data.RTCorpus],
        psyling_df: pd.DataFrame | None = None,
        merge_on: Sequence[str] = ["Corpus", "item", "zone"],
        only_interest: bool = False,
        ) -> pd.DataFrame:

    measurements = [
        data.prepare_RT_measurements(
            data.rt_corpus_to_measurements_file[dataset],
            corpus=dataset,
            only_interest=only_interest)
        for dataset in psyling_datasets
    ]

    if only_interest:
        measurement_keys = [
            "FFD", "GPT", "GD"] if is_et_corpus else ["RT"]
        measurement_keys.extend(
            ["word", "item", "zone", "element", "WorkerId", "Corpus"])
        measurements = [dataset[measurement_keys] for dataset in measurements]

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


def z_score_per_group_(
        df: pd.DataFrame, cols: Collection[str],
        group_col: str) -> None:
    cols = list(cols)
    df[cols] = (
        df
        .groupby(group_col)[cols]
        .transform(lambda x: (x - x.mean()) / x.std()))


def drop_sentences(
        data: pd.DataFrame, goal: str = "unknown",
        value: Any = True, sent_col: str = "item"):
    mask = data[goal].astype(str).eq("True")
    bad_sentences = data.loc[mask, sent_col].unique()

    data = data.loc[~data[sent_col].isin(bad_sentences)].copy()

    return data


def z_score_numerical_per_group_(
        df: pd.DataFrame, exception: Iterable[str],
        group_col: str) -> None:
    num_cols = list(
        set(df.select_dtypes(include="number").columns)
        - set(exception))
    z_score_per_group_(df, num_cols, group_col)


def remove_outliers_per_group(
        df: pd.DataFrame,
        cols: Collection[str],
        group_col: str,
        k: float = 2.5,
        by: None | str | Sequence[str] = None,
        ) -> pd.DataFrame:
    needed = [group_col, *cols]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if by is None:
        by_cols: list[str] = []
    elif isinstance(by, str):
        by_cols = [by]
    else:
        by_cols = list(by)

    missing_by = [c for c in by_cols if c not in df.columns]
    if missing_by:
        raise KeyError(f"Missing grouping columns in `by`: {missing_by}")

    grp_cols = [group_col] + by_cols

    keep = pd.Series(True, index=df.index)

    # For each trimmed column, compute group-wise mean/sd and update keep-mask
    for col in cols:
        x = df[col]

        mu = df.groupby(
            grp_cols, dropna=False)[col].transform("mean")
        sd = df.groupby(
            grp_cols, dropna=False)[col].transform(lambda s: s.std(ddof=1))

        # If sd is 0 or NaN, do not trim for that row/column
        # (i.e., keep = True for this column)
        valid_sd = sd.notna() & (sd != 0)

        lower = mu - k * sd
        upper = mu + k * sd

        in_range = x.isna() | (~valid_sd) | ((x >= lower) & (x <= upper))
        keep &= in_range

    out = df.loc[keep].copy().reset_index(drop=True)

    return out


def apply_cutoff(
        data: pd.DataFrame,
        goal: str,
        lowerRT: float = 200,
        upperRT: float = 2000,
        lowerGPT: float = 80,
        upperGPT: float = 3000,
        lowerOther: float = 80,
        upperOther: float = 1000,
        ) -> pd.DataFrame:
    if goal == "RT":
        out = data.loc[(data["RT"] > lowerRT) & (data["RT"] < upperRT)].copy()
    elif goal == "GPT":
        if goal not in data.columns:
            raise KeyError(f"Column '{goal}' not found in dataframe.")
        out = data.loc[
            (data[goal] > lowerGPT) & (data[goal] < upperGPT)].copy()
    else:
        if goal not in data.columns:
            raise KeyError(f"Column '{goal}' not found in dataframe.")
        out = data.loc[
            (data[goal] > lowerOther) & (data[goal] < upperOther)].copy()

    return out


def print_lme_info(
        model_props: readingtimes.ModelProps,
        lme_formula: readingtimes.ParseResult,
        rank: int | None = None):
    info(
        rank, logger,
        readingtimes.model_props_to_str(
            model_props,
            ["Intercept"] + list(
                lme_formula["covariates"]),
            ["Error_term"]
            + [
                group for group, coefs in
                lme_formula[
                    "random_effects"].items()
                if 1 in coefs]
            + [
                f"{group}_{c}"
                for group, coefs
                in lme_formula[
                    "random_effects"].items()
                for c in coefs if c not in (1, 0)]
        ))


class LengthTracker():
    def __init__(
            self, df: abc.Sized | None = None,
            rank: None | int = None,
            msg: None | str = "Initial length:"):
        self.lengths: list[int] = []
        self.rank = rank

        if df is not None:
            self.lengths.append(len(df))
            if msg is not None:
                info(
                    self.rank, logger,
                    f"{msg} {self.lengths[-1]}")

    def update(
            self, df: abc.Sized, msg: str | None = None,
            mode: Literal["a", "r"] = "a") -> None:
        self.lengths.append(len(df))
        if msg is not None and len(self.lengths) > 1:
            factor = -1 if mode == "r" else 1
            info(
                self.rank, logger,
                f"{msg} {factor*(self.lengths[-1]-self.lengths[-2])}")
