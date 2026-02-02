from ....train.metrics import (
    MetricWriter)
from .... import readingtimes, data
from ... import parsing
from .. import train
from . import objective, sampler

import optuna
import pandas as pd

from typing import Iterable, Sequence, Collection

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
                global_distr=self.arguments.global_distr,
                length_weighted=self.arguments.length_weighted,
                return_arc_logits=False,
                use_ddp=False,
                rank=None)
            # TODO: allow unmasked dataset to be used

            # Untokenisation
            frame: readingtimes.UnsplitFrame
            frame = readingtimes.combine_frames(
                self.tok_frame, self.untok_frame,
                spillover=self.arguments.shift
            )

            if self.arguments.rank is None or self.arguments.rank == 0:
                print(frame.df.head(n=10))

            # Joining
            # This may take some time. Should we precompute this,
            # keep track of the indices and then simply insert
            # the (duplicate) values in columns surprisal, ...?
            # TODO: check how much time this takes

            joined = readingtimes.join(
                frame.df,
                self.measurements,
                "ET" if self.is_et_corpus else "SP",
                rank=self.arguments.rank)
            del frame

            # In concatenated setting, there can be several corpora per
            # item and several corpora per WorkerId. However, we want these
            # to be unique because they come from different corpora. Therefore:
            joined["item"] = joined["Corpus"] + joined["item"].astype(str)
            joined["WorkerId"] = joined["Corpus"] + joined["WorkerId"]

            relevant = {
                "Corpus",
                self.arguments.lme_formula["to_predict"],
                *self.arguments.lme_formula["covariates"],
                *self.arguments.lme_formula["random_effects"],
            }
            joined = joined[list(relevant)]
            joined.dropna(inplace=True)

            if self.arguments.average_psyling:
                # # Remove outliers
                # joined = remove_outliers_per_group(
                #     joined, self.arguments.lme_formula["covariates"],
                #     "Corpus", rank=self.arguments.rank
                # )
                # # Scale predictors
                z_score_per_group_(
                    joined, self.arguments.lme_formula["covariates"],
                    "Corpus"
                )
            else:
                # # Remove outliers
                # joined = remove_outliers(
                #     joined, self.arguments.lme_formula["covariates"],
                #     rank=self.arguments.rank
                # )
                # # Scale predictors
                z_score_(
                    joined, self.arguments.lme_formula["covariates"],
                )
            # Fit lme
            if self.arguments.average_psyling:
                measures: list[float] = []
                for corpus in joined["Corpus"].unique():
                    lme, d0 = readingtimes.fit_gpboost(
                        joined[joined["Corpus"] == corpus],
                        y_col=self.arguments.lme_formula["to_predict"],
                        predictors=self.arguments.lme_formula["covariates"],
                        random_effects=self.arguments.lme_formula[
                            "random_effects"]
                    )

                    model_props = readingtimes.get_model_props(lme, len(d0))
                    info(self.arguments.rank, logger, f"---{corpus}---")
                    print_lme_info(
                        model_props, self.arguments.lme_formula,
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
                lme, d0 = readingtimes.fit_gpboost(
                    joined,
                    y_col=self.arguments.lme_formula["to_predict"],
                    predictors=self.arguments.lme_formula["covariates"],
                    random_effects=self.arguments.lme_formula["random_effects"]
                )

                model_props = readingtimes.get_model_props(lme, len(d0))
                print_lme_info(
                    model_props, self.arguments.lme_formula,
                    self.arguments.rank
                )
                del lme
                loglik = -model_props["negloglik_per_row"]

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
                    metrics["eval"], self.arguments.optimise.lower())
                if isinstance(opt_metric, pd.DataFrame):
                    opt_metric = float(opt_metric.to_numpy().sum())
                trial.report(
                    opt_metric,
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
            return loglik

        else:
            opt_metric = getattr(
                metrics["eval"], self.arguments.optimise.lower())
            if isinstance(opt_metric, pd.DataFrame):
                opt_metric = opt_metric.to_numpy().sum()
            return float(opt_metric)


def get_measurements(
        is_et_corpus: bool,
        psyling_datasets: Iterable[data.RTCorpus],
        psyling_df: pd.DataFrame | None = None,
        merge_on: Sequence[str] = ["Corpus", "item", "zone"]
        ) -> pd.DataFrame:
    measurement_keys = [
        "FFD", "GPT", "GD"] if is_et_corpus else ["RT"]
    measurement_keys.extend(
        ["word", "item", "zone", "element", "WorkerId", "Corpus"])
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


def z_score_per_group_(
        df: pd.DataFrame, cols: Collection[str],
        group_col: str) -> None:
    cols = list(cols)
    df[cols] = (
        df
        .groupby(group_col)[cols]
        .transform(lambda x: (x - x.mean()) / x.std()))


def z_score_numerical_per_group_(
        df: pd.DataFrame, exception: Iterable[str],
        group_col: str) -> None:
    num_cols = list(
        set(df.select_dtypes(include="number").columns)
        - set(exception))
    z_score_per_group_(df, num_cols, group_col)


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


def remove_outliers_per_group(
        df: pd.DataFrame,
        cols: Collection[str],
        group_col: str,
        rank: int | None = None
        ) -> pd.DataFrame:
    cols = list(cols)
    grouped = df.groupby(group_col)

    q1 = grouped[cols].transform("quantile", 0.25)
    q3 = grouped[cols].transform("quantile", 0.75)
    iqr = q3 - q1
    mask = (
        (df[cols] < (q1 - 1.5 * iqr))
        | (df[cols] > (q3 + 1.5 * iqr))).any(axis=1)

    removed_per_group = (
        df.loc[mask]
        .groupby(group_col)
        .size()
    )
    total_per_group = (
        grouped
        .size()
    )

    for (grp, n), (_, t) in zip(
            removed_per_group.items(), total_per_group.items()):
        info(rank, logger, f"Group '{grp}': removed {n} out of {t} rows")
    info(rank, logger, f"Removed {sum(mask)} out of {len(df)} rows total.")
    df = df[~mask]
    return df


def remove_outliers_numerical(
        df: pd.DataFrame,
        exception: Iterable[str], rank: int | None = None
        ) -> pd.DataFrame:
    num_cols = list(
        set(df.select_dtypes(include="number").columns)
        - set(exception))
    return remove_outliers(df, num_cols, rank)


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
