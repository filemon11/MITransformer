from ....train.metrics import (
    MetricWriter)
from .... import readingtimes, data
from ... import parsing
from .. import train
from . import objective, sampler

import os
import optuna
import pandas as pd

from typing import Tuple

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

        self.psyling_datasets = ("naturalstories", "frank_SP")  #(arguments.psyling_dataset,)
        # TODO: check whether tokenisation at surprisal step is correct

        psyling_dfs = [
            readingtimes.io_corpus_convert(
                "custom", dataset,
                data.rt_corpus_to_text_file[dataset],
                verbose=True,
                token_mapper_dir=None)
            for dataset in self.psyling_datasets]

        self.tok_frame, self.untok_frame = get_frames(
            pd.concat(psyling_dfs))

        # Load measurements
        self.measurements = [
            readingtimes.prepare_RTs(
                data.rt_corpus_to_measurements_file[dataset],
                corpus=dataset)
            for dataset in self.psyling_datasets
        ]

        # TODO: allow multiple psyling_datasets by concatenating several
        # 'psyling_df' instances.

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

        assert self.data_provider is not None

        add_method = self.tok_frame.add_
        loglik: None | float = None
        for step, metrics in enumerate(train_iterator, start=1):
            # Handle pruning based on the intermediate value.
            transform = None

            # somehow this is ill-typed. Provider is assigned
            # dataloader by Objective parent class
            if isinstance(
                    self.data_provider.datasets[
                        "train"].dataset,  # type: ignore
                    data.MemMapDepDataset):
                transform = self.data_provider.datasets[
                    "train"].dataset.transform_mask  # type: ignore

            to_add = ["surprisal", *set(self.arguments.lme_formula[
                    "covariates"]) - set(readingtimes.BASELINE_METRICS)]
            to_add = [ta for ta in to_add if "." not in ta]
            # no spillover versions

            add_method(
                *to_add,
                masked=self.arguments.masked,
                token_mapper_dir=self.data_provider.datasets["token_mapper"],
                transform=transform, trainer=trainer,
                masks_setting=self.arguments.masks_setting,
                arc_distr_mode=self.arguments.distr_mode,
                include_current=self.arguments.include_current,
                length_weighted=self.arguments.length_weighted)
            # TODO: allow unmasked dataset to be used
            # TODO: implement candidates
            # Untokenisation
            # We cannot omit this because surprisal can be a sum
            # of token surprisals.
            untok_frame = self.tok_frame.untokenise()
            frame = self.untok_frame | untok_frame

            frame = frame.include_spillover(self.arguments.shift)
            frame.truncate_(right=1)
            unsplit_frame = frame.unsplit()

            print(unsplit_frame.df.head(n=10))

            # Joining
            # This may take some time. Should we precompute this,
            # keep track of the indices and then simply insert
            # the (duplicate) values in columns surprisal, ...?
            # TODO: check how much time this takes

            # !!! TODO: join all measurements in __init__ because
            # iterative joining with the frame does not work.
            # It disregards all Workers that do not appear in
            # the first measurement table.
            joined = unsplit_frame.df
            for dataset, measurements in zip(
                    self.psyling_datasets, self.measurements):
                joined = readingtimes.join(
                    joined,
                    measurements,
                    "ET" if dataset in readingtimes.ET_CORPORA else "SP",
                    how="left")

            # In concatenated setting, there can be several corpora per
            # item. However, we want these to be unique. Therefore:
            joined["item"] = joined["Corpus"] + joined["item"].astype(str)
            print(joined.head(n=10))

            # Fit lme
            lme, _ = readingtimes.fit_gpboost(
                joined,
                y_col=self.arguments.lme_formula["to_predict"],
                predictors=self.arguments.lme_formula["covariates"],
                random_effects=self.arguments.lme_formula["random_effects"]
            )
            # TODO: decide plausible lme structure

            # Get optimisation metric
            loglik = -readingtimes.get_model_props(
                lme, len(unsplit_frame.df))["negloglik"]

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

            add_method = self.tok_frame.reload_

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
            # TODO: add field to metric in order to report
            # TODO: also report loglik in loop above

        if should_prune:
            raise optuna.exceptions.TrialPruned()

        return loglik


def get_frames(
        psyling_df: pd.DataFrame,
        ) -> Tuple[readingtimes.SplitFrame, readingtimes.SplitFrame]:

    orig_frame = readingtimes.UnsplitFrame(
        psyling_df, {"word_col": readingtimes.TOKEN_COL}, tokenised=False)

    for metric in readingtimes.BASELINE_METRICS:
        orig_frame.add_(
            metric)

    # Create conllu frame
    tok_frame = readingtimes.get_conllu_frame(
        psyling_df)
    # omits undefined args

    untok_frame = orig_frame.split([
        len(sentence) for sentence in tok_frame.untokenise().df[
            readingtimes.TOKEN_COL]])

    return tok_frame, untok_frame
