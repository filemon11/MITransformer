from ....train.metrics import (
    MetricWriter)
from .... import readingtimes, data
from ... import parsing
from .. import train
from . import objective, sampler

import optuna

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
        psyling_df = readingtimes.io_corpus_convert(
            "custom", arguments.psyling_dataset,
            data.rt_corpus_to_text_file[arguments.psyling_dataset])
        psyling_df.fillna("NaN")

        orig_frame = readingtimes.UnsplitFrame(
            psyling_df, {"word_col": readingtimes.TOKEN_COL}, tokenised=False)

        for metric in readingtimes.BASELINE_METRICS:
            orig_frame.add_(metric)

        # Create conllu frame
        self.frame = readingtimes.get_conllu_frame(
            psyling_df, arguments.psyling_dataset)
        # omits undefined args

        self.split_frame = orig_frame.split([
                len(sentence) for sentence in self.frame.untokenise().df[
                    readingtimes.TOKEN_COL]])

        # Load measurements
        self.measurements = readingtimes.prepare_RTs(
            data.rt_corpus_to_measurements_file[arguments.psyling_dataset],
            corpus=arguments.psyling_dataset)

        self.lme_formula = arguments.lme_formula

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

            self.frame.add_(
                *(self.lme_formula[
                    "covariates"] - set(readingtimes.BASELINE_METRICS)),
                masked=self.arguments.masked,
                token_mapper_dir=self.data_provider.datasets["token_mapper"],
                transform=transform, trainer=trainer,
                masks_setting=self.arguments.masks_setting)
            # TODO: allow unmasked dataset to be used
            # TODO: implement candidates

            # Untokenisation
            # We cannot omit this because surprisal can be a sum
            # of token surprisals.
            untok_frame = self.frame.untokenise()
            frame = self.split_frame | untok_frame
            frame = frame.include_spillover(self.arguments.shift)
            frame.truncate_(right=1)
            unsplit_frame = frame.unsplit()

            # Joining
            # This may take some time. Should we precompute this,
            # keep track of the indices and then simply insert
            # the (duplicate) values in columns surprisal, ...?
            # TODO: check how much time this takes
            joined = readingtimes.join(
                self.measurements,
                unsplit_frame.df,
                "ET" if self.arguments.psyling_dataset
                in readingtimes.ET_CORPORA else "SP")

            # Fit lme
            lme, _ = readingtimes.fit_gpboost(
                joined,
                y_col=self.lme_formula["to_predict"],
                predictors=self.lme_formula["covariates"],
                random_effects=self.lme_formula["random_effects"]
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

            self.frame.reload_("surprisal")

        assert loglik is not None and metrics is not None, (
            "eval_interval is larger than total number of steps")
        if self.writer is not None:
            self.writer.add_params(
                arguments.to_dict(),
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
