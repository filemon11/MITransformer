from ....train.metrics import (
    MetricWriter)
from .... import readingtimes
from ... import parsing
from .. import train
from . import objective, sampler

import optuna
import pandas as pd

from mitransformer.utils.logmaker import (
    getLogger)

logger = getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


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

        psyling_file = "TODO"
        self.corpus: readingtimes.Corpus = "naturalstories"
        self.token_mapper_dir = ""
        self.shift = 1

        # Load candidates
        psyling_df = readingtimes.io_corpus_convert(
            "custom", self.corpus, psyling_file)
        psyling_df.fillna("NaN")

        orig_frame = readingtimes.UnsplitFrame(
            psyling_df, {"word_col": readingtimes.TOKEN_COL}, tokenised=False)

        for metric in readingtimes.BASELINE_METRICS:
            orig_frame.add_(metric)

        # Create conllu frame
        self.frame = readingtimes.get_conllu_frame(
            psyling_df, self.corpus)
        # omits undefined args

        self.split_frame = orig_frame.split([
            len(sentence) for sentence
            in self.frame.df[readingtimes.TOKEN_COL]])

        # Load measurements
        corpus_to_rt_infile: dict[readingtimes.Corpus, str] = {
            "naturalstories": "RT/data/processed_RTs.tsv",
            "zuco": "zuco/training_data.csv",
            "frank_ET": "frank/eyetracking.RT.txt",
            "frank_SP": "frank/selfpacedreading.RT.txt"
        }
        self.measurements = readingtimes.prepare_RTs(
            corpus_to_rt_infile[self.corpus],
            corpus=self.corpus)

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
        for step, metrics in enumerate(train_iterator, start=1):
            # Handle pruning based on the intermediate value.
            self.frame.add_(
                "surprisal", token_mapper_dir=self.token_mapper_dir,
                transform=None, trainer=trainer, masks_setting=None)
            candidates = (
                "attention_entropy_new",
                "attention_distance"
            )
            # TODO: implement candidates
            self.frame.add_(
                *candidates,
                only_past=True)

            # Untokenisation
            # We cannot omit this because surprisal can be a sum
            # of token surprisals.
            self.frame.untokenise_()
            frame = self.split_frame | self.frame
            frame = frame.include_spillover(self.shift)
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
                "ET" if self.corpus in readingtimes.ET_CORPORA else "SP")

            # Fit lme
            y_col = "RT" if self.corpus in readingtimes.SP_CORPORA else "GPT"
            lme, _ = readingtimes.fit_gpboost(
                joined,
                y_col=y_col,
                predictors=[*readingtimes.BASELINE_METRICS, *candidates],
                random_effects={
                    "WorkerId": [*readingtimes.BASELINE_METRICS, *candidates],
                    "item": [1]}  # item or zone?
            )
            # TODO: decide plausible lme structure

            # Get optimisation metric
            negloglik = readingtimes.get_model_props(
                lme, len(unsplit_frame.df))["negloglik"]

            trial.report(
                negloglik,
                step)

            if trial.should_prune():
                should_prune = True
                break

        assert metrics is not None, (
            "eval_interval is larger than total number of steps")
        if self.writer is not None:
            self.writer.add_params(
                arguments.to_dict(),
                metrics["eval"],
                run_name=str(trial.number),
                global_step=arguments.eval_interval*step)

        if should_prune:
            raise optuna.exceptions.TrialPruned()
        # trial.set_user_attr("metric_dicts", metric_dicts)

        opt_metric = getattr(metrics["eval"], self.arguments.optimise.lower())
        if isinstance(opt_metric, pd.DataFrame):
            opt_metric = opt_metric.to_numpy().sum()
        loss: float = float(opt_metric)
        return loss