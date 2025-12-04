from ...data import DataProvider
from ...train import (
    LMMetric, LMTrainer)
from ...utils.params import is_undef
from ...train.hooks import TreePlotHook, AttentionPlotHook
from .. import parsing
from . import functions
import os

from mitransformer.utils.logmaker import (
    getLogger, info)

logger = getLogger(__name__)


def main_test(
        arguments: "parsing.TestParserArgs",
        world_size: int,
        data_provider: DataProvider | None = None
        ) -> tuple[LMMetric, LMMetric, LMMetric]:
    """Calculates the mean and standard deviation of several
    runs."""

    # device: where to execute computation
    if world_size > 1:
        assert arguments.rank is not None, (
            "Rank cannot be None if word_size > 1.")

    if is_undef(arguments.dependency_mode):
        trainer = LMTrainer.load(
            world_size=world_size,
            **arguments.to_dict())
    else:
        trainer = LMTrainer.load(
            world_size=world_size,
            use_input_mask=arguments.dependency_mode == "input",
            **arguments.to_dict())

    arguments.update_from_kwargs(**trainer.config.to_dict())

    if data_provider is None:
        data_provider = functions._load_data_provider(
            arguments,
            memmaped=True)

    # TODO: Hooks do not save dataset name or number.
    # Idea: add counter to trainer that counts number of received
    # datasets and give this number to hook

    model_dir = os.path.join(trainer.model_dir, arguments.model_name)

    if arguments.att_plot:
        trainer.add_hook(AttentionPlotHook(
            os.path.join(model_dir, "hooks", "att_plots")
        ))

    if arguments.tree_plot:
        trainer.add_hook(TreePlotHook(
            os.path.join(model_dir, "hooks", "tree_plots"),
            masks_setting=arguments.masks_setting
        ))

    # Training setting
    metrics = trainer.test(**data_provider.datasets)

    generated = []
    for _ in range(20):
        generated.append(
            trainer.generate(data_provider.datasets["token_mapper"]))
    info(
        arguments.rank, logger,
        f"Generated model output sample: {generated}")

    del trainer
    del data_provider

    return metrics  # type: ignore
