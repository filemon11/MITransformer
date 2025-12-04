from ...data import (DataProvider, DataConfig)
from ...train import LMTrainer
from .. import parsing

import optuna
import os

from mitransformer.utils.logmaker import (
    getLogger)

from typing import Union

logger = getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


"""
TODO:
- establish dataset naming and loading by name
- save loading config with dataset
- if no name: save as Wikitext_1, _2 etc.
- when loading look for all datasets in dataset folder
and check if there is a config that fits;
then load this dataset
and if no, search for dataset having this name
if no, search for preset datasets and load new
if not, search on huggingface and parse and load new.
"""


def _load_data_provider(
        arguments: Union[
            "parsing.ParserArgs ",
            "parsing.TestParserArgs | parsing.CompareParserArgs"],
        memmaped: bool = False,
        model_num: int | None = None
        ) -> DataProvider:
    try:
        if isinstance(arguments, parsing.TestParserArgs):
            provider = DataProvider.load(
                os.path.join(
                    LMTrainer.model_dir, arguments.model_name,
                    "data_config.json"),
                **arguments.to_dict())
        elif isinstance(arguments, parsing.CompareParserArgs):
            assert isinstance(model_num, int)
            provider = DataProvider.load(
                os.path.join(
                    LMTrainer.model_dir,
                    arguments.model1_name if model_num == 1
                    else arguments.model2_name,
                    "data_config.json"),
                **arguments.to_dict())
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        config = DataConfig.from_kwargs(
            include_test=False,
            memmapped=memmaped,
            **arguments.to_dict())
        provider = DataProvider(config, arguments.rank)
    return provider
