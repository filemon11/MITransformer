import torch
import os

from . import parsing, ddp, mains

from mitransformer.utils.logmaker import (
    getLogger, info)

logger = getLogger(__name__)


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


def main(arguments: "parsing.ParserArgs") -> None:
    if arguments.mode == "dataprep":
        mains.main_dataprep(arguments)
    elif arguments.mode == "split":
        assert isinstance(arguments, parsing.SplitParserArgs)
        mains.main_split(arguments)
    else:
        try:
            n_devices = (
                int(os.environ["WORLD_SIZE"]) if arguments.use_ddp else 1)
        except ValueError:
            n_devices = torch.cuda.device_count() if arguments.use_ddp else 1
        assert not ((n_devices == 1 or not arguments.use_ddp) and
                    (arguments.rank is not None and arguments.rank > 0)), (
            "Rank cannot be larger than 0 if only having one device"
            "/not using ddp. "
            f"Received --local-rank {arguments.rank} "
            f"--use_ddp {arguments.use_ddp} "
            f"and number of recognised CUDA devices is {n_devices}.")
        info(arguments.rank, logger, f"Running on {n_devices} devices.")
        with ddp.ddp(arguments.rank, n_devices) as ddp_status:
            info(arguments.rank, logger, f"Using DDP: {ddp_status}")
            mode = arguments.mode
            match mode:
                case "train":
                    assert isinstance(arguments, parsing.TrainParserArgs)
                    info(arguments.rank, logger, "Launching model training.")
                    mains.main_train_multiple(arguments, n_devices)
                case "test":
                    assert isinstance(arguments, parsing.TestParserArgs)
                    info(arguments.rank, logger, "Launching model testing.")
                    mains.main_test(arguments, n_devices)
                case "hyperopt":
                    assert isinstance(arguments, parsing.HyperoptParserArgs)
                    info(
                        arguments.rank, logger,
                        "Launching hyperparameter tuning.")
                    mains.main_hyperopt(arguments, n_devices)
                case "compare":
                    assert isinstance(arguments, parsing.CompareParserArgs)
                    info(arguments.rank, logger, "Launching model comparison.")
                    mains.main_compare(arguments, n_devices)
                case "rt":
                    assert isinstance(arguments, parsing.RTParserArgs)
                    info(arguments.rank, logger, "Launching model comparison.")
                    mains.main_rt_multiple(arguments, n_devices)
                case _:
                    raise Exception(f"Unknown mode: '{mode}'")
