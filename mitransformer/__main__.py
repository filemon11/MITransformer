import optuna
import sys

from . import io
from .utils import logmaker

logger = logmaker.getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


# TODO: correct parser args for data loading
# TODO: make bool args optionally just acccept flag for True


def parse_args() -> (
        io.TrainParserArgs | io.HyperoptParserArgs
        | io.DataprepParserArgs | io.TestParserArgs
        | io.CompareParserArgs | io.RTParserArgs):
    parser = io.create_parser()
    args = parser.parse_args()

    try:
        mode = args.mode
    except AttributeError:
        raise Exception("mode attribute not found in arguments.")

    match mode:
        case "train":
            return io.TrainParserArgs(**vars(args))
        case "hyperopt":
            return io.HyperoptParserArgs(**vars(args))
        case "dataprep":
            return io.DataprepParserArgs(**vars(args))
        case "test":
            return io.TestParserArgs(**vars(args))
        case "rt":
            return io.RTParserArgs(**vars(args))
        case _:
            return io.CompareParserArgs(**vars(args))


if __name__ == "__main__":
    args: (
        io.TrainParserArgs | io.HyperoptParserArgs
        | io.DataprepParserArgs | io.TestParserArgs
        | io.CompareParserArgs | io.RTParserArgs)
    args = parse_args()
    logmaker.logging_config(logname=args.name)
    # logging_config(
    #     logname="log",
    #     logpath=os.path.join(LMTrainer.model_dir, name))
    #  TODO: save trainer and model log in model dir
    #  but general and hyperopt log in normal logdir

    # tokenise dataset
    # TODO: do this automatically
    # problem: cannot do in parallel and
    # therefore leads to timeout when doing
    # on only one rank.

    io.args_logic(args)

    # For hyperopt we deal with dropout later

    # Some checks
    assert not (args.use_ddp and args.device == "cpu"), (
        "Must set --device to a GPU when setting --use_ddp. "
        f"Received --device {args.device}, --use_ddp {args.use_ddp}")

    logmaker.info(args.rank, logger, f"Arguments provided: {str(sys.argv)}")
    io.main(args)
    # if n_devices > 1:
    #     mp.spawn(main, args=(n_devices,), nprocs=n_devices)  # type: ignore
    # else:
    #     main(None, n_devices)
