from logging import (
    getLogger, Logger, basicConfig,
    INFO, WARNING, DEBUG, ERROR)
from pathlib import Path
import os
from time import localtime, strftime

time_pattern = "%Y-%m-%d %H:%M:%S"


def get_timestr() -> str:
    return strftime(time_pattern, localtime())


def log(rank: int | None, logger: Logger,
        level: str | int, msg: str) -> None:
    if rank is None or rank == 0:
        if isinstance(level, int):
            logger.log(level, msg)
        else:
            getattr(logger, level)(msg)


def info(rank: int | None, logger: Logger,
         msg: str) -> None:
    log(rank, logger, INFO, msg)


def warning(rank: int | None, logger: Logger,
            msg: str) -> None:
    log(rank, logger, WARNING, msg)


def logging_config(logname: str | None = None,
                   logpath: str = "./logs") -> None:
    if logname is None:
        logname = os.path.join(logpath, get_timestr() + ".log")
    else:
        logname = os.path.join(logpath, logname + ".log")

    if os.path.isfile(logname):
        logname = f"{logname[:-4]}_{get_timestr()}.log"

    Path(logpath).mkdir(parents=True, exist_ok=True)
    basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt=time_pattern,
        filemode='w',
        filename=logname,
        level=INFO)
