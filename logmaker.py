from logging import (getLogger, Logger, basicConfig,
                     INFO, WARNING, DEBUG, ERROR)
import logging


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