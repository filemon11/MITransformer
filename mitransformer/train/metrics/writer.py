from torch.utils.tensorboard.writer import SummaryWriter
import pandas as pd
from flatten_json import flatten  # type: ignore
from contextlib import contextmanager

from . import base
from . import utils

from typing import Any, Optional

# ---------------------- writer + helpers ----------------------------------


class MetricWriter(SummaryWriter):
    def add_metric(self, metric: base.Metric, epoch: int, split: str) -> None:
        for key, value in metric.to_dict().items():
            if isinstance(value, pd.DataFrame):
                flattened = flatten(value.to_dict())
                for k2, v2 in flattened.items():
                    try:
                        self.custom_add_scalar(k2, float(v2), epoch, split)
                    except Exception:
                        pass
            else:
                try:
                    self.custom_add_scalar(key, float(value), epoch, split)
                except Exception:
                    # non-scalar values are ignored
                    pass

    def custom_add_scalar(
            self, key: str, value: float, epoch: int, split: str) -> None:
        self.add_scalar(f"{key}/{split}", value, epoch)

    def add_params(
            self, params: dict[str, Any], metric: base.Metric | dict[str, Any],
            run_name: Optional[str] = None,
            global_step: Optional[int] = None) -> None:
        metric_dict = (
            metric.to_dict() if isinstance(metric, base.Metric) else metric)
        self.add_hparams(
            {k: v for k, v in params.items() if utils.check_type(v)},
            {f"_{k}": v for k, v in metric_dict.items()
                if utils.check_numeral(v)},
            run_name=run_name,
            global_step=global_step,
        )


@contextmanager
def metric_writer(*args, **kwargs):
    '''Context manager for `MetricWriter`.
    Closes the writer automatically when leaving
    its scope.

    Parameters
    ----------
    args
        Arguments to initialise `MetricWriter` with.
    kwargs
        Keyword arguments to initialise `MetricWriter` with.
    '''
    # Code to acquire resource, e.g.:
    writer = MetricWriter(*args, **kwargs)
    try:
        yield writer
    finally:
        # Code to release resource, e.g.:
        writer.flush()
