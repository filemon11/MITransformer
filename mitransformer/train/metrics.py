"""
The code defines classes and functions for handling and computing metrics
in a machine
learning context, including methods for calculating and aggregating
metrics, as well as writing
metrics to a TensorBoard SummaryWriter.
"""

import pandas as pd
import torch
import numpy as np
import numbers

from torch.utils.tensorboard.writer import SummaryWriter

import math

from dataclasses import dataclass, fields
from functools import total_ordering
from contextlib import contextmanager
from flatten_json import flatten  # type: ignore

from typing import (Self, Literal, cast,
                    Iterable, Mapping,
                    ClassVar, TypeVar, Callable,
                    Any)

from ..utils.params import Params
from ..utils.logmaker import getLogger, warning


logger = getLogger(__name__)


M = TypeVar("M", bound="Metric")
N = TypeVar("N")


def sum_metrics(metrics: Iterable[M]) -> M:
    '''This function calculates the sum of metrics in an iterable.

    Parameters
    ----------
    metrics : Iterable[M]
        The `metrics` parameter in the `sum_metrics`
    function is expected to be an iterable containing
    elements of type `M`. The function computes the sum of
    all elements in the iterable and returns the
    result.

    Returns
    -------
        The function `sum_metrics` returns the sum of
    all elements in the input `metrics` iterable.

    '''
    s: None | M = None
    for m in metrics:
        if s is None:
            s = m
        else:
            s = cast(M, m + s)
    assert s is not None, "Iterable of metrics cannot be empty."
    return s


def sum_and_std_metrics(
        metrics: "Iterable[Metric]"
        ) -> dict[str, tuple[float, float]]:
    '''This function calculates the mean and standard deviation for each
    metric in a given list of Metric
    objects.

    Parameters
    ----------
    metrics : "Iterable[Metric]"
        The `metrics` parameter is expected to be an
        iterable containing objects of type `Metric`.

    Returns
    -------
        The function `sum_and_std_metrics` returns
    a dictionary where each key corresponds to a metric name
    and the value is a tuple containing the mean and standard deviation of
    that metric calculated from
    the input list of Metric objects.
    '''
    ms = list(metrics)
    n = len(ms)
    out_dict: dict[str, tuple[float, float]] = dict()
    means: dict[str, float] = sum_metrics(ms).to_dict()
    for key, mean_value in means.items():
        if check_numeral(mean_value):
            xs = [getattr(m, key) for m in ms]
            out_dict[key] = (
                mean_value,
                math.sqrt(sum([(x-mean_value)**2 for x in xs]) / n))
    return out_dict


minimise = {"lm_loss": True,
            "loss": True,
            "arc_loss": True,
            "perplexity": True,
            "uas": False,
            }
# The `minimise` dictionary is used to specify whether each metric should be
# minimized or maximized
# during optimization.


@total_ordering
@dataclass
class Metric(Params):
    '''This Python class defines a Metric object with various properties
    and methods for calculating
    and manipulating metrics in a machine learning context.

    Attributes
    ----------
    num : int
        The number of tokens for which the metrics were
        accumulated.
    _lm_loss : torch.Tensor
        Language modelling loss.
    main_metric : str
        Main metric. Useful to directly perform backpropagation
        on.

    Returns
    -------
    The code provided defines a class `Metric` with various
    methods and properties for handling
    metrics in a machine learning context. The class includes
    functionality for calculating mean
    values, adding metrics together, printing results, detaching
    tensors, moving tensors to
    different devices, comparison operations, and
    converting metrics to dictionaries.
    '''
    num: float = 0
    _lm_loss: torch.Tensor = torch.tensor(0)

    _to_mean: ClassVar[set[str]] = {"lm_loss", "loss"}

    _convert: ClassVar[dict[str, Callable[[N], N]]] = {}    # type: ignore

    main_metric: str = "loss"

    # Whether optimisation means minimising (if False: maximising)
    minimise: ClassVar[dict[str, bool]] = minimise
    loss: ClassVar[torch.Tensor]
    # just for typing so that we can safely call metric.loss.backward()
    # without the type checker complaining

    @property
    def device(self) -> torch.device:
        '''This function returns the device of the loss attribute.

        Returns
        -------
        torch.device
            The `device` property is being returned, which is
            the device of the loss tensor.
        '''
        return self.loss.device

    @property
    def is_cuda(self) -> bool:
        '''This function checks if the loss object is using CUDA
        for computation and returns a boolean
        value.

        Returns
        -------
        bool
            The `is_cuda` property of the `loss` attribute of the
            object referenced by `self` is being
            returned. This property likely indicates whether the
            loss tensor is stored on a CUDA device
            (GPU) or not.
        '''
        return self.loss.is_cuda

    def minval(self) -> float:
        '''This function returns positive infinity
        if the 'minimise' flag for the main metric is True,
        otherwise it returns negative infinity.

        Returns
        -------
        float
            The `minval` method is returning positive infinity if the condition
            `self.minimise[self.main_metric]` is true, otherwise it
            returns negative infinity.
        '''
        return math.inf if self.minimise[self.main_metric] else -math.inf

    def maxval(self) -> float:
        '''This function returns negative infinity
        if the 'minimise' flag for the main metric is True,
        otherwise it returns positive infinity.

        Returns
        -------
        float
            The `minval` method is returning negative infinity if the condition
            `self.minimise[self.main_metric]` is true, otherwise it
            returns positive infinity.
        '''
        return -self.minval()

    def __getattr__(self, prop: str):
        '''The function `__getattr__` calculates the mean for
        metrics based on the provided property.

        Parameters
        ----------
        prop : str
            The `prop` parameter in the `__getattr__` method
            represents the name of the attribute that is
            being accessed or requested. This method is called when an
            attribute is not found through the
            usual process of looking up the attribute in the instance's
            dictionary. It allows you to
            dynamically compute or provide attributes

        Returns
        -------
        None
        '''
        if prop in self._to_mean:
            val = self.__getattribute__(f"_{prop}") / self.num
            if prop in self._convert:
                val = self._convert[prop](val)
            return val
        else:
            raise AttributeError(
                f"'{self.__class__}' has no attribute '{prop}' or '_{prop}'.")

    def _add_fields(self,
                    name: str,
                    m1: "Metric",
                    m2: "Metric") -> float | torch.Tensor:
        '''This function sums two fields from
        two Metric objects, handling special cases for the
        "main_metric" field and missing values.

        Parameters
        ----------
        name : str
            The `name` parameter in the `_add_fields` method
            is a string that represents the name of the
            field you want to add from the `m1` and `m2` objects.
        m1 : "Metric"
            "m1" and "m2" are instances of the class "Metric".
            The function `_add_fields` takes in the name
            of a field, along with two instances of the "Metric"
            class (m1 and m2), and returns the sum of
            the values of the specified field from both
        m2 : "Metric"
            It looks like the code snippet you provided defines
            a method `_add_fields` that takes in three

        Returns
        -------
            The function returns a float or a torch.Tensor,
            depending on the values of the
            input parameters `m1` and `m2`.
        '''
        val1 = getattr(m1, name)
        val2 = getattr(m2, name)
        if name == "main_metric":
            # val1 and val2 are the names of the main metric
            assert val1 == val2, "Main metrics do not correspond!"
            return val1
        elif val1 is None:
            assert m1.num == 0
            return val2
        elif val2 is None:
            assert m2.num == 0
            return val1
        else:
            return val1 + val2

    def __add__(self, other: "Metric") -> "Metric":
        '''This Python function defines an addition
        operation for a Metric class, ensuring that metrics of
        the same type can be added together.

        Parameters
        ----------
        other : "Metric"
            The `other` parameter in the `__add__` method
            represents another instance of the `Metric` class
            that you want to add to the current instance.
            The method is designed to add two instances of the
            `Metric` class together, ensuring that the operation
            is valid based on the types of the metric fields.

        Returns
        -------
        "Metric"
            The `__add__` method is returning the result of adding two Metric
            objects together. The method
            first checks if the `other` object is of a lower type
            than `self`, and if so, it swaps the
            objects to ensure that `self` is of the higher type.
            Then, it iterates over the fields of the
            higher type object and adds the corresponding fields of `self`.
        '''

        higher = None
        if (isinstance(self, other.__class__)
                and not isinstance(other, self.__class__)):
            return other.__add__(self)
        elif isinstance(other, self.__class__):
            higher = self
        assert higher is not None, (f"Cannot add metrics "
                                    f"of types {self.__class__} "
                                    f"and {other.__class__}")

        return higher.__class__(
            **{
                f.name: self._add_fields(f.name, self, other)  # type: ignore
                for f in fields(higher)})

    def __radd__(self, other: "Metric") -> "Metric":
        '''Supports the addition operation when the left
        operand does not support addition with the right operand.

        Parameters
        ----------
        other : "Metric"

        Returns
        -------
        "Metric"

        '''
        return self + other

    def __truediv__(self, other: float) -> "Metric":
        '''This function allows to apply division
        to the metric.
        This is done by multiplying `self.num` with
        the `other` parameter. We divide the metrics by `self.num`
        before retrieving them.

        Parameters
        ----------
        other : float
            The `other` parameter in the `__truediv__` method represents
            the value by which the current
            object will be divided.

        Returns
        -------
        "Metric"
            A new instance of the class "Metric" is being returned.
        '''
        new = self.__class__(main_metric=self.main_metric) + self
        new.num *= other
        return new

    def print(
            self, epoch: int,
            total_epochs: int, kind: str) -> None:
        '''The function `print` formats and prints information
        about the current epoch, total epochs, and a
        specified kind of data.

        Parameters
        ----------
        epoch : int
            The `epoch` parameter represents the current
            epoch number during training or iteration in a
            machine learning model. It is typically an
            integer value indicating the current iteration step.
        total_epochs : int
            Total_epochs is the total number of epochs in the training process.
            It represents the number of
            times the algorithm has gone through the entire training dataset.
        kind : str
            Kind refers to the type of operation or
            process being performed during the current epoch.
            It could be training, validation, testing, or any
            other relevant operation that is being executed.
        '''
        strs = [f"{name}: {getattr(self, name):.2f}" for name in self._to_mean]
        print(f"[{epoch}/{total_epochs}] {kind}:: " + ", ".join(strs))

    def print_test(self) -> None:
        '''The function `print_test` prints formatted
        test results based on attributes of the object it is
        called on.
        '''
        strs = [f"{name}: {getattr(self, name):.2f}" for name in self._to_mean]
        print("Test results: " + ", ".join(strs))

    @property
    def _loss(self) -> torch.Tensor:
        '''This function returns the private
        attribute `_lm_loss` as a torch.Tensor.

        Returns
        -------
        torch.Tensor
            The `_loss` property is returning the `_lm_loss` attribute,
            which is expected to be a torch.Tensor.
        '''
        return self._lm_loss

    def detach(self) -> None:
        '''The `detach` function iterates through
        the fields of an object and detaches any `torch.Tensor`
        values found.
        '''
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                setattr(self, f.name, value.detach())

    def to_(self, device: str | torch.device) -> None:
        '''The function `to_` takes a device as input
        and converts all torch.Tensor attributes of an object
        to that device in place.

        Parameters
        ----------
        device : str | torch.device
            The `device` parameter in the `to_` method can be either
            a string or an instance of
            `torch.device`. This parameter is used to specify the
            device to which the `torch.Tensor` objects
            within the object should be moved to.

        '''
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                setattr(self, f.name, value.to(device))

    def to(self, device: str | torch.device) -> Self:
        '''The `to` function in Python takes a device argument and
        returns a new instance of the class with
        its attributes converted to the specified device
        if they are `torch.tensor`s.

        Parameters
        ----------
        device : str | torch.device
            The `device` parameter in the `to` method can be
            either a string or an instance of
            `torch.device`. This parameter specifies the device
            to which the tensors in the object should be
            moved.

        Returns
        -------
        Self
            The `to` method is returning a new instance of the class
            with all attributes converted to the
            specified device (if they are torch Tensors)
            or left unchanged if they are not torch Tensors.
        '''
        return self.__class__(
            **{f.name: (getattr(self, f.name).to(device)
                        if isinstance(getattr(self, f.name), torch.Tensor)
                        else getattr(self, f.name)) for f in fields(self)})

    @property
    def main_value(self) -> torch.Tensor:
        '''Returns the attribute specified
        by `self.main_metric`.

        Returns
        -------
        torch.Tensor
        '''
        return getattr(self, self.main_metric)

    def __gt__(self, other: object) -> bool:
        '''This function compares two objects based on their main values
        with consideration for a
        minimizing or maximizing factor.

        Parameters
        ----------
        other : object
            The `other` parameter in the `__gt__` method
            represents another object that you are comparing
            with the current object.

        Returns
        -------
        bool
            If the other object is an instance of the `Metric` class,
            it compares the main values
            of both objects after applying a factor based
            on whether the `minimise` flag is set. If the
            other object is not a `Metric` instance,
            it attempts to convert it to a float
        '''
        factor = -1 if self.minimise[self.main_metric] else 1

        self_attr = (
            self.main_value.item()
            if isinstance(self.main_value, torch.Tensor)
            else self.main_value)

        if isinstance(other, Metric):
            other_attr = (
                other.main_value.item()
                if isinstance(other.main_value, torch.Tensor)
                else other.main_value)

            return factor*self_attr > factor*other_attr
        else:
            try:
                return (factor*self_attr
                        > factor*float(other))  # type: ignore
            except ValueError:
                return False

    def __eq__(self, other: object) -> bool:
        '''This function compares the main values of two objects based
        on a specified metric and returns
        True if they are equal.

        Parameters
        ----------
        other : object
            The `other` parameter in the `__eq__` method represents the
            object that the current object is
            being compared to for equality.

        Returns
        -------
        bool
            The `__eq__` method is returning a boolean value based
            on the comparison of the main values of
            two objects. If the other object is an instance of the
            Metric class, it compares the main values
            of both objects after applying a factor based on whether the main
            metric is set to minimize or
            not. If the other object is not an instance of the Metric class,
            it attempts to convert the
            other
        '''
        self_attr = (
            self.main_value.item()
            if isinstance(self.main_value, torch.Tensor)
            else self.main_value)
        if isinstance(other, Metric):
            return other == self_attr
        else:
            try:
                return self_attr == float(other)  # type: ignore
            except ValueError:
                return False

    def to_dict(self, as_str: bool = False,
                omit_undefined: bool = False) -> dict[str, Any]:
        if as_str:
            return {attr: str(getattr(self, attr))
                    for attr in self._to_mean}
        else:
            return {attr: self._to_float(getattr(self, attr))
                    for attr in self._to_mean}

    def _to_float(self, item: Any) -> Any:
        try:
            return float(item)
        except TypeError:
            return item


@dataclass
class SupervisedMetric(Metric):
    '''This Python class extends the Metric class
    by adding arc loss for the supervised setting
    and a weight (alpha) for balancing language modelling
    loss and dependency loss.

    Attributes
    ----------
    alpha : float
        Weight for balancing language modelling and dependency
        loss. `loss = lm_loss*alpha + arc_loss*(1-alpha)` if
        alpha is not None.
    _arc_loss : torch.Tensor
        Dependency parsing loss.
    '''

    _arc_loss: torch.Tensor = torch.tensor(0)
    alpha: float | None = None
    _to_mean: ClassVar[set[str]] = Metric._to_mean | {"arc_loss"}

    @property
    def _loss(self) -> torch.Tensor:
        '''This function calculates a loss value based on
        two sub-losses, with the weighting between them
        determined by the alpha parameter.

        Returns
        -------
        torch.Tensor
            The method `_loss` returns a torch.Tensor that is
            calculated based on the value of the `alpha`
            attribute. If `alpha` is None, it returns the sum
            of `_lm_loss` and `_arc_loss`. If `alpha` is
            not None, it returns a weighted sum of `_lm_loss`
            and `_arc_loss` based on the value of `alpha`.
        '''
        if self.alpha is None:

            warning(None, logger, "SupervisedMetric.alpha is None!")
            return (self._lm_loss
                    + self._arc_loss)
        else:
            return (self.alpha*self._lm_loss
                    + (1-self.alpha)*self._arc_loss)

    def _add_fields(self,
                    name: str,
                    m1: "Metric",
                    m2: "Metric") -> float | torch.Tensor:
        '''The function `_add_fields` takes in
        two Metric objects and a field name. If the field to sum
        is "alpha", then it checks whether alpha is the same for both
        objects or whether one of the objects has None as alpha
        and returns alpha if true, else it raises
        an assertion error if they are different. For all other names,
        it calls `super()._add_fields`.

        Parameters
        ----------
        name : str
            The `name` parameter in the `_add_fields` method
            is a string that specifies the field name for
            which the operation is being performed.
        m1 : "Metric"
            Metric m1 is an instance of the class "Metric".
        m2 : "Metric"
            m2 is an instance of the class "Metric".

        Returns
        -------
        float | torch.Tensor
            Combined field value.
        '''
        if name == "alpha":
            val1 = getattr(m1, name)
            val2 = getattr(m2, name)
            assert (val1 == val2 or val2 is None
                    or val1 is None), (
                        "Cannot combine metrics with different alpha."
                    )
            return val2 if val2 is not None else val1

        return super()._add_fields(name, m1, m2)


@dataclass
class EvalMetric(Metric):
    '''This Python class extends the Metric class
    by adding additional metric fields that should be computed for
    model evaluation.

    Attributes
    ----------
    _perplexity : float
        Perplexity.
    '''
    _perplexity: float = 0
    _to_mean: ClassVar[set[str]] = Metric._to_mean | {"perplexity"}
    _convert = Metric._convert | {"perplexity": math.exp}    # type: ignore


@dataclass
class SupervisedEvalMetric(SupervisedMetric, EvalMetric):
    '''This Python class extends the SupervisedMetric class
    and EvalMetric classes by adding additional metric fields
    that should be computed for model evaluation.

    Attributes
    ----------
    _uas : float
        Unlabelled attachment score.
    _att_entropy: pd.DataFrame
        Attention entropy of the model heads.
    '''
    _uas: float = 0
    _att_entropy: pd.DataFrame | None = None
    _to_mean: ClassVar[set[str]] = (SupervisedMetric._to_mean
                                    | EvalMetric._to_mean
                                    | {"uas", "att_entropy"})


class MetricWriter(SummaryWriter):
    '''Class to manage metric and hyperparameter tracking
    with TensorBoard.
    '''
    def add_metric(
            self,
            metric: Metric,
            epoch: int,
            split: Literal["train", "eval", "test"]
            ) -> None:
        '''The `add_metric` function adds metrics to TensorBoard.

        Parameters
        ----------
        metric : Metric
            The `metric` parameter refers to an object of the `Metric` class.
            It is used to store and represent metrics such as
            accuracy, loss, etc., during the training or
            evaluation of a model.
        epoch : int
            The `epoch` parameter in the `add_metric` method represents
            the current epoch number during
            training or evaluation of a model.
        split : Literal["train", "eval", "test"]
            The `split` parameter is used to specify the dataset split
            for which the metric is being added
            (training set, evaluation set, or test set
        '''
        for key, value in metric.to_dict().items():
            if isinstance(value, pd.DataFrame):
                flattened = flatten(value.to_dict())
                for k2, v2 in flattened.items():
                    self.add_scalar(f"{k2}/{split}", v2, epoch)
            else:
                self.add_scalar(f"{key}/{split}", value, epoch)

    def add_params(
            self,
            params: Mapping[str, Any],
            metric: Metric,
            run_name: str | None = None,
            global_step: int | None = None,
            ) -> None:
        '''The `add_params` function takes in parameters,
        a metric, and optional run information to add
        hyperparameters and metrics to a logging system.

        Parameters
        ----------
        params : Mapping[str, Any]
            The `params` parameter is a mapping (dictionary-like object)
            that contains key-value pairs
            where the keys are strings and the values can be of any type.
            Only the key-value pairs where the values are of type float, int,
            torch.Tensor, bool or str are recorded.
        metric : Metric
            The `metric` object to record.
        run_name : str | None
            The `run_name` parameter in the `add_params` method
            is a string that represents the name of the
            run or experiment.
        global_step : int | None
            The `global_step` parameter in the `add_params` method
            is an optional integer value that
            represents the global step or iteration number at which the
            parameters and metrics are being added. It is used to track
            the progress of the training process.
        '''
        self.add_hparams(
            {key: value for key, value
                in params.items() if check_type(value)},
            {f"_{key}": value for key, value in metric.to_dict().items()
                if check_numeral(value)},
            run_name=run_name,
            global_step=global_step)


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


def check_type(value: Any) -> bool:
    if (isinstance(value, float)
            or isinstance(value, int)
            or isinstance(value, torch.Tensor)
            or isinstance(value, bool)
            or isinstance(value, str)):
        return True
    return False


def check_numeral(value: Any) -> bool:
    if (isinstance(value, numbers.Number)
            or isinstance(value, torch.Tensor)
            or isinstance(value, np.ndarray)):
        return True
    return False
