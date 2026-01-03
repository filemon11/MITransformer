from __future__ import annotations


from typing import Any, Type, TypeVar, Self
from collections.abc import Sequence
from abc import ABC

import torch
import math
import numpy as np


from . import field, utils
from ...utils import params

# ----------------------------- Metric --------------------------------------

M = TypeVar("M", bound="Metric")


class Metric(params.Params, ABC):
    """Base class for metrics using a declarative fields table.

    Subclasses should declare a class-level `fields: dict[str, MetricField]`.
    Keys should match the runtime attribute names we store on instances. By
    convention, "internal" numeric fields that accumulate sums use a leading
    underscore (e.g. "_lm_loss") and counters like "num", "arc_num" are
    plain names.
    """

    # Example default fields for the base Metric. Subclasses extend via
    # `fields = {**Metric.fields, **{"_new": MetricField(...)}}`.
    fields: dict[str, field.MetricField] = {
        "num": field.num,
        "main_metric": field.main_metric("loss"),
    }

    def __init__(self, **kwargs: Any) -> None:
        # initialize all fields from fields, allowing overrides via kwargs
        # print(kwargs)
        # print(self.fields.items())
        num_losses = len(
            [mf for mf in self.fields.values() if mf.include_in_loss])
        for name, mf in self.fields.items():
            value = kwargs.get(name, mf.default)
            if isinstance(mf, field.WeightField):
                if value is None:
                    value = [1.0]*num_losses
                assert isinstance(value, Sequence), (
                    "The weights must be provided in a sized object")
                assert len(value) == num_losses, (
                    f"Length of {name} weight field is not equal"
                    f" to the number of losses of metric {self.__class__}")

            # copy tensors to avoid accidental sharing
            if isinstance(value, torch.Tensor):
                value = value.clone()
            self._set_raw(name, value)

    # ---------------------- attribute access / presentation -----------------

    def _get_raw(self, name: str) -> Any:
        """Return the stored raw field value (no conversion/division).

        Attempts exact name first, then with a leading underscore if present.
        """
        if hasattr(self, f"_{name}"):
            return getattr(self, f"_{name}")
        raise AttributeError(name)

    def _set_raw(self, name: str, val: Any) -> Any:
        """TODO
        """
        setattr(self, f"_{name}", val)

    def __getattr__(self, attr: str) -> Any:  # called when normal lookup fails
        """Expose presented (possibly normalized and converted) attributes.

        Behavior:
            TODO
        """
        # special-case 'loss' property
        if attr == "loss":
            return self._compute_loss()

        # try to resolve to a declared field (support both
        # 'lm_loss' and '_lm_loss')
        if attr in self.fields:
            mf = self.fields[attr]
            raw = self._get_raw(attr)
            # normalization
            if mf.reduce_by is not None:
                counter = getattr(self, mf.reduce_by)
                try:
                    # guard division by zero
                    raw_presented = raw / counter if counter != 0 else raw
                except Exception:
                    raw_presented = raw
            else:
                raw_presented = raw
            # conversion
            if mf.converter is not None:
                try:
                    return mf.converter(raw_presented)
                except Exception:
                    # don't crash metrics printing - return raw_presented
                    return raw_presented
            return raw_presented

        # attribute not a field -> normal AttributeError
        raise AttributeError(attr)

    # ---------------------- arithmetic / aggregation ----------------------

    def __add__(self: Self, other: M) -> M:
        """Merge two metrics. The more-derived type wins when mixing types.

        Rule: if one operand is instance of the other's class but not vice
        versa,
        delegate to the other operand so fields of the more-derived class are
        kept.
        """
        outtype: Type[Self] = self.__class__

        # delegate to more-derived left operand
        if isinstance(self, other.__class__) and not isinstance(
                other, self.__class__):
            return other.__add__(self)
        if not isinstance(other, self.__class__):
            raise TypeError("Cannot add metrics: incompatible classes")

        # now both are same class (or other is subclass)
        out: Self = outtype()
        for name, mf in self.fields.items():
            a = self._get_raw(name)
            b = other._get_raw(name)
            # statics must match (unless one side is None)
            if mf.static:
                if a is None and b is None:
                    value = None
                elif a is None:
                    value = b
                elif b is None:
                    value = a
                else:
                    if isinstance(a, torch.Tensor) and isinstance(
                            b, torch.Tensor):
                        if torch.all(a == b):
                            value = a
                        else:
                            raise AssertionError(
                                f"Static field {name} differs: {a} != {b}")
                    else:
                        if a == b:
                            value = a
                        else:
                            raise AssertionError(
                                f"Static field {name} differs: {a} != {b}")
            else:
                # handle combination logic for tensors and numbers
                if a is None:
                    value = b
                elif b is None:
                    value = a
                else:
                    if isinstance(a, torch.Tensor) and isinstance(
                            b, torch.Tensor):
                        value = a + b
                    elif isinstance(a, (int, float)) and isinstance(
                            b, (int, float)):
                        value = a + b
                    else:
                        # for dataframes and other objects,
                        # prefer b if a is "empty"
                        try:
                            value = a + b  # type: ignore[operator]
                        except Exception:
                            value = b
            out._set_raw(name, value)
        return out  # type: ignore[return-value]

    def __radd__(self: M, other: Any) -> M:  # support sum() use
        if other == 0:
            return self
        return self + other  # type: ignore[arg-type]

    def __truediv__(self: M, scalar: float) -> M:
        Factory: Type[M] = type(self)
        out = Factory()
        for name, mf in self.fields.items():
            val = self._get_raw(name)
            if mf.counter:
                out._set_raw(name, val*scalar)
            else:
                out._set_raw(name, val)
            val = self._get_raw(name)
        return out

    # ---------------------- device / detach helpers -----------------------

    @property
    def device(self) -> torch.device:
        # find a tensor field to ask device from
        for name in self.fields.keys():
            v = self._get_raw(name)
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")

    @property
    def is_cuda(self) -> bool:
        if self.device == "cpu" or self.device == torch.device("cpu"):
            return False
        return True

    def detach_(self) -> None:
        for name in self.fields.keys():
            v = self._get_raw(name)
            if isinstance(v, torch.Tensor):
                self._set_raw(name, v.detach())

    def to(self, device: torch.device | str) -> Self:
        Factory: Type[Self] = type(self)
        out = Factory()
        for name in self.fields.keys():
            v = self._get_raw(name)
            if isinstance(v, torch.Tensor):
                out._set_raw(name, v.to(device))
            else:
                out._set_raw(name, v)
        return out

    def to_(self, device: torch.device | str) -> None:
        for name in self.fields.keys():
            v = self._get_raw(name)
            if isinstance(v, torch.Tensor):
                self._set_raw(name, v.to(device))

    # ---------------------- presentation / serialization -----------------

    @property
    def loss_fields(self) -> tuple[str, ...]:
        loss_field_list: list[str] = []
        for name, mf in self.fields.items():
            if mf.include_in_loss:
                loss_field_list.append(name)
        return tuple(loss_field_list)

    def _compute_loss(self) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for name in self.loss_fields:
            val = getattr(self, name)
            if isinstance(val, torch.Tensor):
                parts.append(val)
            else:
                # try convertable
                try:
                    parts.append(torch.tensor(float(val)))
                except Exception:
                    pass
        if not parts:
            return torch.tensor(0.)
        return sum(parts)   # type: ignore

    @property
    def main_metric(self) -> str:
        mm = self._get_raw("main_metric")
        if isinstance(mm, str) and ":" in mm:
            mm, *_ = mm.split(":")
        return mm

    @property
    def main_value(self) -> Any:
        return getattr(self, self.main_metric)

    @property
    def _main_direction(self) -> bool:
        name = self.main_metric
        if name in self.fields:
            assert self.fields[name].minimise is not None
            return self.fields[name].minimise  # type: ignore
        elif name == "loss":
            return True
        else:
            raise Exception(
                f"Main metric '{name}' is not 'loss' and "
                "cannot be found in the Metric fields")

    def __gt__(self, other: object) -> bool:
        factor = -1 if self._main_direction else 1
        self_val = self.main_value
        if isinstance(self_val, torch.Tensor):
            self_val = float(self_val.item())
        if isinstance(other, Metric):
            other_val = other.main_value
            if isinstance(other_val, torch.Tensor):
                other_val = float(other_val.item())
            return factor * float(self_val) > factor * float(other_val)
        else:
            try:
                return factor * float(
                    self_val) > factor * float(other)  # type: ignore[arg-type]
            except Exception:
                return False

    def __eq__(self, other: object) -> bool:
        self_val = self.main_value
        if isinstance(self_val, torch.Tensor):
            self_val = float(self_val.item())
        if isinstance(other, Metric):
            other_val = other.main_value
            if isinstance(other_val, torch.Tensor):
                other_val = float(other_val.item())
            return self_val == other_val
        else:
            try:
                return float(
                    self_val) == float(other)  # type: ignore[arg-type]
            except Exception:
                return False

    def to_dict(
            self, as_str: bool = False,
            omit_undefined: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, mf in self.fields.items():
            # skip internal main metric declaration
            if name == "main_metric":
                continue
            raw = self._get_raw(name)
            presented = raw
            if mf.reduce_by is not None:
                counter = self._get_raw(mf.reduce_by)
                try:
                    presented = raw / counter if counter != 0 else raw
                except Exception:
                    presented = raw
            if mf.converter is not None:
                try:
                    presented = mf.converter(presented)
                except Exception:
                    pass
            key = name.lstrip("_")
            out[key] = str(
                presented) if as_str else self._to_primitive(presented)
        out["loss"] = str(self._compute_loss())
        return out

    def _to_primitive(self, v: Any) -> Any:
        # try to convert tensors and numpy
        if isinstance(v, torch.Tensor):
            try:
                return float(v.detach().cpu().item())
            except Exception:
                return v
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

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
        return math.inf if self._main_direction else -math.inf

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


class WeightedMetric(Metric):
    fields = {
        **Metric.fields,
        "weights": field.WeightField()
    }

    def _compute_loss(self) -> torch.Tensor:
        # custom composition

        weights: Sequence[float] = getattr(self, "weights")
        components: list[torch.Tensor] = [
            weights[i]*utils.to_t(getattr(self, name))
            for i, name in enumerate(self.loss_fields)]
        summed: torch.Tensor = sum(
            components, torch.zeros_like(components[0]))

        return summed
