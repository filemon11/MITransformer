import torch
import math

from dataclasses import dataclass

from typing import Optional, Any, Callable

# ------------------------- MetricField -------------------------------------


@dataclass(frozen=True)
class MetricField:
    """Descriptor for a metric attribute.

    Attributes
    ----------
    default: Any
        Default value stored for this field.
    reduce_by: Optional[str]
        Name of a counter field (e.g. "num", "arc_num") used to normalize this
        field when presenting as a per-example/per-token value. If None,
        the field is treated as a scalar that should not be divided.
    converter: Optional[Callable[[Any], Any]]
        Optional function to convert the (possibly normalized) field for
        presentation (e.g. perplexity = exp(loss)).
    static: bool
        If True, this field must be equal on metrics that are added together
        (unless one side is None). Used for things like hyperparams.
    include_in_loss: bool
        If True, this field participates in the computed `loss` property.
    """
    default: Any
    reduce_by: Optional[str] = None
    converter: Optional[Callable[[Any], Any]] = None
    static: bool = False
    include_in_loss: bool = False

    # True → minimize; False → maximize; None → not a main metric
    minimise: Optional[bool] = None


# ------------------------- Important Fields ----------------------------------

def main_metric(name: str) -> MetricField:
    return MetricField(name, static=True)


def loss(reduce_by: str = "num") -> MetricField:
    return MetricField(
        torch.tensor(0,), reduce_by=reduce_by,
        include_in_loss=True, minimise=True)


num = MetricField(0)

perplexity = MetricField(
    0.0, reduce_by="num", converter=math.exp, minimise=True)
uas = MetricField(0.0, minimise=False)
att_entropy = MetricField(None)

attention_entropy_loss = loss("num")
distance_loss = loss("num")

weight = MetricField(1.0, static=True)
