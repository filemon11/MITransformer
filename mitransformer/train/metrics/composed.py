from . import base, field

import torch

from typing import Sequence, Type

# ---------------------- concrete metric classes ---------------------------

_dynamic_weighted_metric_registry: dict[tuple[str, ...], str] = {}
_dynamic_weighted_eval_metric_registry: dict[tuple[str, ...], str] = {}


class LMMetric(base.Metric):
    fields = {
        **base.Metric.fields,
        "lm_loss": field.loss("num")
    }


class SupervisedMetric(LMMetric):
    fields = {
        **LMMetric.fields,
        "arc_num": field.num,
        "arc_loss": field.loss("arc_num"),
        "alpha": field.MetricField(None, static=True),
    }

    def _compute_loss(self) -> torch.Tensor:
        lm_loss = getattr(self, "lm_loss")
        arc_loss = getattr(self, "arc_loss")
        alpha = getattr(self, "alpha")

        return (alpha*lm_loss + (1-alpha)*arc_loss)


class EvalMetric(LMMetric):
    fields = {
        **LMMetric.fields,
        "perplexity": field.perplexity,
    }


class SupervisedEvalMetric(SupervisedMetric, EvalMetric):
    # merge field maps carefully: rightmost wins
    fields: dict[str, field.MetricField] = {
        **SupervisedMetric.fields, **EvalMetric.fields,
        "uas": field.uas,
        "att_entropy": field.att_entropy,
    }


class CostsMetric(LMMetric, base.WeightedMetric):
    fields = {
        **LMMetric.fields,
        **base.WeightedMetric.fields,
        "attention_entropy_loss": field.attention_entropy_loss,
        "attention_distance_loss": field.attention_distance_loss,
        "attention_difference_loss": field.attention_difference_loss,
        "attention_activation_loss": field.attention_activation_loss,
        "cosine_loss": field.cosine_loss,
        "surprox_loss": field.surprox_loss,
    }


class CostsEvalMetric(CostsMetric, EvalMetric):
    fields: dict[str, field.MetricField] = {
        **CostsMetric.fields, **EvalMetric.fields,
    }


def DynamicWeightedMetric(
        loss_names: Sequence[str]
        ) -> Type[base.WeightedMetric]:
    key = tuple(loss_names)
    if key in _dynamic_weighted_metric_registry:
        return globals()[_dynamic_weighted_metric_registry[key]]

    class_name = f"DynamicWeightedMetric_{'_'.join(key)}"
    fields_dict = {
        **base.WeightedMetric.fields,
        **{ln: field.loss("num") for ln in loss_names}}
    cls = type(class_name, (base.WeightedMetric,), {"fields": fields_dict})
    globals()[class_name] = cls  # ensure top-level reference for pickle
    _dynamic_weighted_metric_registry[key] = class_name
    return cls  # type: ignore


def DynamicWeightedEvalMetric(
        loss_names: Sequence[str]
        ) -> Type[base.WeightedMetric]:
    key = tuple(loss_names)
    if key in _dynamic_weighted_eval_metric_registry:
        return globals()[_dynamic_weighted_eval_metric_registry[key]]

    TrainMetric = DynamicWeightedMetric(loss_names)
    class_name = f"DynamicWeightedEvalMetric_{'_'.join(key)}"
    # Merge fields for EvalMetric
    fields_dict = {**TrainMetric.fields, **EvalMetric.fields}
    cls = type(class_name, (TrainMetric, EvalMetric), {"fields": fields_dict})
    globals()[class_name] = cls  # top-level reference
    _dynamic_weighted_eval_metric_registry[key] = class_name
    return cls  # type: ignore
