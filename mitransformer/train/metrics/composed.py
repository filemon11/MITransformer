from . import base, field

import torch

from typing import Any

# ---------------------- concrete metric classes ---------------------------


class SupervisedMetric(base.Metric):
    fields = {
        **base.Metric.fields,
        "arc_num": field.num,
        "arc_loss": field.loss("arc_num"),
        "alpha": field.MetricField(None, static=True),
    }

    def _compute_loss(self) -> torch.Tensor:
        lm_loss = getattr(self, "lm_loss")
        arc_loss = getattr(self, "arc_loss")
        alpha = getattr(self, "alpha")

        return (alpha*lm_loss + (1-alpha)*arc_loss)


class EvalMetric(base.Metric):
    fields = {
        **base.Metric.fields,
        "perplexity": field.perplexity,
    }


class SupervisedEvalMetric(SupervisedMetric, EvalMetric):
    # merge field maps carefully: rightmost wins
    fields: dict[str, field.MetricField] = {
        **SupervisedMetric.fields, **EvalMetric.fields,
        "uas": field.uas,
        "att_entropy": field.att_entropy,
    }


class CostsMetric(base.Metric):
    fields = {
        **base.Metric.fields,
        "attention_entropy_loss": field.attention_entropy_loss,
        "distance_loss": field.distance_loss,
        "w1": field.weight,
        "w2": field.weight,
        "w3": field.weight,
    }

    def _compute_loss(self) -> torch.Tensor:
        # custom composition: w1*lm + w2*att + w3*distance

        lm = getattr(self, "_lm_loss")
        att = getattr(self, "_attention_entropy_loss")
        dist = getattr(self, "_distance_loss")
        w1 = getattr(self, "w1")
        w2 = getattr(self, "w2")
        w3 = getattr(self, "w3")
        # coerce to tensors

        def _to_t(x: Any) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x
            try:
                return torch.tensor(float(x))
            except Exception:
                return torch.tensor(0.)
        return (w1 * _to_t(lm) + w2 * _to_t(att) + w3 * _to_t(dist))


class CostsEvalMetric(CostsMetric, EvalMetric):
    fields: dict[str, field.MetricField] = {
        **CostsMetric.fields, **EvalMetric.fields,
        "att_entropy": field.att_entropy}
