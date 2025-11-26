from . import base, field

import torch

# ---------------------- concrete metric classes ---------------------------


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
        **base.WeightedMetric.fields,
        "attention_entropy_loss": field.attention_entropy_loss,
        "distance_loss": field.distance_loss
    }


class CostsEvalMetric(CostsMetric, EvalMetric):
    fields: dict[str, field.MetricField] = {
        **CostsMetric.fields, **EvalMetric.fields,
    }
