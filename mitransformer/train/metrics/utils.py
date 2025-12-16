import torch
import numbers
import numpy as np

from typing import Any


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


minimise = {"lm_loss": True,
            "loss": True,
            "arc_loss": True,
            "perplexity": True,
            "uas": False,
            "attention_distance_loss": True,
            "attention_difference_loss": True,
            "attention_activation_loss": True,
            "attention_entropy_loss": True,
            "cosine_loss": True,
            }


def to_t(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    try:
        return torch.tensor(float(x))
    except Exception:
        return torch.tensor(0.)
