from .arc import arc_loss  # noqa: F401
from .lm import lm_loss  # noqa: F401
from .attention import (  # noqa: F401
    attention_entropy_loss, attention_distance_loss,
    attention_difference_loss, get_attention_entropy,
    attention_activation_loss)
from .utils import entropy  # noqa: F401
from .cosine import cosine_loss  # noqa: F401
