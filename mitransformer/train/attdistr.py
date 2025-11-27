from .. import models

import torch

from typing import Literal


def normalize_by_norms(tensor: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(tensor, dim=-1)
    return norms/norms.sum(dim=-1, keepdim=True)


def arc_distribution(
        additional: models.AdditionalResults,
        mode: Literal["att", "att-n"]
        ) -> torch.Tensor:
    """additional can contain:
    att (required): (l b mh s s mhe)
    proj_states: (l b mh s s mhe)

    returns: (b lmh s s)
    """

    def merge_layer_heads(stack: torch.Tensor) -> torch.Tensor:
        # input: (l b mh ...)

        stack = stack.transpose(0, 1)
        # (l b mh ...) -> (b l mh ...)
        stack = stack.contiguous().view(
            stack.shape[0], stack.shape[1]*stack.shape[2], *stack.shape[3:])
        # (b l mh ...) -> (b lmh ...)
        return stack

    match mode:
        case "att":
            att = merge_layer_heads(additional["att"])
            return att

        case "att-n":
            assert "proj_states" in additional.keys()
            proj_states = merge_layer_heads(
                additional["proj_states"])  # type: ignore
            att = normalize_by_norms(proj_states)
            return att

        case _:
            raise Exception
