from .. import models

import torch

from typing import Literal


def normalize_by_norms(tensor):
    norms = torch.norm(tensor, dim=-1)
    return norms/norms.sum(dim=-1, keepdims=True)


def arc_distribution(
        additional: models.AdditionalResults,
        mode: Literal["attn", "attn-n"]
        ) -> torch.Tensor:
    """additional can contain:
    att (required): (l b mh s s mhe)
    proj_states: (l b mh s s mhe)
    """

    def merge_layer_heads(stack: torch.Tensor) -> torch.Tensor:
        # input: (l b mh s s mhe)

        # TODO: implement other options
        stack = stack.permute(1, 0, 2, 3, 4, 5)
        # (l b mh s s mhe) -> (b l mh s s mhe)
        stack = stack.view(
            stack.shape[0], stack.shape[1]*stack.shape[2], *stack.shape[3:])
        # (b l mh s s mhe) -> (b lmh s s mhe)
        return stack

    match mode:
        case "attn":
            att = merge_layer_heads(additional["att"])
            return att

        case "attn-n":
            assert "proj_states" in additional.keys()
            proj_states = merge_layer_heads(additional["proj_states"])
            print(proj_states.shape)  # TODO: check whether view is correct...
            # TODO
            return proj_states
