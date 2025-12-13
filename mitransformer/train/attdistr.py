from .. import models

import torch

from typing import Literal


def normalise(
        probs: torch.Tensor,
        without_diagonal: bool = False,
        without_dummy_prefixes: int = 0) -> torch.Tensor:
    if not without_diagonal and without_dummy_prefixes == 0:
        return probs

    if without_diagonal:
        probs = torch.tril(probs, diagonal=-1)
    if without_dummy_prefixes > 0:
        probs[..., :without_dummy_prefixes] = 0
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-4)

    if without_diagonal:
        probs = torch.tril(probs, diagonal=-1)
    if without_dummy_prefixes:
        probs[..., :without_dummy_prefixes] = 0
    return probs


def normalize_by_norms(tensor: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(tensor, dim=-1)
    return norms/norms.sum(dim=-1, keepdim=True)


def arc_distribution(
        additional: models.AdditionalResults,
        mode: Literal["att", "att-n"],
        without_diagonal: bool = False,
        without_dummy_prefixes: int = 0
        ) -> torch.Tensor:
    """additional can contain:
    att (required): (... s s)
    proj_states: (... s s mhe)

    returns: (... s s)
    """

    match mode:
        case "att":
            assert "att" in additional.keys()
            att = additional["att"]  # type: ignore

        case "att-n":
            assert "proj_states" in additional.keys()
            proj_states = additional["proj_states"]  # type: ignore
            att = normalize_by_norms(proj_states)

        case _:
            raise Exception

    return normalise(
        att, without_diagonal=without_diagonal,
        without_dummy_prefixes=without_dummy_prefixes)


def merge_layer_heads(stack: torch.Tensor) -> torch.Tensor:
    """input: (l b mh ...)
    (b lmh ...) """

    stack = stack.transpose(0, 1)
    # (l b mh ...) -> (b l mh ...)
    stack = stack.contiguous().view(
        stack.shape[0], stack.shape[1]*stack.shape[2], *stack.shape[3:])
    # (b l mh ...) -> (b lmh ...)
    return stack
