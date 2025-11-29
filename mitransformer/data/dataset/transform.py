
import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod

from typing import (Callable, Mapping,
                    Concatenate)

from ...utils.logmaker import getLogger

logger = getLogger(__name__)


TransformFunc = Callable[
    [npt.NDArray[np.bool_]],
    Mapping[str, npt.NDArray[np.bool_]]]


class MaskTransform(ABC):
    @abstractmethod
    def __call__(
            self, mask: npt.NDArray[np.bool_],
            ) -> dict[str, npt.NDArray[np.bool_]]:
        ...


class TransformMaskHeadChild(MaskTransform):
    def __init__(
            self,
            keys_for_head: set[str] = {"head"},
            keys_for_child: set[str] = {"child"},
            triangulate: int | None = 0,
            connect_with_dummy: bool = True,
            connect_with_self: bool = False,):
        assert not (connect_with_dummy and connect_with_self), (
            "You cannot represent non-existant arcs both with "
            "arcs to the dummy node and with self-arcs."
        )

        self.keys_for_head = keys_for_head
        self.keys_for_child = keys_for_child
        self.triangulate = triangulate
        self.connect_with_dummy = connect_with_dummy
        self.connect_with_self = connect_with_self

    def __call__(
            self,
            mask: npt.NDArray[np.bool_],
            ) -> dict[str, npt.NDArray[np.bool_]]:
        """Assumes a dummy token in the beginning. Therefore: one needs to
        add arcs for nodes that
        do not have a left head and nodes that do not have left children.

        ATTENTION: modifies the matrix inplace."""

        head = mask
        child = mask.T
        # head[range(len(head)), range(len(head))] = True
        # child[range(len(child)), range(len(child))] = True

        if self.connect_with_dummy:
            tril_head = np.tril(head, -1)
            set_true = ~tril_head.any(1)
            head[:, 0] = np.logical_or(set_true, head[:, 0])

            tril_child = np.tril(child, -1)
            set_true = ~tril_child.any(1)
            child[:, 0] = np.logical_or(set_true, child[:, 0])

        if self.connect_with_self:
            child = child.copy()
            tril_head = np.tril(head, -1)
            length = head.shape[0]
            set_true = ~tril_head.any(1)
            head[
                np.arange(0, length),
                np.arange(0, length)] = np.logical_or(
                    set_true,
                    head.diagonal(
                        axis1=-1,
                        axis2=-2
                    ))
            tril_child = np.tril(child, -1)
            length = head.shape[0]
            set_true = ~tril_child.any(1)
            child[
                np.arange(0, length),
                np.arange(0, length)] = np.logical_or(
                    set_true,
                    child.diagonal(
                        axis1=-1,
                        axis2=-2
                    ))

        if self.triangulate is not None:
            head = np.tril(head, self.triangulate)
            child = np.tril(child, self.triangulate)

        out_dict: dict[str, npt.NDArray[np.bool_]] = {}

        for key in self.keys_for_head:
            out_dict[key] = head

        for key in self.keys_for_child:
            out_dict[key] = child
        # print(child)

        return out_dict


class TransformMaskFull(MaskTransform):
    def __init__(
            self, keys_for_empty: set[str] = {"standard"}):
        self.keys_for_empty = keys_for_empty

    def __call__(
            self,
            mask: npt.NDArray[np.bool_],
            ) -> dict[str, npt.NDArray[np.bool_]]:
        """No arcs are masked"""

        # maybe this should produce all True matrices to easy collation
        trues = np.full(mask.shape, True)
        return {key: trues for key in self.keys_for_empty}


def transform_combined(
        mask: npt.NDArray[np.bool_],
        funcs_and_args: set[
            tuple[
                Callable[
                    Concatenate[npt.NDArray[np.bool_], ...],
                    Mapping[str, npt.NDArray[np.bool_]]],
                dict[str, npt.NDArray[np.bool_]]]]
        ) -> dict[str, npt.NDArray[np.bool_]]:

    out_dict: dict[str, npt.NDArray[np.bool_]] = {}

    for func, kwargs in funcs_and_args:
        out_dict.update(func(mask, **kwargs))

    return out_dict
