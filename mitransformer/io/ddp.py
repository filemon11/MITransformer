import torch
import torch.distributed as dist

from contextlib import contextmanager

from mitransformer.utils.logmaker import (
    getLogger, info)

from typing import (
    Iterator)

logger = getLogger(__name__)


def setup_group(world_size, backend: str = "gloo") -> dist.ProcessGroup | None:
    if world_size > 1:
        info(
            None, logger,
            f"Initialising process group with backend {backend}")
        pg = dist.new_group(backend=backend)
        return pg
    else:
        return None


def clean_group(world_size, pg) -> None:
    if world_size > 1:
        dist.destroy_process_group(pg)


@contextmanager
def new_pg(
        world_size, backend: str = "gloo") -> Iterator[
            dist.ProcessGroup
            | None]:
    pg = None
    try:
        pg = setup_group(world_size, backend)
        yield pg
    finally:
        clean_group(world_size, pg)


def setup_ddp(rank, world_size, backend: str = "nccl") -> bool:
    if world_size > 1:
        info(
            None, logger,
            (
                f"Initialising process group with backend {backend}, "
                f"world size {world_size} and rank {rank}."))
        dist.init_process_group(backend, world_size=world_size, rank=rank)
        torch.cuda.set_device(torch.distributed.get_rank())
        return True
    else:
        return False


def clean_ddp(world_size, pg=None) -> None:
    """if None then destroy dist.group.WORLD"""
    if world_size > 1:
        pg = dist.group.WORLD if pg is None else pg
        clean_group(world_size, pg)


@contextmanager
def ddp(rank: int | None, world_size: int) -> Iterator[bool]:
    try:
        yield setup_ddp(rank, world_size)
    finally:
        clean_ddp(world_size)
