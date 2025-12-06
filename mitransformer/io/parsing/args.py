import torch

from ...data import (
    MasksSetting)
from ...models import (
    TransformerDescription)
from ...utils.params import Params, Undefined

import random
import os
import numpy as np
from dataclasses import dataclass

from typing import (
    Literal)


@dataclass
class ParserArgs(Params):
    mode: Literal[
        "train", "hyperopt", "dataprep", "test",
        "compare"]
    rank: int | None
    n_workers: int
    name: str
    device: str
    use_ddp: bool
    dataset_name: str
    max_len_train: None | int
    max_len_eval_test: None | int
    masked: bool
    vocab_size: int | None
    triangulate: int
    first_k: int | None
    first_k_eval_test: int | None
    connect_with_dummy: bool
    connect_with_self: bool
    masks_setting: MasksSetting
    seed: int


@dataclass
class TrainParserArgs(ParserArgs):
    n_runs: int

    model_name: str
    dependency_mode: Literal["supervised", "input", "standard"]
    batch_size: int
    use_steps: bool
    max_steps: int | None
    eval_interval: int
    early_stop_after: int | None
    early_stop_metric: str | None
    epochs: int
    gradient_acc: int | None
    learning_rate: float
    loss_alpha: float | None
    combined_loss: bool
    distr_mode: Literal["att", "att-n"]
    global_distr: bool
    length_weighted: bool
    include_current: bool
    losses: None | dict[str, float | int]
    arc_loss_weighted: bool
    discriminative: bool

    transformer_description: TransformerDescription | None
    layer_design: tuple[str, ...]
    use_standard: bool
    width: int
    depth: int
    unrestricted_before: int
    unrestricted_after: int
    d_ff_factor: int
    dropout: float | None
    dropout_attn: float | None
    dropout_resid: float | None
    dropout_ff: float | None
    dropout_embd: float | None
    dropout_lstm: float | None
    use_lstm: bool
    block_size: int
    n_embd: int
    overlay_causal: bool
    use_dual_fixed: bool
    bias: bool
    pos_enc: Literal["embedding", "sinusoidal"]


@dataclass
class HyperoptParserArgs(ParserArgs):
    optimise: Literal["perplexity", "uas", "loss", "lm_loss", "arc_loss"]
    n_warmup_steps: int
    n_startup_trials: int
    n_trials: int

    dependency_mode: Literal["supervised", "input", "standard"]
    combined_loss: bool
    batch_size: int
    use_steps: bool
    max_steps: int | None
    eval_interval: int
    early_stop_after: int | None
    early_stop_metric: str | None
    epochs: int
    gradient_acc: int | None

    distr_mode: Literal["att", "att-n"] | list[Literal["att", "att-n"]]
    length_weighted: bool | list[bool]
    include_current: bool | list[bool]
    global_distr: bool | list[bool]
    learning_rate: float | tuple[float, float] | list[float]
    loss_alpha: float | tuple[float, float] | list[float | None] | None
    losses: list[dict[str, float | int]] | dict[str, float | int] | None

    arc_loss_weighted: bool | list[bool]
    discriminative: bool | list[bool]

    block_size: int
    overlay_causal: bool

    transformer_description: (
        TransformerDescription
        | list[TransformerDescription])
    layer_design: tuple[str, ...] | list[tuple[str, ...]]
    use_standard: bool | list[bool]
    width: int | tuple[int, int] | list[int]
    depth: int | tuple[int, int] | list[int]
    unrestricted_before: int | tuple[int, int] | list[int]
    unrestricted_after: int | tuple[int, int] | list[int]
    d_ff_factor: int | tuple[int, int] | list[int]
    dropout: float | tuple[int, int] | list[int | None] | None
    dropout_attn: float | tuple[int, int] | list[int | None] | None
    dropout_resid: float | tuple[int, int] | list[int | None] | None
    dropout_ff: float | tuple[int, int] | list[int | None] | None
    dropout_embd: float | tuple[int, int] | list[int | None] | None
    dropout_lstm: float | tuple[int, int] | list[int | None] | None
    use_lstm: bool | list[bool]
    n_embd: int | tuple[int, int] | list[int]
    use_dual_fixed: bool | list[bool]
    bias: bool | list[bool]
    pos_enc: (
        Literal["embedding", "sinusoidal"]
        | list[Literal["embedding", "sinusoidal"]])


@dataclass
class TestParserArgs(ParserArgs):
    model_name: str
    dependency_mode: Literal["supervised", "input", "standard"] | Undefined
    combined_loss: bool | Undefined
    distr_mode: Literal["att", "att-n"] | Undefined
    length_weighted: bool | Undefined
    include_current: bool | Undefined
    global_distr: bool | Undefined
    batch_size: int | Undefined
    loss_alpha: float | None | Undefined
    losses: dict[str, float | int] | None | Undefined
    arc_loss_weighted: bool | Undefined

    att_plot: bool
    tree_plot: bool


@dataclass
class DataprepParserArgs(ParserArgs):
    pass


@dataclass
class CompareParserArgs(ParserArgs):
    model1_name: str
    model2_name: str
    batch_size: int


@dataclass
class RTParserArgs(ParserArgs):
    n_runs: int
    lme: bool

    model_name: str
    dependency_mode: Literal["supervised", "input", "standard"] | Undefined
    combined_loss: bool | Undefined
    distr_mode: Literal["att", "att-n"] | Undefined
    length_weighted: bool | Undefined
    include_current: bool | Undefined
    global_distr: bool | Undefined
    batch_size: int | Undefined
    loss_alpha: float | None | Undefined
    losses: dict[str, float | int] | None | Undefined
    arc_loss_weighted: bool | Undefined

    shift: int
    only_content_words_cost: bool
    only_content_words_left: bool
    mapper: str


def seed_everything(seed: int):
    """There might be nondeterministic torch algorithms.
    We're not making them deterministic here."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # according to
    # https://pytorch.org/docs/stable/data.html#data-loading-randomness
    # each dataloader worker will have its PyTorch seed set to
    # base_seed + worker_id. Thus, with the same number
    # of workers, the process is deterministic


def make_device_str(string: str) -> str:
    try:
        return "cuda:" + str(int(string))
    except ValueError:
        return string


def args_logic(args: (
        TrainParserArgs | HyperoptParserArgs
        | DataprepParserArgs | TestParserArgs
        | CompareParserArgs | RTParserArgs)
        ) -> None:
    seed_everything(args.seed)
    args.device = make_device_str(args.device)
    if isinstance(args, TrainParserArgs):
        # parse transformer description

        # override dropout that was not set specifically
        for specific_dropout in (
                "dropout_attn", "dropout_resid",
                "dropout_ff", "dropout_embd",
                "dropout_lstm"):
            if getattr(args, specific_dropout) == -1:
                setattr(args, specific_dropout, args.dropout)

    if isinstance(args, (TrainParserArgs, TestParserArgs)):
        if args.model_name is None:
            args.model_name = args.name

    if args.rank is None and args.use_ddp:
        try:
            args.rank = int(os.environ["LOCAL_RANK"])
        except AttributeError:
            pass
