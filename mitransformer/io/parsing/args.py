import torch

from . import hyperopt
from ...data import (
    MasksSetting, RTCorpus)
from ...models import (
    TransformerDescription)
from ...utils.params import Params, Undefined
from ...readingtimes.lme import ParseResult as LMEParseResults

import random
import os
import numpy as np
from dataclasses import dataclass

from typing import (
    Literal, Tuple)


@dataclass
class ParserArgs(Params):
    mode: Literal[
        "train", "hyperopt", "dataprep", "test",
        "compare", "split"]
    rank: int | None
    n_workers: int
    name: str
    device: str
    use_ddp: bool
    use_amp: bool
    dataset_name: str
    max_len_train: None | int
    max_len_eval_test: None | int
    min_len_train: None | int
    min_len_eval_test: None | int
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
    k_negatives: None | int
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
    optimise: Literal[
        "perplexity", "uas", "loss", "lm_loss", "arc_loss", "loglik"]
    psyling_eval: bool
    n_warmup_steps: int
    sampler_startup_trials: int
    pruner_startup_trials: int
    n_trials: int
    psyling_dataset: Tuple[RTCorpus, ...]
    average_psyling: bool
    load_psyling_mmap: str | None
    lme_formula: LMEParseResults
    shift: int
    sampler: Literal["tpe", "random"]
    pruner: Literal["hyperband", "median"]

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

    distr_mode: Literal["att", "att-n"] | hyperopt.Choices[
        Literal["att", "att-n"]]
    length_weighted: bool | hyperopt.Choices[bool]
    include_current: bool | hyperopt.Choices[bool]
    global_distr: bool | hyperopt.Choices[bool]
    learning_rate: float | hyperopt.Range | hyperopt.Choices[float]
    loss_alpha: float | hyperopt.Range | hyperopt.Choices[float | None] | None
    losses: list[dict[
        str, float | int | None]] | dict[str, float | int | None] | None
    k_negatives: None | int | hyperopt.Choices[int | None] | hyperopt.Range

    arc_loss_weighted: bool | hyperopt.Choices[bool]
    discriminative: bool | hyperopt.Choices[bool]

    block_size: int
    overlay_causal: bool

    transformer_description: (
        TransformerDescription
        | list[TransformerDescription])
    layer_design: tuple[str, ...] | hyperopt.Choices[tuple[str, ...]]
    use_standard: bool | hyperopt.Choices[bool]
    width: int | hyperopt.Range | hyperopt.Choices[int]
    depth: int | hyperopt.Range | hyperopt.Choices[int]
    unrestricted_before: int | hyperopt.Range | hyperopt.Choices[int]
    unrestricted_after: int | hyperopt.Range | hyperopt.Choices[int]
    d_ff_factor: int | hyperopt.Range | hyperopt.Choices[int]
    dropout: float | hyperopt.Range | hyperopt.Choices[float | None] | None
    dropout_attn: float | hyperopt.Range | hyperopt.Choices[
        float | None] | None
    dropout_resid: float | hyperopt.Range | hyperopt.Choices[
        float | None] | None
    dropout_ff: float | hyperopt.Range | hyperopt.Choices[float | None] | None
    dropout_embd: float | hyperopt.Range | hyperopt.Choices[
        float | None] | None
    dropout_lstm: float | hyperopt.Range | hyperopt.Choices[
        float | None] | None
    use_lstm: bool | hyperopt.Choices[bool]
    n_embd: int | hyperopt.Range | hyperopt.Choices[int]
    use_dual_fixed: bool | hyperopt.Choices[bool]
    bias: bool | hyperopt.Choices[bool]
    pos_enc: (
        Literal["embedding", "sinusoidal"]
        | hyperopt.Choices[Literal["embedding", "sinusoidal"]])


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
    k_negatives: None | int | Undefined
    arc_loss_weighted: bool | Undefined

    att_plot: bool
    tree_plot: bool


@dataclass
class DataprepParserArgs(ParserArgs):
    pass


@dataclass
class SplitParserArgs(ParserArgs):
    proportion: float


@dataclass
class CompareParserArgs(ParserArgs):
    model1_name: str
    model2_name: str
    batch_size: int


@dataclass
class RTParserArgs(ParserArgs):
    n_runs: int
    lme: bool
    to_add: tuple[str, ...] | None
    legacy_process: bool

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
    k_negatives: None | int | Undefined
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
        | CompareParserArgs | RTParserArgs | SplitParserArgs)
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
