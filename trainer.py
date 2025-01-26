from model import MITransformer, MITransformerLM, MITransformerConfig
from data import (DepDataset, DUMMY, ROOT, EOS,
                  TransformMaskHeadChild, CoNLLUDataset,
                  DataLoader, get_loader, IDSen, IDBatch,
                  CoNNLUTokenisedBatch, EssentialBatch, D)
from tokeniser import TokenMapper
from parse import parse_list_of_words_with_spacy
from dependencies import (mst, merge_head_child_scores,
                          dummy_mask_removal, mask_to_headlist, uas_absolute)
from params import Params

import numpy as np
import torch
from torch.optim.adam import Adam
from torch.optim import Optimizer
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard.writer import SummaryWriter

import math
from pathlib import Path
import os
import pickle

from dataclasses import dataclass, fields, field
from functools import total_ordering
from contextlib import contextmanager
from collections import defaultdict

from typing import (Self, Literal, cast,
                    Container, Iterable, Mapping,
                    ClassVar, TypeVar, Callable,
                    Any, Generator, TypedDict, NotRequired)

from logmaker import getLogger, info, warning, get_timestr

logger = getLogger(__name__)

Mode = Literal["standard", "input", "supervised"]
M = TypeVar("M", bound="Metric")
N = TypeVar("N")


def sum_metrics(metrics: Iterable[M]) -> M:
    """Computes mean wrt. elements."""
    s: None | M = None
    for m in metrics:
        if s is None:
            s = m
        else:
            s = cast(M, m + s)
    assert s is not None, "Iterable of metrics cannot be empty."
    return s


def sum_and_std_metrics(
        metrics: "Iterable[Metric]"
        ) -> dict[str, tuple[float, float]]:
    ms = list(metrics)
    n = len(ms)
    out_dict: dict[str, tuple[float, float]] = dict()
    means: dict[str, float] = sum_metrics(ms).to_dict()
    for key, mean_value in means.items():
        xs = [getattr(m, key) for m in ms]
        out_dict[key] = (mean_value,
                         math.sqrt(sum([(x-mean_value)**2 for x in xs]) / n))
    return out_dict


minimise = {"lm_loss": True,
            "loss": True,
            "arc_loss": True,
            "perplexity": True,
            "uas": False,
            }


@total_ordering
@dataclass
class Metric(Params):
    num: float = 0
    _lm_loss: torch.Tensor = torch.tensor(0)

    _to_mean: ClassVar[set[str]] = {"lm_loss", "loss"}

    _convert: ClassVar[dict[str, Callable[[N], N]]] = {}    # type: ignore

    main_metric: str = "loss"

    # Whether optimisation means minimising (if False: maximising)
    minimise: ClassVar[dict[str, bool]] = minimise
    loss: ClassVar[torch.Tensor]
    # just for typing so that we can safely call metric.loss.backward()
    # without the type checker complaining

    @property
    def device(self) -> torch.device:
        return self.loss.device

    @property
    def is_cuda(self) -> bool:
        return self.loss.is_cuda

    def minval(self) -> float:
        return math.inf if self.minimise[self.main_metric] else -math.inf

    def maxval(self) -> float:
        return -self.minval()

    def __getattr__(self, prop: str):
        """Calculate mean for metrics"""
        if prop in self._to_mean:
            val = self.__getattribute__(f"_{prop}") / self.num
            if prop in self._convert:
                val = self._convert[prop](val)
            return val
        else:
            raise AttributeError(
                f"'{self.__class__}' has no attribute '{prop}' or '_{prop}'.")

    def _add_fields(self,
                    name: str,
                    m1: "Metric",
                    m2: "Metric") -> float | torch.Tensor:
        val1 = getattr(m1, name)
        val2 = getattr(m2, name)
        if name == "main_metric":
            assert val1 == val2, "Main metrics do not correspond!"
            return val1
        else:
            return val1 + val2

    def __add__(self, other: "Metric") -> "Metric":
        # other must be a lower type or Self

        higher = None
        if (isinstance(self, other.__class__)
                and not isinstance(other, self.__class__)):
            return other.__add__(self)
        elif isinstance(other, self.__class__):
            higher = self
        assert higher is not None, (f"Cannot add metrics "
                                    f"of types {self.__class__} "
                                    f"and {other.__class__}")

        return higher.__class__(
            **{f.name: self._add_fields(f.name, self, other)  # type: ignore
               for f in fields(higher)})

    def __radd__(self, other: "Metric") -> "Metric":
        return self + other

    def __truediv__(self, other: float) -> "Metric":
        new = self.__class__(main_metric=self.main_metric) + self
        new.num *= other
        return new

    def print(self, epoch: int,
              total_epochs: int, kind: str) -> None:
        strs = [f"{name}: {getattr(self, name):.2f}" for name in self._to_mean]
        print(f"[{epoch}/{total_epochs}] {kind}:: " + ", ".join(strs))

    def print_test(self) -> None:
        strs = [f"{name}: {getattr(self, name):.2f}" for name in self._to_mean]
        print("Test results: " + ", ".join(strs))

    @property
    def _loss(self) -> torch.Tensor:
        return self._lm_loss

    def detach(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                setattr(self, f.name, value.detach())

    def to_(self, device: str | torch.device) -> None:
        """Inplace"""
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                setattr(self, f.name, value.to(device))

    def to(self, device: str | torch.device) -> Self:
        return self.__class__(
            **{f.name: (getattr(self, f.name).to(device)
                        if isinstance(getattr(self, f.name), torch.Tensor)
                        else getattr(self, f.name)) for f in fields(self)})

    @property
    def main_value(self) -> torch.Tensor:
        return getattr(self, self.main_metric)

    def __gt__(self, other: object) -> bool:
        factor = -1 if self.minimise[self.main_metric] else 1

        self_attr = (self.main_value.item()
                     if isinstance(self.main_value, torch.Tensor)
                     else self.main_value)

        if isinstance(other, Metric):
            other_attr = (other.main_value.item()
                          if isinstance(other.main_value, torch.Tensor)
                          else other.main_value)

            return factor*self_attr > factor*other_attr
        else:
            try:
                return (factor*self_attr
                        > factor*float(other))  # type: ignore
            except ValueError:
                return False

    def __eq__(self, other: object) -> bool:
        factor = -1 if self.minimise[self.main_metric] else 1

        self_attr = (self.main_value.item()
                     if isinstance(self.main_value, torch.Tensor)
                     else self.main_value)
        if isinstance(other, Metric):
            other_attr = (other.main_value.item()
                          if isinstance(other.main_value, torch.Tensor)
                          else other.main_value)
            return factor*self_attr == factor*other_attr
        else:
            try:
                return factor*self_attr == factor*float(other)  # type: ignore
            except ValueError:
                return False

    def to_dict(self, as_str: bool = False,
                omit_undefined: bool = False) -> dict[str, Any]:
        if as_str:
            return {attr: str(getattr(self, attr))
                    for attr in self._to_mean}
        else:
            return {attr: float(getattr(self, attr))
                    for attr in self._to_mean}


@dataclass
class SupervisedMetric(Metric):
    _arc_loss: torch.Tensor = torch.tensor(0)
    alpha: float | None = None
    _to_mean: ClassVar[set[str]] = Metric._to_mean | {"arc_loss"}

    @property
    def _loss(self) -> torch.Tensor:
        if self.alpha is None:

            warning(None, logger, "SupervisedMetric.alpha is None!")
            return (self._lm_loss
                    + self._arc_loss)
        else:
            return (self.alpha*self._lm_loss
                    + (1-self.alpha)*self._arc_loss)

    def _add_fields(self,
                    name: str,
                    m1: "Metric",
                    m2: "Metric") -> float | torch.Tensor:
        if name == "alpha":
            val1 = getattr(m1, name)
            val2 = getattr(m2, name)
            assert (val1 == val2 or val2 is None
                    or val1 is None), (
                        "Cannot combine metrics with different alpha."
                    )
            return val2 if val2 is not None else val1

        return super()._add_fields(name, m1, m2)

    def __add__(self, other: Metric) -> Metric:
        # other must be a lower type or Self

        higher = None
        if (isinstance(self, other.__class__)
                and not isinstance(other, self.__class__)):
            return other.__add__(self)
        elif isinstance(other, self.__class__):
            higher = self
        assert higher is not None, (f"Cannot add metrics "
                                    f"of types {self.__class__} "
                                    f"and {other.__class__}")
        other = cast(Self, other)

        return higher.__class__(
            **{f.name: self._add_fields(f.name, self, other)  # type: ignore
               for f in fields(higher)})


@dataclass
class EvalMetric(Metric):
    _perplexity: float = 0
    _to_mean: ClassVar[set[str]] = Metric._to_mean | {"perplexity"}
    _convert = Metric._convert | {"perplexity": math.exp}    # type: ignore


@dataclass
class SupervisedEvalMetric(SupervisedMetric, EvalMetric):
    _uas: float = 0
    _to_mean: ClassVar[set[str]] = (SupervisedMetric._to_mean
                                    | EvalMetric._to_mean
                                    | {"uas"})


class MetricWriter(SummaryWriter):
    def add_metric(
            self,
            metric: Metric,
            epoch: int,
            split: Literal["train", "eval", "test"]
            ) -> None:
        for key, value in metric.to_dict().items():
            self.add_scalar(f"{key}/{split}", value, epoch)

    def add_params(
            self,
            params: Mapping[str, Any],
            metric: Metric,
            run_name: str | None = None,
            global_step: int | None = None,
            ) -> None:
        def check_type(value: Any) -> bool:
            if (isinstance(value, float)
                    or isinstance(value, int)
                    or isinstance(value, torch.Tensor)
                    or isinstance(value, bool)
                    or isinstance(value, str)):
                return True
            return False
        self.add_hparams(
            {key: value for key, value
             in params.items() if check_type(value)},
            {f"_{key}": value for key, value in metric.to_dict().items()},
            run_name=run_name,
            global_step=global_step)


@contextmanager
def metric_writer(*args, **kwds):
    # Code to acquire resource, e.g.:
    writer = MetricWriter(*args, **kwds)
    try:
        yield writer
    finally:
        # Code to release resource, e.g.:
        writer.flush()


class Result(TypedDict):
    train: Metric
    eval: Metric


class TestResult(Result):
    test: NotRequired[Metric]


@dataclass
class GeneralConfig(Params):
    batch_size: int = 16
    dependency_mode: Mode = "supervised"
    loss_alpha: float | None = 0.5
    arc_loss_weighted: bool = False
    device: str | int = "cpu"
    rank: int | None = None
    world_size: int = 1
    n_workers: int = 0
    model_name: str = field(default_factory=get_timestr)
    early_stop_metric: str = "loss"
    use_ddp: bool = False
    discriminative: bool = False


@dataclass
class TrainConfig(GeneralConfig):
    eval_interval: int = 1
    epochs: int = 100
    learning_rate: float = 1e-3
    early_stop_after: int | None = 1
    use_steps: bool = False
    max_steps: int | None = None


class LMTrainer():
    model_dir: str = "./models/"

    def __init__(self, transformerlm: MITransformerLM,
                 transformer_config: MITransformerConfig,
                 config: GeneralConfig):
        self.writer = MetricWriter(
            log_dir=os.path.join("./runs", config.model_name))
        self.transformerlm: MITransformerLM | DDP = transformerlm
        self.transformerlm.to(config.device)
        self.transformer_config: MITransformerConfig = transformer_config

        self.optimiser: Optimizer | None
        self.__config: GeneralConfig
        self.config = config

        rank = config.rank
        device = config.device
        self.use_ddp = config.use_ddp

        self.transformerlm.to(config.device)
        if self.use_ddp:
            self.transformerlm = DDP(
                self.transformerlm,
                device_ids=[rank],
                output_device=device,
                find_unused_parameters=False)

    @property
    def config(self) -> GeneralConfig:
        return self.__config

    @config.setter
    def config(self, config: GeneralConfig) -> None:
        self.__config = config
        if hasattr(config, "learning_rate"):
            self.optimiser = Adam(
                self.transformerlm.parameters(),
                lr=config.learning_rate)  # type: ignore
        else:
            self.optimiser = None

    @property
    def train_config(self) -> TrainConfig | None:
        if hasattr(self.config, "learning_rate"):
            return cast(TrainConfig, self.config)
        else:
            return None

    @classmethod
    def load_model(cls, model_name: str, device: str = "cpu"
                   ) -> tuple[MITransformerLM, MITransformerConfig]:
        state_dict, transformer_config = torch.load(
            os.path.join(cls.model_dir, model_name, "model"),
            map_location=device).values()
        transformer_config = cast(MITransformerConfig, transformer_config)
        model: MITransformerLM = MITransformerLM(
            MITransformer(transformer_config))
        model.load_state_dict(state_dict)
        return model, transformer_config

    @classmethod
    def load(cls, model_name: str,
             device: str = "cpu",
             **optional_config: Any) -> Self:

        model, transformer_config = cls.load_model(model_name,
                                                   device)

        with open(os.path.join(
                    cls.model_dir, model_name, "config"), 'rb') as handle:
            config: GeneralConfig = pickle.load(handle)

        config.update_from_kwargs(device=device, **optional_config)

        cls.model_info(model, transformer_config, config)

        return cls(model, transformer_config, config)

    @classmethod
    def new(cls, transformer_config: MITransformerConfig,
            config: GeneralConfig) -> Self:

        model: MITransformerLM = MITransformerLM(
            MITransformer(transformer_config))

        cls.model_info(model, transformer_config, config)
        return cls(model, transformer_config, config)

    @classmethod
    def model_info(cls, model: MITransformerLM,
                   transformer_config: MITransformerConfig,
                   config: GeneralConfig) -> None:
        info(config.rank, logger,
             "Initialised model with params:\n" +
             transformer_config.info)

        model_parameters = filter(
            lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        info(config.rank, logger, f"Number of parameters: {params}")

        info(config.rank, logger,
             "Initialised trainer with params:\n" + config.info)

    def save(self) -> None:
        assert self.train_config is not None
        if self.use_ddp:
            dist.barrier()
        if not self.use_ddp or self.config.rank == 0:
            model = self.transformerlm
            if self.use_ddp:
                model = self.transformerlm.module
            dir = os.path.join(self.model_dir, self.train_config.model_name)
            Path(dir).mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(),
                 "config": self.transformer_config},
                os.path.join(dir, "model"))

            # overwrites config
            with open(os.path.join(dir, "config"), 'wb') as handle:
                pickle.dump(self.train_config,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(
            self, model_name: str | None = None) -> None:
        if model_name is None:
            model_name = self.config.model_name
        state_dict, _ = torch.load(
            os.path.join(self.model_dir, model_name, "model")).values()
        if self.use_ddp:
            self.transformerlm.module.load_state_dict(state_dict)
        else:
            self.transformerlm.load_state_dict(state_dict)
        self.transformerlm.to(self.config.device)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor,
             ignore_index: int = -100,
             reduction: Literal["sum", "mean"] = "mean"
             ) -> torch.Tensor:

        logits = torch.swapaxes(logits, 1, 2)
        if self.config.discriminative:
            probs = F.sigmoid(logits)
            mask = labels != ignore_index
            labels_without_negative = labels
            labels_without_negative[~mask] = 0  # to ignore later
            one_hot = F.one_hot(
                labels_without_negative,
                logits.shape[-2]).float()
            loss = F.binary_cross_entropy(
                probs.swapaxes(-1, -2), one_hot, reduction='none')

            # ignore padding tokens
            loss[~mask] = 0
            loss = loss.sum() / mask.sum()

        else:
            loss = F.cross_entropy(
                logits, labels,
                ignore_index=ignore_index,
                reduction=reduction)
        return loss

    def arc_loss(
            self, score_preds: torch.Tensor,
            score_gold: torch.BoolTensor,
            to_ignore_mask: torch.BoolTensor | None,
            reduction: Literal["sum", "mean"] = "mean"
            ) -> torch.Tensor:
        """reduction sum takes a mean across dim 1
        of the mask"""
        # TODO: what if we have more than two masks?
        masks_dim = 0
        batch_dim = 1
        seq1_dim = 2
        seq2_dim = 3
        shape = score_preds.shape

        M = shape[masks_dim]
        B = shape[batch_dim]
        S = shape[seq1_dim]

        # Calculation if unpadded (no ignore mask)
        total_len = B * S
        factor = int((S+1) / 2 * M)
        num_scores = total_len * factor

        if to_ignore_mask is not None:
            # assumes lens are the same for each M
            lens = (~to_ignore_mask).select(
                seq2_dim, 0).select(masks_dim, 0).sum(seq1_dim-1).float()
            # TODO: make it possbile to give lens as parameter
            # since we compute them already in normal loss calculation
            # and compute the to_ignore_mask here using broadcasting...

            total_len = int(lens.sum().item())
            # divide through M since each head mask contributes 1x total_len

            num_scores = (
                torch.dot((lens+1), lens) / 2 * M).item()  # type: ignore
            # divide score/factor by number of tokens to get average
            # per-token loss
            # score_gold = cast(torch.BoolTensor,
            #                  score_gold[~to_ignore_mask])

            # score_preds = score_preds[~to_ignore_mask]

        # print(score_preds.detach().cpu().numpy().round(2),
        #       score_gold.to(score_preds.dtype).detach().cpu().numpy())
        loss = F.binary_cross_entropy(
            score_preds,
            score_gold.to(score_preds.dtype),
            reduction='none')

        if self.config.arc_loss_weighted:
            true_el = torch.sum(score_gold)
            false_el = num_scores - true_el
            f_true = 0.5*num_scores/true_el
            f_false = 0.5*num_scores/false_el
            weights = torch.zeros(*score_preds.shape,
                                  device=score_preds.device)
            weights[score_gold] = f_true
            weights[~score_gold] = f_false
            loss *= weights

        loss = torch.sum(loss)
        if reduction == "mean":
            loss /= num_scores
        else:
            loss /= num_scores/total_len  # = factor
            # Each position can be attended to S+1 times
        return loss

    @staticmethod
    def filter_arc_scores(
            arc_scores: Mapping[str, list[torch.Tensor]],
            keep_keys: Container[str] | Iterable[str]
            ) -> dict[str, list[torch.Tensor]]:
        return {key: arc_scores[key]
                for key in arc_scores.keys() if key in keep_keys}

    @staticmethod
    def stack_pred_scores(
            arc_scores: Mapping[str, list[torch.Tensor]]
            ) -> torch.Tensor | None:
        if len(arc_scores) == 0:
            return None
        return torch.concat(
            [torch.stack(sc_list)
             for sc_list in arc_scores.values()])

    @staticmethod
    def expand_gold_scores(
            masks: Mapping[str, torch.BoolTensor],
            arc_scores: Mapping[str, list[torch.Tensor]]
            ) -> torch.BoolTensor | None:
        expanded = [gold.unsqueeze(0).expand(len(arc_scores), -1, -1, -1)
                    for gold, arc_scores in zip(
                        masks.values(),
                        arc_scores.values())]
        if len(expanded) == 0:
            return None
        return cast(torch.BoolTensor, torch.concat(expanded))

    @classmethod
    def align_scores(
            cls,
            arc_scores: Mapping[str, list[torch.Tensor]],
            masks: Mapping[str, torch.BoolTensor]
            ) -> tuple[torch.Tensor, torch.BoolTensor] | tuple[None, None]:
        score_preds = cls.stack_pred_scores(arc_scores)
        score_gold = cls.expand_gold_scores(masks, arc_scores)
        if score_preds is None or score_gold is None:
            return None, None
        else:
            return score_preds, score_gold

    @classmethod
    def prepare_scores(
            cls, arc_scores: Mapping[str, list[torch.Tensor]],
            masks: Mapping[str, torch.BoolTensor]
            ) -> tuple[torch.Tensor, torch.BoolTensor] | tuple[None, None]:
        arc_scores = cls.filter_arc_scores(
            arc_scores,
            set(masks.keys()))
        return cls.align_scores(arc_scores, masks)

    @staticmethod
    def get_ignore_mask(
            scores: torch.Tensor,
            label_ids: torch.Tensor,
            ignore_id: int) -> torch.BoolTensor:
        """TODO:  Shouldn't this return a triangle?"""

        not_to_ignore: torch.BoolTensor
        not_to_ignore = (label_ids == ignore_id)  # type: ignore

        not_to_ignore = not_to_ignore.logical_not()  # type: ignore
        not_to_ignore = not_to_ignore.unsqueeze(0).expand(  # type: ignore
            scores.shape[0], -1, -1)
        not_to_ignore_sq1 = not_to_ignore.unsqueeze(3).expand(  # type: ignore
            -1, -1, -1, scores.shape[2])

        not_to_ignore_sq2 = not_to_ignore_sq1.permute(0, 1, 3, 2)
        not_to_ignore = torch.logical_and(  # type: ignore
            not_to_ignore_sq1,
            not_to_ignore_sq2)
        return ~not_to_ignore  # type: ignore

    @classmethod
    def batch_to(cls, batch: dict, device) -> None:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)
            elif isinstance(value, dict):
                cls.batch_to(value, device)

    def get_metric(self,
                   num_instances: int,
                   lm_loss: torch.Tensor,
                   arc_loss: torch.Tensor | None = None,
                   perplexity: float | None = None,
                   uas: float | None = None) -> Metric:
        if perplexity is not None:
            if arc_loss is not None:
                assert uas is not None
                return SupervisedEvalMetric(
                    num_instances,
                    _lm_loss=lm_loss,
                    _perplexity=perplexity,
                    _arc_loss=arc_loss,
                    alpha=self.config.loss_alpha,
                    _uas=uas,
                    main_metric=self.config.early_stop_metric
                    )
            else:
                return EvalMetric(
                    num_instances, _lm_loss=lm_loss,
                    _perplexity=perplexity,
                    main_metric=self.config.early_stop_metric)

        if arc_loss is None:
            return Metric(
                num_instances, _lm_loss=lm_loss,
                main_metric=self.config.early_stop_metric)
        else:
            return SupervisedMetric(
                num_instances, _lm_loss=lm_loss, _arc_loss=arc_loss,
                alpha=self.config.loss_alpha,
                main_metric=self.config.early_stop_metric)

    def train_step(self,
                   batch: CoNNLUTokenisedBatch | EssentialBatch,
                   ignore_index: int) -> Metric:
        assert self.train_config is not None, "Config missing training params."
        assert self.optimiser is not None
        self.batch_to(batch, device=self.config.device)  # type: ignore
        logits, arc_scores = self.transformerlm(**batch)
        # remove from arc_scores those that should not be used...
        lm_loss = self.loss(
            logits, batch["label_ids"],
            ignore_index=ignore_index,
            reduction="sum")
        arc_loss: torch.Tensor | None = None

        if self.train_config.dependency_mode == "supervised":
            score_preds, score_gold = self.prepare_scores(
                arc_scores, batch["masks"])
            if score_preds is not None and score_gold is not None:

                to_ignore = self.get_ignore_mask(
                    score_preds,
                    batch["label_ids"],
                    ignore_index)

                arc_loss = self.arc_loss(
                    score_preds,
                    score_gold,
                    to_ignore,
                    reduction="sum")

        num_instances = int((batch["label_ids"] != ignore_index).sum().item())

        metric = self.get_metric(
            num_instances,
            lm_loss=lm_loss,
            arc_loss=arc_loss)

        metric.loss.backward()   # backward pass
        self.optimiser.step()   # update parameters
        self.optimiser.zero_grad(set_to_none=True)

        metric.detach()
        metric.to_("cpu")
        return metric

    def eval_step(self,
                  batch: CoNNLUTokenisedBatch | EssentialBatch,
                  mode: Mode,
                  ignore_index: int) -> Metric:
        self.batch_to(batch, device=self.config.device)  # type: ignore

        logits, arc_scores = self.transformerlm(**batch)
        # remove from arc_scores those that should not be used...

        labels = batch["label_ids"]
        lm_loss = self.loss(
            logits, labels,
            ignore_index=ignore_index, reduction="sum")

        # print(batch["masks"])

        num_instances = int((labels != ignore_index).sum().item())

        surprisal_sum = sum_depadded(
            logits_to_surprisal(
                logits, labels,
                ignore_index,
                softmax=not self.config.discriminative),
            labels, ignore_index).sum().detach().cpu().item()

        uas_abs = None
        arc_loss = None
        if mode == "supervised":
            score_preds, score_gold = self.prepare_scores(
                arc_scores, batch["masks"])
            if score_preds is not None and score_gold is not None:

                to_ignore = self.get_ignore_mask(
                    score_preds,
                    batch["label_ids"],
                    ignore_index)

                arc_loss = self.arc_loss(
                    score_preds,
                    score_gold,
                    to_ignore,
                    reduction="sum")

                # clean this up a bit

                # in case of multiple heads of the same type
                # scores get averaged
                middle = score_preds.shape[0]//2
                preds_head = score_preds[
                    :middle].mean(0).detach().cpu().numpy()
                golds_head = score_gold[
                    :middle].float().mean(0).detach().cpu().numpy()
                preds_child = score_preds[
                    middle:].mean(0).detach().cpu().numpy()
                golds_child = score_gold[
                    middle:].float().mean(0).detach().cpu().numpy()
                # print("golds_head", golds_head.astype(float))
                # print("golds_child", golds_child.astype(float))
                preds_arcs = dummy_mask_removal(
                    merge_head_child_scores(preds_head, preds_child))
                golds_arcs = dummy_mask_removal(
                    merge_head_child_scores(golds_head, golds_child))

                # print(preds_arcs.round(2))
                # print(golds_arcs.astype(float))
                not_padding = (batch["label_ids"]
                               != ignore_index).cpu().numpy()[:, 1:]

                uas_abs = sum(map(get_uas_abs, zip(
                    preds_arcs, golds_arcs, not_padding)))

                # # does not appear to be faster:
                # uas_abs_l = [0]
                # if self.config.rank == 0 or self.config.rank is None:
                #     pool = mp.Pool(
                #         processes=self.config.n_workers*self.config.world_size)
                #     uas_abs_l = [sum(
                #         pool.map(
                #             get_uas_abs, zip(
                #                 preds_arcs, golds_arcs, not_padding)))]
                #     pool.close()
                #
                # if self.use_ddp:
                #     dist.barrier()
                #     dist.broadcast_object_list(uas_abs, src=0)
                #     # to tensor better with NCCL?

                arc_loss = arc_loss

        metric = self.get_metric(
            num_instances,
            lm_loss=lm_loss,
            arc_loss=arc_loss,
            perplexity=surprisal_sum,
            uas=uas_abs)
        metric.to_("cpu")
        metric.detach()
        return metric

    def check_early_stop(
            self,
            evals_without_improvement: int) -> bool:
        assert self.train_config is not None
        early_stop_after = self.train_config.early_stop_after
        early_stop = (
            early_stop_after is not None
            and early_stop_after <= evals_without_improvement)
        early_stop = sum(self.gather_ddp(early_stop)) > 0
        if early_stop:
            info(self.config.rank, logger,
                 f"Aborting training after {evals_without_improvement} "
                 "evals without improvement.")
            return True
        return False

    def train_iter(
            self,
            train: DepDataset[IDSen] | DataLoader,
            eval: DepDataset[IDSen] | DataLoader,
            **kwargs) -> Generator[Result,
                                   None,
                                   tuple[tuple[int, int], tuple[int, int]]]:
        assert self.train_config is not None, "Config missing training params."
        train_config = self.train_config
        device = train_config.device
        assert device is not None

        train = self.get_loader(train)
        eval = self.get_loader(eval)

        eval_interval = train_config.eval_interval

        self.transformerlm.train()

        best: float | Metric | None = None
        evals_without_improvement: int = 0
        total_steps: int = 0
        break_training: bool = False
        max_epochs = (train_config.epochs
                      if train_config is not None
                      else train_config.max_steps)
        # since we cannot run out of epochs if we use
        # max_steps
        best_epoch: int = 0
        best_step: int = 0
        for epoch in range(1, max_epochs+1):
            if break_training:
                epoch -= 1
                break
            info(self.config.rank,
                 logger, f"Epoch: {epoch}/{max_epochs}")
            if self.use_ddp:
                train.sampler.set_epoch(epoch)  # type: ignore

            # Steps
            for train_metric in self._train(train):
                total_steps += 1  # equal epochs in case of not use_steps

                if total_steps % eval_interval == 0:
                    info(self.config.rank,
                         logger,
                         (f"Step: {total_steps}/" +
                          ('inf' if train_config.max_steps
                           is None  # type: ignore
                           else str(train_config.max_steps))))
                    self.log_metric(train_metric, total_steps, "train")
                    info(self.config.rank, logger,
                         f"train metric:\n{train_metric.info}")

                    eval_metric = self._eval(eval)
                    self.transformerlm.train()
                    self.log_metric(eval_metric, total_steps, "eval")
                    info(self.config.rank, logger,
                         f"eval metric:\n{eval_metric.info}")

                    if best is None:
                        best = eval_metric.minval()
                    if eval_metric > best:       # greater means better
                        best = eval_metric
                        self.save()

                        best_epoch = epoch
                        best_step = total_steps

                        info(self.config.rank, logger,
                             "Saving model at epoch "
                             f"{epoch} ({total_steps})...")
                        evals_without_improvement = 0
                    else:
                        evals_without_improvement += 1

                    yield {"train": train_metric,
                           "eval": eval_metric}

                    if self.check_early_stop(evals_without_improvement):
                        break_training = True
                        break
                    if (self.train_config.max_steps is not None
                            and total_steps
                            >= self.train_config.max_steps):
                        break_training = True
                        break

        return (epoch, total_steps), (best_epoch, best_step)

    def train(self,
              train: DepDataset[IDSen] | DataLoader,
              eval: DepDataset[IDSen] | DataLoader,
              test: DepDataset[IDSen] | DataLoader | None = None,
              **kwargs) -> TestResult:
        assert self.train_config is not None, "Config missing training params."

        train = self.get_loader(train)
        eval = self.get_loader(eval)

        # TODO: upgrade to Python 3.13 and replace with gen.report()
        gen = self.train_iter(train, eval, **kwargs)
        current: tuple[int, int]
        best: tuple[int, int]
        while True:
            try:
                _ = next(gen)
            except StopIteration as e:
                current, best = e.value
                break
        # load best (saved) into transformerlm
        self.load_state()

        info(self.config.rank, logger,
             f"Ended training after {current[0]} epochs, {current[1]} steps.")
        info(self.config.rank, logger,
             f"Found best model after {best[0]} epochs, {best[1]} steps.")

        return self.test(train=train,  # type: ignore
                         eval=eval,
                         test=test)
        # if score_preds is not None and score_gold is not None:
        #    token_mapper: TokenMapper = TokenMapper.load("./tokmap.pickle")
        #    for i in range(10):
        #        pred_head = score_preds[0][i].detach().cpu().numpy()
        #        gold_head = score_gold[0][i].detach().cpu().numpy()
        #        pred_child = score_preds[1][i].detach().cpu().numpy()
        #        gold_child = score_gold[1][i].detach().cpu().numpy()
        #        fig, ax = plt.subplots(2, 2)
        #        sns.heatmap(
        #            pred_head, ax=ax[0][0])
        #        sns.heatmap(
        #            gold_head, ax=ax[0][1])
        #        sns.heatmap(
        #            pred_child, ax=ax[1][0])
        #        sns.heatmap(
        #            gold_child, ax=ax[1][1])
        #        fig.savefig(f"plot_{i}.png")
        #
        #        not_padding = (batch["input_ids"][i]
        #                       != token_mapper.pad_id).cpu().numpy()
        #
        #        # TODO: use heuristic of leaving out root node
        #        # and then calculating MST
        #        pred_scores = merge_head_child_scores(
        #            pred_head, pred_child)[not_padding][:, not_padding]
        #        pred_scores = dummy_mask_removal(pred_scores)
        #        gold_scores = merge_head_child_scores(
        #            gold_head, gold_child)[not_padding][:, not_padding]
        #        gold_scores = dummy_mask_removal(gold_scores)
        #
        #        pred_headlist = mst(pred_scores)
        #        gold_headlist = mask_to_headlist(gold_scores)
        #
        #        ids = batch["input_ids"].detach().cpu().numpy()[
        #            i, not_padding][1:]
        #        labels = token_mapper.decode([ids.tolist()])[0]
        #        plot_tree(f"dep_plot_pred_{i}.png",
        #                  pred_headlist.tolist(),
        #                  labels)
        #
        #        plot_tree(f"dep_plot_gold_{i}.png",
        #                  gold_headlist.tolist(),
        #                  labels)

    def _train(self, loader: DataLoader[IDBatch, D]) -> Iterable[Metric]:
        assert self.train_config is not None, "Config missing training params."
        self.transformerlm.train()

        metrics: list[Metric]
        if self.train_config.use_steps:
            for batch in loader:
                metric = self.train_step(
                    batch,
                    loader.dataset.keys_for_padding["label_ids"])
                yield self.gather_metrics(metric)
        else:
            metrics = [
                self.train_step(
                    batch,
                    loader.dataset.keys_for_padding["label_ids"])
                for batch in loader]
            yield self.gather_metrics(sum_metrics(metrics))

    def _eval(self, loader: DataLoader[IDBatch, D]) -> Metric:
        self.transformerlm.eval()
        with torch.no_grad():
            # eval loop: no backprop on this data, to avoid storing
            # all intermediatte variable
            metrics = [
                self.eval_step(
                    batch,
                    self.config.dependency_mode,
                    loader.dataset.keys_for_padding["label_ids"])
                for batch in loader]
        return self.gather_metrics(sum_metrics(metrics))

    def test(self, **datasets: DepDataset | DataLoader | Any
             ) -> dict[str, Metric]:
        metrics = {n: self._eval(self.get_loader(ds))
                   for n, ds in datasets.items()
                   if isinstance(ds, (DataLoader, DepDataset))}
        for n, m in metrics.items():
            info(self.config.rank, logger,
                 f"Test metric for {n} split:\n{m.info}")
        return metrics

    def predict(
            self, dataset: DepDataset,
            make_prob: bool = False,
            only_true: bool = False
            ) -> tuple[list[torch.Tensor], dict[str, list[torch.Tensor]]]:
        """Returns logits and arc scores"""
        # TODO: Does this work with ddp? Batches are distributed but not
        # joined back together.
        loader = get_loader(
            dataset, batch_size=self.config.batch_size,
            bucket=False,
            shuffle=False, droplast=False,
            n_workers=self.config.n_workers)

        ignore_index = dataset.keys_for_padding["label_ids"]

        unpadded_logits: list[torch.Tensor] = []
        unpadded_arc_scores: defaultdict[str, list[torch.Tensor]]
        unpadded_arc_scores = defaultdict(list)
        self.transformerlm.eval()
        with torch.no_grad():
            # eval loop: no backprop on this data, to avoid storing
            # all intermediatte variable
            logits: torch.Tensor
            arc_scores: dict[str, list[torch.Tensor]]
            for batch in loader:
                self.batch_to(batch, device=self.config.device)  # type: ignore
                logits, arc_scores = self.transformerlm(**batch)
                labels = batch["label_ids"]

                if make_prob:
                    logits = logits_to_probs(
                        logits,
                        not self.config.discriminative)

                if only_true:
                    logits = select_true(logits, labels, ignore_index)

                unpadded_logits.extend(
                    unpad(logits, labels, ignore_index))

                for key in arc_scores.keys():
                    unpadded_arc_scores[key].extend(
                        unpad_masks(torch.stack(
                            arc_scores[key]).swapaxes(0, 1),
                            labels, ignore_index))
        return unpadded_logits, dict(unpadded_arc_scores)

    def generate(
            self, token_mapper: TokenMapper,
            start: str | None = None, max_len: int = 40) -> str:

        g: list[int]

        if self.use_ddp:
            model = self.transformerlm.module
        else:
            model = self.transformerlm

        if start is None:
            idx = torch.zeros((1, 2),
                              dtype=torch.long,
                              device=self.config.device)
            idx[0, 0] = token_mapper.token2id[DUMMY]
            idx[0, 1] = token_mapper.token2id[ROOT]
            g = model.generate(
                idx, max_new_tokens=max_len).tolist()[0]
            # support an initial mask here
        else:
            conllu = parse_list_of_words_with_spacy(start.split(), min_len=0)

            transform = TransformMaskHeadChild(
                keys_for_head={"head"},
                keys_for_child={"child"},
                triangulate=True)

            dataset: CoNLLUDataset = CoNLLUDataset.from_str(
                conllu, transform, max_len=None)

            dataset.map_to_ids(token_mapper)
            dataloader: DataLoader = get_loader(        # type: ignore
                dataset, batch_size=1,                  # type: ignore
                bucket=False, min_size=0, max_size=50,
                shuffle=False, droplast=False,
                n_workers=self.config.n_workers)

            for batch in dataloader:
                # take last batch, i.e. last sentence
                # TODO: this is a weird solution
                pass

            # TODO: support mask
            g = model.generate(batch["input_ids"][0],
                               max_new_tokens=max_len).tolist()[0]

        eos_id = token_mapper.token2id[EOS]
        first_eos = next((i for i, x in enumerate(g) if x == eos_id), len(g))
        g = g[:first_eos + 1]
        return token_mapper.decode([g], to_string=True)[0]

    def __del__(self) -> None:
        self.writer.flush()

    def gather_ddp(self, data: N) -> list[N]:
        if self.use_ddp:
            outputs: list[N] = [data]*self.config.world_size
            dist.all_gather_object(outputs, data)
            if isinstance(data, (torch.Tensor, Metric)) and data.is_cuda:
                outputs = [t.to(data.device) for t in outputs]  # type: ignore
            return outputs
        return [data]

    def gather_metrics(self, metric: M) -> M:
        if self.use_ddp:
            return sum_metrics(self.gather_ddp(metric))
        else:
            return metric

    def log_metric(self, metric: Metric,
                   epoch: int,
                   split: Literal["train", "eval", "test"]) -> None:
        if not self.use_ddp or self.config.rank == 0:
            self.writer.add_metric(metric, epoch, split)

    def get_loader(self, data: DataLoader | DepDataset) -> DataLoader:
        if not isinstance(data, DataLoader):
            assert self.config.batch_size <= len(data), (
                "Batch size larger than dataset. "
                f"dataset size: {len(data)}, batch size: "
                f"{self.config.batch_size}")
            return get_loader(
                data, batch_size=self.config.batch_size,
                bucket=False,
                shuffle=False, droplast=False,
                world_size=self.config.world_size,
                rank=self.config.rank,
                n_workers=self.config.n_workers)
        return data


# def logits_to_probs(logits: torch.Tensor,
#                    labels: torch.Tensor,
#                    ignore_id: int) -> list[torch.Tensor]:
#    probs = torch.softmax(logits, dim=-1)
#    pred_prob_list = []
#    for probs_sen, label_ids_sen in zip(
#            probs, labels):
#        select_entries = label_ids_sen != ignore_id
#        unpadded_labels = label_ids_sen[select_entries][1:-1]
#        probs_sen = probs_sen[1:-1]
#        unpadded_indices = torch.arange(
#            label_ids_sen.shape[0])[select_entries][1:-1] - 1
#        pred_prob = probs_sen[unpadded_indices,
#                              unpadded_labels.long()]
#        pred_prob_list.append(pred_prob)
#    return pred_prob_list


def select_true(preds: torch.Tensor,
                labels: torch.Tensor,
                ignore_index: int | None = None) -> torch.Tensor:
    """If ignore_index is given, it selects the probability for element zero"""
    labels = labels.unsqueeze(-1)
    if ignore_index is not None:
        labels = labels.clone()
        labels[labels == ignore_index] = 0
    return torch.gather(preds, -1, labels).squeeze(-1)


def logits_to_probs(logits: torch.Tensor, softmax: bool = True
                    ) -> torch.Tensor:
    if softmax:
        return torch.softmax(logits, dim=-1)
    else:
        return torch.sigmoid(logits)


def logits_to_true_probs(logits: torch.Tensor,
                         labels: torch.Tensor,
                         ignore_index: int | None = None,
                         softmax: bool = True) -> torch.Tensor:
    probs = logits_to_probs(logits, softmax)
    return select_true(probs, labels, ignore_index)


def logits_to_surprisal(logits: torch.Tensor,
                        labels: torch.Tensor,
                        ignore_index: int | None = None,
                        softmax: bool = True) -> torch.Tensor:
    return -torch.log(logits_to_true_probs(
        logits, labels, ignore_index, softmax))


def sum_depadded(values: torch.Tensor,
                 labels: torch.Tensor,
                 ignore_index: int) -> torch.Tensor:
    values[labels == ignore_index] = 0
    return values.sum(-1)


def mean_depadded(values: torch.Tensor,
                  labels: torch.Tensor,
                  ignore_index: int) -> torch.Tensor:
    num_items = (labels != ignore_index).sum(-1)
    sums = sum_depadded(values, labels, ignore_index)
    return sums / num_items


def logits_to_perplexity(logits: torch.Tensor,
                         labels: torch.Tensor,
                         ignore_index: int,
                         softmax: bool = True) -> torch.Tensor:
    # Should we disregard first node (root from dummy?)
    surprisal = logits_to_surprisal(logits, labels, softmax)
    means = mean_depadded(surprisal, labels, ignore_index)
    return torch.exp(means)


def unpad(sentences: torch.Tensor, labels: torch.Tensor,
          ignore_index: int) -> list[torch.Tensor]:
    unpadded_list: list[torch.Tensor] = []
    for sentence, sen_labels in zip(sentences, labels):
        unpadded_list.append(sentence[sen_labels != ignore_index])
    return unpadded_list


def unpad_masks(masks: torch.Tensor, labels: torch.Tensor,
                ignore_index: int) -> list[torch.Tensor]:
    unpadded_list: list[torch.Tensor] = []
    for sentence, sen_labels in zip(masks, labels):
        unpadded_list.append(
            sentence[:, sen_labels != ignore_index][
                :, :, sen_labels != ignore_index])
    return unpadded_list


def get_uas_abs(inp: tuple[np.ndarray, np.ndarray, int]) -> int:
    pred_arcs, gold_arcs, upto_not_padding = inp
    pred_arcs = pred_arcs[upto_not_padding][:, upto_not_padding]
    gold_arcs = gold_arcs[upto_not_padding][:, upto_not_padding]
    pred_headlist = mst(pred_arcs)
    # one could max for each row as head instead
    # but this would not correspond to the max probability
    # tree given the scores
    gold_headlist = mask_to_headlist(gold_arcs)
    # print(pred_headlist, gold_headlist)
    uas_s = uas_absolute(pred_headlist, gold_headlist) + 1
    # add 1 for dummy mask since metric divides through
    # the number of all tokens)
    return uas_s
