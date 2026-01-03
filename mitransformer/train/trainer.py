from .. import models, data, utils
from . import hooks, losses, attdistr, functions, metrics

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler
from torch.optim.adam import Adam
from torch.optim import Optimizer
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
import os

from dataclasses import dataclass, field
from collections import defaultdict
from types import MappingProxyType
import itertools

from ..utils import pickle

from typing import (Self, Literal, cast,
                    Container, Iterable, Mapping,
                    Any, Generator, TypedDict, NotRequired,
                    TypeVar, DefaultDict,)

from ..utils.logmaker import getLogger, info, get_timestr, warning

logger = getLogger(__name__)


M = TypeVar("M", bound=metrics.LMMetric)
N = TypeVar("N")
K = TypeVar("K")
V = TypeVar("V")


class Undefined():
    """A special class that represents an undefined
    value. This is useful for argument parsing to
    distinguish between None and an undefined parameter
    for instance, when loading a standard configuration
    file and giving the user the option to overwrite
    parts of that configuration."""
    pass


class AdditionalPrediction(TypedDict):
    proj_states: NotRequired[list[torch.Tensor]]
    att: list[torch.Tensor]
    embeddings: list[torch.Tensor]
    activations: list[torch.Tensor]
    logits: list[torch.Tensor]
    label_ids: list[torch.Tensor]


class Result(TypedDict):
    train: metrics.LMMetric
    eval: metrics.LMMetric


class TestResult(Result):
    test: NotRequired[metrics.LMMetric]


Mode = Literal["standard", "input", "supervised"]


@dataclass
class GeneralConfig(utils.Params):
    batch_size: int = 16
    dependency_mode: Mode = "supervised"
    loss_alpha: float | None = 0.5
    arc_loss_weighted: bool = False
    device: str | int = "cpu"
    use_amp: bool = True
    rank: int | None = None
    world_size: int = 1
    n_workers: int = 0
    model_name: str = field(default_factory=get_timestr)
    early_stop_metric: str = "loss"
    use_ddp: bool = False
    discriminative: bool = False
    masks_setting: data.MasksSetting = "current"
    combined_loss: bool = False
    distr_mode: Literal["att", "att-n"] = "att"
    global_distr: bool = True
    length_weighted: bool = False
    include_current: bool = True
    losses: None | Mapping[str, int | float] = MappingProxyType(
        {"lm": 1})
    k_negatives: None | int = None


@dataclass
class TrainConfig(GeneralConfig):
    eval_interval: int = 1
    epochs: int = 100
    learning_rate: float = 1e-3
    early_stop_after: int | None = 1
    use_steps: bool = False
    max_steps: int | None = None
    gradient_acc: int | None = None
    seed: int = 0


class LMTrainer():
    model_dir: str = "./models/"

    def __init__(
            self, transformerlm: models.MITransformerLM,
            transformer_config: models.MITransformerConfig,
            config: GeneralConfig):
        self.use_amp = config.use_amp
        self.device_type = "cpu" if config.device == "cpu" else "cuda"
        self.scaler = GradScaler(
            self.device_type,
            enabled=self.use_amp)

        self.writer = metrics.MetricWriter(
            log_dir=os.path.join("./runs", config.model_name))

        transformerlm.compile()
        # possibly amp autocast should be wrapped around LMTrainer usage
        self.transformerlm: models.MITransformerLM | DDP = transformerlm
        self.transformerlm.to(config.device)
        self.transformer_config: models.MITransformerConfig
        self.transformer_config = transformer_config

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

        self.hooks: list[hooks.Hook] = []

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
    def load_model(
                cls, model_name: str, device: str = "cpu",
                legacy_support: bool = True
                ) -> tuple[
                    models.MITransformerLM, models.MITransformerConfig]:
        if (legacy_support
                and "config" in (loaded_dict := torch.load(
                os.path.join(cls.model_dir, model_name, "model"),
                map_location=device,
                weights_only=False,
                pickle_module=pickle)).keys()):
            state_dict, transformer_config = loaded_dict.values()

        else:
            transformer_config = models.MITransformerConfig.load(os.path.join(
                cls.model_dir, model_name, "transformer_config.json"
            ))
            state_dict = torch.load(
                    os.path.join(cls.model_dir, model_name, "model"),
                    weights_only=True,
                    map_location=device)
        transformer_config = cast(
            models.MITransformerConfig, transformer_config)
        model: models.MITransformerLM = models.MITransformerLM(
            models.MITransformer(transformer_config))
        model.load_state_dict(state_dict)
        return model, transformer_config

    @classmethod
    def load(
            cls, model_name: str,
            device: str = "cpu",
            legacy_support: bool = True,
            check_legacy_filename: bool = True,
            **optional_config: Any) -> Self:
        optional_config["model_name"] = model_name

        model, transformer_config = cls.load_model(
            model_name,
            device,
            legacy_support=legacy_support)

        train_config: TrainConfig = TrainConfig.load(
            os.path.join(
                cls.model_dir, model_name, "config.json"),
            legacy_support=legacy_support,
            check_legacy_filename=check_legacy_filename)
        config = GeneralConfig.from_kwargs(
            **train_config.asdict())
        config.update_from_kwargs(device=device, **optional_config)

        cls.model_info(model, transformer_config, config)

        return cls(model, transformer_config, config)

    @classmethod
    def new(cls, transformer_config: models.MITransformerConfig,
            config: GeneralConfig) -> Self:

        model: models.MITransformerLM = models.MITransformerLM(
            models.MITransformer(transformer_config))

        cls.model_info(model, transformer_config, config)
        return cls(model, transformer_config, config)

    @classmethod
    def model_info(
            cls, model: models.MITransformerLM,
            transformer_config: models.MITransformerConfig,
            config: GeneralConfig) -> None:
        info(
            config.rank, logger,
            "Initialised model with params:\n")
        info(
            config.rank, logger,
            transformer_config.info)

        model_parameters = filter(
            lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        info(config.rank, logger, f"Number of parameters: {params}")

        info(
            config.rank, logger,
            "Initialised trainer with params:\n")
        info(config.rank, logger, config.info)

    def save(self, legacy: bool = False) -> None:
        assert self.train_config is not None
        if self.use_ddp:
            dist.barrier()
        if not self.use_ddp or self.config.rank == 0:
            model = self.transformerlm
            if self.use_ddp:
                assert isinstance(
                    self.transformerlm.module, models.MITransformerLM)
                model = self.transformerlm.module
            dir = os.path.join(self.model_dir, self.train_config.model_name)
            Path(dir).mkdir(parents=True, exist_ok=True)
            if legacy:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": self.transformer_config},
                    os.path.join(dir, "model"))
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(dir, "model"))
                self.transformer_config.save(
                    os.path.join(dir, "transformer_config.json"))

            # overwrites config
            self.train_config.save(os.path.join(dir, "config.json"))

    def load_state(
            self, model_name: str | None = None,
            legacy_support: bool = True) -> None:
        if model_name is None:
            model_name = self.config.model_name
        if (legacy_support
                and "config" in (loaded_dict := torch.load(
                os.path.join(self.model_dir, model_name, "model"),
                map_location=str(self.config.device),
                weights_only=False,
                pickle_module=pickle)).keys()):
            state_dict, _ = loaded_dict.values()

        else:
            state_dict = torch.load(
                    os.path.join(self.model_dir, model_name, "model"),
                    weights_only=True)
        if self.use_ddp:
            assert isinstance(
                self.transformerlm.module, models.MITransformerLM)
            self.transformerlm.module.load_state_dict(state_dict)
        else:
            self.transformerlm.load_state_dict(state_dict)
        self.transformerlm.to(self.config.device)

    def add_hook(
            self,
            hook: hooks.Hook
            ) -> None:
        self.hooks.append(hook)

    def run_hooks(
            self, input: data.IdBatch | data.MaskIdBatch,
            output: tuple[torch.Tensor, dict[str, torch.Tensor] | None]
            ) -> None:
        for hook in self.hooks:
            assert output[1] is not None
            hook(input, output)  # type: ignore

    def init_hooks(
            self, dataloader: data.DataLoader, dataset_name: str,
            epoch: int | None = None,
            token_mapper: data.TokenMapper | None = None) -> None:
        for hook in self.hooks:
            if epoch is None:
                hook.init(dataloader, token_mapper, dataset_name)
            else:
                hook.init(
                    dataloader, token_mapper,
                    "_".join((dataset_name, f"ep:{epoch}")))

    def loss(
            self, logits: torch.Tensor, labels: torch.Tensor,
            ignore_index: int = -100,
            reduction: Literal["sum", "mean"] = "sum"
            ) -> torch.Tensor:
        return losses.lm_loss(
            logits, labels, ignore_index,
            reduction, self.config.discriminative,
            k_negatives=self.config.k_negatives)

    def arc_loss(
            self, score_preds: torch.Tensor,
            score_gold: torch.BoolTensor,
            to_ignore_mask: torch.BoolTensor | None,
            reduction: Literal["sum", "mean"] = "sum"
            ) -> tuple[torch.Tensor, int]:
        """reduction sum takes a mean across dim 1
        of the mask"""

        return losses.arc_loss(
            score_preds, score_gold, to_ignore_mask,
            reduction, self.config.arc_loss_weighted)

    def attention_entropy_loss(
            self, arc_distributions: torch.Tensor,
            to_ignore_mask: torch.BoolTensor | Literal["triangular"] | None,
            label_ids: torch.Tensor | None = None,
            ignore_index: int = -100,
            reduction: Literal["sum", "none"] = "sum",
            ) -> torch.Tensor:
        return losses.attention_entropy_loss(
            arc_distributions, to_ignore_mask, reduction=reduction,
            global_distr=self.config.global_distr,
            include_current=self.config.include_current,
            length_weighted=self.config.length_weighted,
            prefix_dummies=2,
            label_ids=label_ids, ignore_index=ignore_index)

    def attention_distance_loss(
            self, arc_distributions: torch.Tensor,
            to_ignore_mask: torch.BoolTensor | Literal["triangular"] | None,
            label_ids: torch.Tensor | None = None,
            ignore_index: int = -100,
            reduction: Literal["sum", "none"] = "sum",
            ) -> torch.Tensor:
        return losses.attention_distance_loss(
            arc_distributions, to_ignore_mask, reduction=reduction,
            global_distr=self.config.global_distr,
            length_weighted=self.config.length_weighted,
            prefix_dummies=2,
            label_ids=label_ids, ignore_index=ignore_index)

    def attention_difference_loss(
            self, arc_distributions: torch.Tensor,
            to_ignore_mask: torch.BoolTensor | Literal["triangular"] | None,
            label_ids: torch.Tensor | None = None,
            ignore_index: int = -100,
            reduction: Literal["sum", "none"] = "sum",
            ) -> torch.Tensor:
        return losses.attention_difference_loss(
            arc_distributions, to_ignore_mask, reduction=reduction,
            global_distr=self.config.global_distr,
            length_weighted=self.config.length_weighted,
            include_current=self.config.include_current,
            prefix_dummies=2,
            label_ids=label_ids, ignore_index=ignore_index)

    def attention_activation_loss(
            self, arc_distributions: torch.Tensor,
            to_ignore_mask: torch.BoolTensor | Literal["triangular"] | None,
            label_ids: torch.Tensor | None = None,
            ignore_index: int = -100,
            reduction: Literal["sum", "none"] = "sum",
            ) -> torch.Tensor:
        return losses.attention_activation_loss(
            arc_distributions, to_ignore_mask, reduction=reduction,
            global_distr=self.config.global_distr,
            length_weighted=self.config.length_weighted,
            prefix_dummies=2,
            include_current=self.config.include_current,
            label_ids=label_ids, ignore_index=ignore_index)

    def attention_losses(
            self, additional: models.AdditionalResults,
            to_ignore_mask: torch.BoolTensor | Literal["triangular"] | None,
            label_ids: torch.Tensor | None = None,
            ignore_index: int = -100,
            reduction: Literal["sum", "none"] = "sum"
            ) -> dict[str, torch.Tensor]:
        """proj_states cannot be none if mode is `att-n`.
        arc_logits have form (M, B, S, S)
        TODO: restructure so that we have two modes of returning arcs;
        one mode (item 2 returned) for alpha computation and second mode
        (item 3 returned with dict of proj_states and att
        for combined loss mode)"""
        out_dict: dict[str, torch.Tensor] = {}
        assert self.config.losses is not None
        if len(self.config.losses) == 1:
            return out_dict

        additional = {
            key: attdistr.merge_layer_heads(tensor)  # type: ignore
            for key, tensor in additional.items()}
        arc_distribution = attdistr.arc_distribution(
            additional, mode=self.config.distr_mode,
            without_diagonal=not self.config.include_current,
            without_dummy_prefixes=2)
        del additional

        for loss in self.config.losses:
            match loss:
                case "attention_entropy":
                    out_dict[f"{loss}_loss"] = (
                        self.attention_entropy_loss(
                            arc_distribution, to_ignore_mask,
                            label_ids,
                            ignore_index, reduction))
                case "attention_distance":
                    out_dict[f"{loss}_loss"] = (
                        self.attention_distance_loss(
                            arc_distribution, to_ignore_mask,
                            label_ids,
                            ignore_index, reduction))
                case "attention_difference":
                    out_dict[f"{loss}_loss"] = (
                        self.attention_difference_loss(
                            arc_distribution, to_ignore_mask,
                            label_ids,
                            ignore_index, reduction))
                case "attention_activation":
                    out_dict[f"{loss}_loss"] = (
                        self.attention_activation_loss(
                            arc_distribution, to_ignore_mask,
                            label_ids,
                            ignore_index, reduction))
                case "lm":
                    pass
                case _:
                    pass

        return out_dict

    def cosine_loss(
            self,
            additional: models.AdditionalResults,
            label_ids: torch.Tensor | None = None,
            ignore_index: int = -100,
            reduction: Literal["sum", "none"] = "sum",
            ) -> torch.Tensor:
        assert "embeddings" in additional
        assert "activations" in additional
        return losses.cosine_loss(
            additional["embeddings"], additional["activations"],
            label_ids, ignore_index, reduction=reduction,
            prefix_dummies=2)

    def surprox_loss(
            self,
            logits: torch.Tensor,
            label_ids: torch.Tensor,
            ignore_index: int = -100,
            reduction: Literal["sum", "none"] = "sum",
            ) -> torch.Tensor:
        return losses.surprox_loss(
            logits, label_ids, ignore_index, reduction=reduction,
            prefix_dummies=2)

    def additional_losses(
            self, additional: models.AdditionalResults,
            to_ignore_mask: torch.BoolTensor | Literal[
                "triangular"] | None = "triangular",
            label_ids: torch.Tensor | None = None,
            ignore_index: int = -100,
            logits: torch.Tensor | None = None,
            reduction: Literal["sum", "none"] = "sum"
            ) -> dict[str, torch.Tensor]:
        additional_losses = self.attention_losses(
            additional, to_ignore_mask=to_ignore_mask,
            reduction=reduction,
            label_ids=label_ids, ignore_index=ignore_index)
        assert self.config.losses is not None
        if "cosine" in self.config.losses.keys():
            additional_losses["cosine"] = self.cosine_loss(
                additional, label_ids, ignore_index=ignore_index,
                reduction=reduction
            )
        if "surprox" in self.config.losses.keys():
            assert logits is not None
            assert label_ids is not None
            additional_losses["surprox"] = self.surprox_loss(
                logits, label_ids,
                ignore_index=ignore_index, reduction=reduction)
        return additional_losses

    @staticmethod
    def filter_arc_scores(
            arc_scores: Mapping[str, torch.Tensor],
            keep_keys: Container[str] | Iterable[str]
            ) -> dict[str, torch.Tensor]:
        return {key: arc_scores[key]
                for key in arc_scores.keys() if key in keep_keys}

    @staticmethod
    def stack_pred_scores(
            arc_scores: Mapping[str, torch.Tensor]
            ) -> dict[str, torch.Tensor] | None:
        if len(arc_scores) == 0:
            return None
        return dict(arc_scores)

    @staticmethod
    def expand_gold_scores(
            masks: Mapping[str, torch.BoolTensor],
            arc_scores: Mapping[str, torch.Tensor]
            ) -> dict[str, torch.BoolTensor] | None:
        expanded = {key: gold.unsqueeze(0).expand(len(arc_scores), -1, -1, -1)
                    for (key, gold), arc_scores in zip(
                        masks.items(),
                        arc_scores.values())}
        if len(expanded) == 0:
            return None
        return cast(dict[str, torch.BoolTensor], expanded)

    @classmethod
    def align_scores(
            cls,
            arc_scores: Mapping[str, torch.Tensor],
            masks: Mapping[str, torch.BoolTensor]
            ) -> tuple[
                dict[str, torch.Tensor], dict[str, torch.BoolTensor]] | None:
        score_preds = cls.stack_pred_scores(arc_scores)
        score_gold = cls.expand_gold_scores(masks, arc_scores)
        if score_preds is None or score_gold is None:
            return None
        else:
            return score_preds, score_gold

    @classmethod
    def prepare_scores(
            cls, arc_scores: Mapping[str, torch.Tensor],
            masks: Mapping[str, torch.BoolTensor]
            ) -> tuple[
                dict[str, torch.Tensor], dict[str, torch.BoolTensor]] | None:
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

        nums = (label_ids != ignore_id).sum(1)
        # [B]

        inds = torch.arange(label_ids.shape[1], device=nums.device)
        # [S]

        inds_mat = (inds.unsqueeze(0) + inds.unsqueeze(1)).unsqueeze(0)
        # [1, S, S]

        not_to_ignore = inds_mat < nums.unsqueeze(1).unsqueeze(2)
        # [B, S, S]

        not_to_ignore = not_to_ignore.unsqueeze(0).expand(
            scores.shape[0], -1, -1, -1)
        # [M, B, S, S]

        return ~not_to_ignore  # type: ignore

    @classmethod
    def batch_to[D](cls, batch: D, device) -> D:
        new_batch: D = {}  # type: ignore
        assert isinstance(batch, dict)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                new_batch[key] = value.to(  # type: ignore
                    device, non_blocking=True)
            elif isinstance(value, dict):
                new_batch[key] = cls.batch_to(value, device)  # type: ignore
            else:
                new_batch[key] = value  # type: ignore
        return new_batch  # type: ignore

    def get_metric(
            self,
            num_instances: int,
            lm_loss: torch.Tensor,
            num_arc_instances: int | None = None,
            arc_loss: torch.Tensor | None = None,
            perplexity: float | None = None,
            uas: float | pd.DataFrame | None = None,
            att_entropy: pd.DataFrame | None = None,
            additional_losses: Mapping[str, torch.Tensor] | None = None,
            weights: Mapping[str, int | float] | None = None
            ) -> metrics.LMMetric:
        if perplexity is not None:
            if arc_loss is not None:
                assert uas is not None
                assert num_arc_instances is not None
                return metrics.SupervisedEvalMetric(
                    num=num_instances,
                    arc_num=num_arc_instances,
                    lm_loss=lm_loss,
                    perplexity=perplexity,
                    arc_loss=arc_loss,
                    alpha=self.config.loss_alpha,
                    uas=uas,
                    att_entropy=att_entropy,
                    main_metric=self.config.early_stop_metric
                    )
            else:
                if self.config.combined_loss is False:
                    return metrics.EvalMetric(
                        num=num_instances,
                        lm_loss=lm_loss,
                        perplexity=perplexity,
                        main_metric=self.config.early_stop_metric)
                else:
                    assert (
                        additional_losses is not None
                        and weights is not None
                    )
                    losses = ["lm_loss"] + list(additional_losses.keys())
                    return metrics.DynamicWeightedEvalMetric(
                            losses)(  # type: ignore
                        num=num_instances,
                        lm_loss=lm_loss,
                        main_metric=self.config.early_stop_metric,
                        **additional_losses,
                        weights=list(weights.values()),
                        perplexity=perplexity
                    )

        if arc_loss is None:
            if self.config.combined_loss is False:
                return metrics.LMMetric(
                    num=num_instances,
                    lm_loss=lm_loss,
                    main_metric=self.config.early_stop_metric)
            else:
                assert (
                    additional_losses is not None
                    and weights is not None
                )
                losses = ["lm_loss"] + list(additional_losses.keys())
                return metrics.DynamicWeightedMetric(losses)(  # type: ignore
                    num=num_instances,
                    lm_loss=lm_loss,
                    main_metric=self.config.early_stop_metric,
                    **additional_losses,
                    weights=list(weights.values()),
                )
        else:
            assert num_arc_instances is not None
            return metrics.SupervisedMetric(
                num=num_instances,
                arc_num=num_arc_instances,
                lm_loss=lm_loss,
                arc_loss=arc_loss,
                alpha=self.config.loss_alpha,
                main_metric=self.config.early_stop_metric)

    def train_step(
            self,
            batch: data.IdBatch | data.MaskIdBatch,
            ignore_index: int,
            perform_opt: bool = True) -> metrics.LMMetric:
        assert self.train_config is not None, "Config missing training params."
        assert self.optimiser is not None
        batch = self.batch_to(batch, device=self.config.device)

        additional: models.AdditionalResults
        arc_logits: dict[str, torch.Tensor] | None
        with torch.autocast(
                self.device_type, dtype=torch.float16, enabled=self.use_amp):

            logits, arc_logits, additional = self.transformerlm(
                **batch,
                return_arc_logits=not self.config.combined_loss,
                return_proj_states=(
                    self.config.combined_loss
                    and self.config.distr_mode == "att-n"),
                return_att=(
                    self.config.combined_loss
                    and self.config.distr_mode == "att"),
                return_embeddings=(
                    self.config.combined_loss
                    and self.config.losses is not None
                    and "cosine" in self.config.losses
                ),
                return_activations=(
                    self.config.combined_loss
                    and self.config.losses is not None
                    and "cosine" in self.config.losses
                ))

            self.run_hooks(batch, (logits, arc_logits))
            # remove from arc_scores those that should not be used...

            lm_loss = self.loss(
                logits, batch["label_ids"],
                ignore_index=ignore_index,
                reduction="sum")
            arc_loss: torch.Tensor | None = None
            additional_losses: dict[str, torch.Tensor] | None = None

            num_arc_instances: int | None = None
            if self.train_config.dependency_mode == "supervised":
                assert "masks" in batch
                assert arc_logits is not None
                batch = cast(data.MaskIdBatch, batch)
                score_pair = self.prepare_scores(
                    arc_logits, batch["masks"])

                if score_pair is not None:
                    score_preds, score_golds = score_pair
                    score_preds = {
                        key: F.sigmoid(
                            pred) for key, pred in score_preds.items()}

                    preds_concat = torch.concat(list(score_preds.values()))
                    golds_concat = torch.concat(list(score_golds.values()))
                    del score_pair
                    del score_preds
                    del score_golds

                    to_ignore = self.get_ignore_mask(
                            preds_concat,
                            batch["label_ids"],
                            ignore_index)
                    # TODO: is this correctly masked?
                    # For current should we shift to_ignore?

                    arc_loss, num_arc_instances = self.arc_loss(
                        preds_concat,
                        cast(torch.BoolTensor, golds_concat),
                        to_ignore,
                        reduction="sum")
                    del preds_concat
                    del golds_concat
                else:
                    warning(
                        self.config.rank, logger,
                        "Scores did not align. Check keys.")

            elif self.config.combined_loss:
                additional_losses = self.additional_losses(
                    additional, to_ignore_mask="triangular",
                    logits=logits,
                    label_ids=batch["label_ids"], ignore_index=ignore_index,
                    reduction="sum")
            del additional

            num_instances = int(
                (batch["label_ids"] != ignore_index).sum().item())
            del batch

            metric = self.get_metric(
                num_instances,
                num_arc_instances=num_arc_instances,
                lm_loss=lm_loss,
                arc_loss=arc_loss,
                additional_losses=additional_losses,
                weights=self.config.losses)

            loss: torch.Tensor = metric.loss
            metric.detach_()
            metric.to_("cpu")

        if self.train_config.gradient_acc is not None:
            self.scaler.scale(
                (loss / self.train_config.gradient_acc)).backward()
            # divide the per-instance avg loss by the number of gradient_acc.
            # Otherwise the gradient would be gradient_acc-times as high
            # as in the non-accumulating setting.
            # Note that the loss is different in the two settings due to
            # the per-instance averaging: the num_instances that the
            # loss is divided by can be different for the items of an
            # accumulating batch. I.e. we are taking an average of averages
            # which can be different from a gloal average.
        else:
            self.scaler.scale(loss).backward()
        del loss

        if perform_opt:
            self.scaler.step(self.optimiser)   # update parameters
            self.scaler.update()
            self.optimiser.zero_grad(set_to_none=True)

        return metric

    def eval_step(
            self,
            batch: data.IdBatch | data.MaskIdBatch,
            mode: Mode,
            ignore_index: int) -> metrics.LMMetric:
        batch = self.batch_to(batch, device=self.config.device)

        additional: models.AdditionalResults
        arc_logits: None | dict[str, torch.Tensor]

        with torch.autocast(
                self.device_type, dtype=torch.float16, enabled=self.use_amp):
            logits, arc_logits, additional = self.transformerlm(
                **batch,
                return_arc_logits=not self.config.combined_loss,
                return_proj_states=(
                    self.config.combined_loss
                    and self.config.distr_mode == "att-n"),
                return_att=(
                    self.config.combined_loss
                    and self.config.distr_mode == "att"),
                return_embeddings=(
                    self.config.combined_loss
                    and self.config.losses is not None
                    and "cosine" in self.config.losses
                ),
                return_activations=(
                    self.config.combined_loss
                    and self.config.losses is not None
                    and "cosine" in self.config.losses
                ))
            self.run_hooks(batch, (logits, arc_logits))
            # remove from arc_scores those that should not be used...

            labels = batch["label_ids"]
            lm_loss = self.loss(
                logits, labels,
                ignore_index=ignore_index, reduction="sum")

            num_instances = int((labels != ignore_index).sum().item())

            surprisal_sum = functions.sum_depadded(
                functions.logits_to_surprisal(
                    logits, labels,
                    ignore_index,
                    softmax=not self.config.discriminative),
                labels, ignore_index).sum().detach().cpu().item()
            del labels

            uas_abs: None | pd.DataFrame | float = None
            arc_loss: None | torch.Tensor = None
            additional_losses: None | dict[str, torch.Tensor] = None

            att_entropy = None
            num_arc_instances = None
            if mode == "supervised":
                assert arc_logits is not None
                assert "masks" in batch
                batch = cast(data.MaskIdBatch, batch)
                score_pair = self.prepare_scores(
                    arc_logits, batch["masks"])

                if score_pair is not None:
                    score_preds, score_golds = score_pair
                    score_logits = score_preds
                    score_preds = {
                        key: F.sigmoid(pred) for key, pred
                        in score_preds.items()}

                    preds_concat = torch.concat(list(score_preds.values()))
                    golds_concat = torch.concat(list(score_golds.values()))

                    to_ignore_dict = {
                        key: self.get_ignore_mask(
                            preds,
                            batch["label_ids"],
                            ignore_index)
                        for key, preds in score_preds.items()}
                    to_ignore = cast(
                        torch.BoolTensor, torch.concat(
                            list(to_ignore_dict.values())))

                    arc_loss, num_arc_instances = self.arc_loss(
                        preds_concat,
                        cast(torch.BoolTensor, golds_concat),
                        to_ignore,
                        reduction="sum")

                    # TODO allow to manage current and next dep
                    # => two UAS metrics

                    uas = []
                    if self.config.masks_setting in ("current", "both"):
                        uas.append(functions.uas_composition(
                            score_preds, score_golds, batch["label_ids"],
                            ignore_index, "current",
                            "head_current", "child_current"))
                    if self.config.masks_setting in ("next", "both"):
                        uas.append(functions.uas_composition(
                            score_preds, score_golds, batch["label_ids"],
                            ignore_index, "next", "head_next", "child_next"))
                    if len(uas) == 1:
                        uas_abs = uas[0]
                    elif len(uas) > 1:
                        uas_abs = pd.DataFrame({
                            "current": [uas[0]],
                            "next": [uas[1]]})
                    else:
                        raise Exception(
                            "masks_setting should not be "
                            + self.config.masks_setting)

                    att_entropy = pd.DataFrame({
                        key: losses.get_attention_entropy(
                            logits_preds.softmax(-1), to_ignore_dict[key],
                            "none").flatten(1).sum(-1).detach().cpu()
                        for key, logits_preds in score_logits.items()})
                    # can make separate list of heads
            elif self.config.combined_loss:
                additional_losses = self.additional_losses(
                    additional, to_ignore_mask="triangular",
                    logits=logits,
                    label_ids=batch["label_ids"], ignore_index=ignore_index,
                    reduction="sum")
            del logits

        metric = self.get_metric(
            num_instances,
            lm_loss=lm_loss,
            arc_loss=arc_loss,
            num_arc_instances=num_arc_instances,
            perplexity=surprisal_sum,
            uas=uas_abs,
            att_entropy=att_entropy,
            additional_losses=additional_losses,
            weights=self.config.losses)
        metric.detach_()
        metric.to_("cpu")
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
            info(
                self.config.rank, logger,
                f"Aborting training after {evals_without_improvement} "
                "evals without improvement.")
            return True
        return False

    @torch.compile()
    def train_iter(
            self,
            train: (
                data.TokenisedDataset[data.IdsSentence]
                | data.DataLoader[data.IdsSentence, data.IdBatch]),
            eval: (
                data.TokenisedDataset[data.IdsSentence]
                | data.DataLoader[data.IdsSentence, data.IdBatch]),
            token_mapper: data.TokenMapper | None = None,
            **kwargs) -> Generator[
                Result,
                None,
                tuple[tuple[int, int], tuple[int, int]]]:
        assert self.train_config is not None, "Config missing training params."
        train_config = self.train_config
        device = train_config.device
        assert device is not None

        train = self.get_loader(train, "train")
        eval = self.get_loader(eval, "eval")

        eval_interval = train_config.eval_interval

        self.transformerlm.train()

        best: float | metrics.LMMetric | None = None
        evals_without_improvement: int = 0
        total_steps: int = 0
        break_training: bool = False
        max_epochs = (
            train_config.epochs
            if train_config.max_steps is None
            else train_config.max_steps)
        # since we cannot run out of epochs if we use
        # max_steps
        best_epoch: int = 0
        best_step: int = 0
        if self.train_config.max_steps is not None:
            pbar_steps = tqdm(total=self.train_config.max_steps, desc="Steps")
        else:
            pbar_steps = None

        epoch = 0
        for epoch in tqdm(range(1, max_epochs+1), desc="Epochs"):
            self.init_hooks(train, "train", epoch, token_mapper)
            if break_training:
                epoch -= 1
                break
            info(
                self.config.rank,
                logger, f"Epoch: {epoch}/{max_epochs}")
            if self.use_ddp:
                if train.sampler is not None and hasattr(
                        train.sampler, "set_epoch"):
                    train.sampler.set_epoch(epoch)  # type: ignore
                if train.batch_sampler is not None and hasattr(
                        train.batch_sampler, "set_epoch"):
                    train.batch_sampler.set_epoch(epoch)  # type: ignore

            # Steps
            for train_metric in self._train(train):
                total_steps += 1  # equal epochs in case of not use_steps

                if pbar_steps is not None:
                    pbar_steps.update(1)

                if total_steps % eval_interval == 0:
                    info(
                        self.config.rank,
                        logger,
                        (
                            f"Step: {total_steps}/" +
                            (
                                'inf' if train_config.max_steps
                                is None  # type: ignore
                                else str(train_config.max_steps))))
                    self.log_metric(train_metric, total_steps, "train")
                    info(
                        self.config.rank, logger,
                        f"train metric:\n{train_metric.info}")

                    self.init_hooks(eval, "eval", epoch, token_mapper)
                    eval_metric = self._eval(eval)

                    self.log_metric(eval_metric, total_steps, "eval")
                    info(
                        self.config.rank, logger,
                        f"eval metric:\n{eval_metric.info}")

                    # TODO make it possible to save without checking if
                    # there was an improvement
                    if best is None:
                        best = eval_metric.minval()
                    if eval_metric > best:       # greater means better
                        best = eval_metric
                        self.save()

                        best_epoch = epoch
                        best_step = total_steps

                        info(
                            self.config.rank, logger,
                            "Saving model at epoch "
                            f"{epoch} ({total_steps})...")
                        evals_without_improvement = 0
                    else:
                        evals_without_improvement += 1

                    yield {
                        "train": train_metric,
                        "eval": eval_metric}

                    if self.check_early_stop(evals_without_improvement):
                        break_training = True
                        break
                    if (self.train_config.max_steps is not None
                            and total_steps
                            >= self.train_config.max_steps):
                        break_training = True
                        break

                    # Set this here, so that trainer is in eval
                    # mode in outside loop (i.e. when using yield)
                    self.init_hooks(train, "train", epoch, token_mapper)
                    self.transformerlm.train()

        if pbar_steps is not None:
            pbar_steps.close()
        return (epoch, total_steps), (best_epoch, best_step)

    @torch.compile()
    def train(
            self,
            train: (
                data.TokenisedDataset[data.IdsSentence]
                | data.DataLoader[data.IdsSentence, data.IdBatch]),
            eval: (
                data.TokenisedDataset[data.IdsSentence]
                | data.DataLoader[data.IdsSentence, data.IdBatch]),
            test: (
                data.TokenisedDataset[data.IdsSentence]
                | data.DataLoader[data.IdsSentence, data.IdBatch]
                | None) = None,
            **kwargs) -> TestResult:
        assert self.train_config is not None, "Config missing training params."

        train = self.get_loader(train, "train")
        eval = self.get_loader(eval, "eval")

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
        del gen

        # load best (saved) into transformerlm
        self.load_state()

        info(
            self.config.rank, logger,
            f"Ended training after {current[0]} epochs, {current[1]} steps.")
        info(
            self.config.rank, logger,
            f"Found best model after {best[0]} epochs, {best[1]} steps.")

        return self.test(
            train=train,  # type: ignore
            eval=eval,
            test=test)

    def _train(self, loader: (
            data.DataLoader[data.SentenceIds, data.IdBatch])
            ) -> Iterable[metrics.LMMetric]:
        assert self.train_config is not None, "Config missing training params."
        self.transformerlm.train()

        def iterate():
            assert self.train_config is not None, (
                "Config missing training params.")
            metrics_list: list[metrics.LMMetric] = list()
            for i, batch in tqdm(enumerate(loader), desc="Batches"):
                metrics_list.append(self.train_step(
                    batch,
                    loader.dataset.keys_for_padding["label_ids"],
                    perform_opt=(po := functions.check_perform_opt(
                        self.train_config.gradient_acc, i))))
                del batch
                if po:
                    yield metrics.sum_metrics(metrics_list)
                    metrics_list = list()

        if self.train_config.use_steps:
            for e in iterate():
                yield self.gather_metrics(e)
        else:
            yield self.gather_metrics(metrics.sum_metrics(list(iterate())))

    def _eval(self, loader: (
            data.DataLoader[data.SentenceIds, data.IdBatch])
            ) -> metrics.LMMetric:
        self.transformerlm.eval()
        with torch.no_grad():
            # eval loop: no backprop on this data, to avoid storing
            # all intermediatte variable
            metrics_list = [
                self.eval_step(
                    batch,
                    self.config.dependency_mode,
                    loader.dataset.keys_for_padding["label_ids"])
                for batch in tqdm(loader, desc="Batches")]
        return self.gather_metrics(metrics.sum_metrics(metrics_list))

    @torch.compile()
    def test(
            self, token_mapper: data.TokenMapper | None = None,
            **datasets: (
                data.TokenisedDataset[data.IdsSentence]
                | data.DataLoader[data.IdsSentence, data.IdBatch] | Any)
            ) -> dict[str, metrics.LMMetric]:
        metrics_dict: dict[str, metrics.LMMetric] = {}
        for n, ds in datasets.items():
            if isinstance(ds, (data.DataLoader, data.TokenisedDataset)):
                ds = self.get_loader(ds, "test")
                self.init_hooks(ds, n, token_mapper=token_mapper)
                metrics_dict[n] = self._eval(ds)
                info(
                    self.config.rank, logger,
                    f"Test metric for {n} split:\n{metrics_dict[n].info}")
        return metrics_dict

    @torch.compile()
    def predict(
            self, dataset: data.TokenisedDataset[data.IdsSentence],
            make_prob: bool = False,
            only_true: bool = False,
            dataset_name: str | None = None,
            token_mapper: data.TokenMapper | None = None,
            return_arc_logits: bool | None = None,
            return_proj_states: bool | None = None,
            return_att: bool | None = None,
            return_embeddings: bool | None = None,
            return_activations: bool | None = None,
            return_logits: bool | None = None,
            return_label_ids: bool = False,
            ) -> tuple[
                list[torch.Tensor], dict[str, list[torch.Tensor]],
                AdditionalPrediction]:
        """Returns logits and arc scores"""

        probs_global: list[torch.Tensor] = []
        attention_logits_global: DefaultDict[
            str, list[torch.Tensor]] = defaultdict(list)
        additional_global: DefaultDict[
            models.AdditionalKeys, list[torch.Tensor]] = defaultdict(list)
        for (
            pred_probs, attention_logits,
            additional) in self.predict_batched(
                dataset,
                make_prob=make_prob,
                only_true=only_true,
                dataset_name=dataset_name,
                token_mapper=token_mapper,
                return_arc_logits=return_arc_logits,
                return_proj_states=return_proj_states,
                return_att=return_att,
                return_embeddings=return_embeddings,
                return_activations=return_activations,
                return_logits=return_logits,
                return_label_ids=return_label_ids,):

            for key, attention in attention_logits.items():
                attention_logits_global[key].extend(attention)

            for key, tensorlist in additional.items():
                additional_global[key].extend(  # type: ignore
                    tensorlist)  # type: ignore
            probs_global.extend(pred_probs)

        return (
            probs_global, dict(attention_logits_global),
            dict(additional_global))  # type: ignore

    @torch.compile()
    def predict_batched(
            self, dataset: data.TokenisedDataset[data.IdsSentence],
            make_prob: bool = False,
            only_true: bool = False,
            dataset_name: str | None = None,
            token_mapper: data.TokenMapper | None = None,
            return_arc_logits: bool | None = None,
            return_proj_states: bool | None = None,
            return_att: bool | None = None,
            return_embeddings: bool | None = None,
            return_activations: bool | None = None,
            return_logits: bool | None = None,
            return_label_ids: bool = False,
            ) -> Iterable[tuple[
                list[torch.Tensor], dict[str, list[torch.Tensor]],
                AdditionalPrediction]]:
        """Returns logits and arc scores"""
        # TODO: Does this work with ddp? Batches are distributed but not
        # joined back together.
        loader = data.get_loader(  # type: ignore
            dataset,
            bucket=False,
            batch_size=self.config.batch_size,
            shuffle=False,
            droplast=False,
            fill_incomplete=False,
            n_workers=self.config.n_workers,
            rank=self.config.rank,
            world_size=self.config.world_size)
        self.init_hooks(
            loader, (dataset_name if dataset_name is not None else "ds"),
            token_mapper=token_mapper)

        ignore_index = dataset.keys_for_padding["label_ids"]

        unpadded_logits: list[torch.Tensor]
        unpadded_arc_logits: dict[str, list[torch.Tensor]]
        unpadded_additional: dict[str, list[torch.Tensor]]

        self.transformerlm.eval()
        with torch.no_grad():
            # eval loop: no backprop on this data, to avoid storing
            # all intermediate variable
            logits: torch.Tensor
            arc_logits: dict[str, torch.Tensor] | None
            additional: models.AdditionalResults
            for batch in tqdm(loader, desc="Batches"):
                unpadded_arc_logits = {}
                unpadded_additional = {}
                batch = self.batch_to(batch, device=self.config.device)

                if return_arc_logits is None:
                    return_arc_logits = not self.config.combined_loss
                if return_proj_states is None:
                    return_proj_states = (
                        self.config.combined_loss
                        and self.config.distr_mode == "att-n")
                if return_att is None:
                    return_att = (
                        self.config.combined_loss
                        and self.config.distr_mode == "att")
                if return_embeddings is None:
                    return_embeddings = (
                        self.config.combined_loss
                        and self.config.losses is not None
                        and "cosine" in self.config.losses
                    )
                if return_activations is None:
                    return_activations = (
                        self.config.combined_loss
                        and self.config.losses is not None
                        and "cosine" in self.config.losses
                    )

                with torch.autocast(
                        self.device_type, dtype=torch.float16,
                        enabled=self.use_amp):
                    logits, arc_logits, additional = self.transformerlm(
                        **batch,
                        return_arc_logits=return_arc_logits,
                        return_proj_states=return_proj_states,
                        return_att=return_att,
                        return_embeddings=return_embeddings,
                        return_activations=return_activations)

                self.run_hooks(batch, (logits, arc_logits))
                labels = batch["label_ids"]

                if return_logits:
                    unpadded_additional["logits"] = (
                            functions.unpad(
                                logits,
                                labels, ignore_index
                                )
                            )

                if make_prob:
                    logits = functions.logits_to_probs(
                        logits,
                        not self.config.discriminative)

                if only_true:
                    logits = functions.select_true(
                        logits, labels, ignore_index)

                unpadded_logits = (
                    functions.unpad(logits, labels, ignore_index))

                if arc_logits is not None:
                    for key in arc_logits.keys():
                        unpadded_arc_logits[key] = (
                            functions.unpad_masks(
                                arc_logits[key].swapaxes(0, 1),
                                labels, ignore_index))

                additional_key: models.AdditionalKeys
                for additional_key in ("proj_states", "att"):
                    if additional_key in additional:  # type: ignore
                        num_after_square = 0
                        if additional_key in ("proj_states",):
                            num_after_square = 1
                        unpadded_additional[additional_key] = (
                            functions.unpad_masks(
                                additional[
                                    additional_key].swapaxes(  # type: ignore
                                        0, 1),
                                labels, ignore_index,
                                num_after_square=num_after_square))

                for additional_key in ("embeddings", "activations"):
                    if additional_key in additional:  # type: ignore
                        unpadded_additional[additional_key] = (
                            functions.unpad(
                                additional[additional_key],  # type: ignore
                                labels, ignore_index
                            )
                        )
                        # probably removes embedding of <EOS> and
                        # prediction of first padding token.
                        # <EOS> dot pred(.) is part of loss.

                if return_label_ids:
                    unpadded_additional["label_ids"] = (
                        functions.unpad(
                            labels, labels, ignore_index
                        )
                    )

                # This should collect the data across all processes.
                # The distributed sampler chunked it in an interleaved
                # fashion and did not shuffle it, so getting back
                # the correct order should be just a matter of interleaving
                # the lists of results. TODO: test
                unpadded_logits = self.gather_list(
                    unpadded_logits, interleave=True)
                unpadded_arc_logits_out = self.gather_dict_of_lists(
                    dict(unpadded_arc_logits), interleave=True)
                unpadded_additional_out = self.gather_dict_of_lists(
                    dict(unpadded_additional), interleave=True)
                yield (
                    unpadded_logits, unpadded_arc_logits_out,
                    cast(AdditionalPrediction, unpadded_additional_out))

    @torch.compile()
    def generate(
            self, token_mapper: data.TokenMapper,
            start: str | None = None, max_len: int = 40) -> str:

        g: list[int]

        if self.use_ddp:
            model = self.transformerlm.module
        else:
            model = self.transformerlm

        assert isinstance(model, models.MITransformerLM)

        if start is None:
            idx = torch.zeros(
                (1, 2),
                dtype=torch.long,
                device=self.config.device)
            idx[0, 0] = token_mapper.token2id[data.DUMMY]
            idx[0, 1] = token_mapper.token2id[data.ROOT]
            g = model.generate(
                idx, max_new_tokens=max_len).tolist()[0]
            # support an initial mask here
        else:
            conllu = data.parse_list_of_words_with_spacy(
                start.split(), min_len=0)

            transform = data.TransformMaskHeadChild(
                keys_for_head={"head"},
                keys_for_child={"child"},
                triangulate=True)

            dataset: data.CoNLLUDataset = data.CoNLLUDataset.from_str(
                conllu_str=conllu, transform_masks=transform, max_len=None)

            dataset.map_to_ids(token_mapper)
            dataloader: data.DataLoader = data.get_loader(    # type: ignore
                dataset, batch_size=1,                  # type: ignore
                bucket=False, min_size=0, max_size=50,
                shuffle=False, droplast=False,
                n_workers=self.config.n_workers)

            for batch in dataloader:
                # take last batch, i.e. last sentence
                # TODO: this is a weird solution
                pass

            # TODO: support mask
            try:
                g = model.generate(
                    batch["input_ids"][0],  # type: ignore
                    max_new_tokens=max_len).tolist()[0]
            except NameError:
                raise NameError("dataloader was empty.")

        eos_id = token_mapper.token2id[data.EOS]
        first_eos = next((i for i, x in enumerate(g) if x == eos_id), len(g))
        g = g[:first_eos + 1]
        return token_mapper.decode([g], to_string=True)[0]

    def __del__(self) -> None:
        self.writer.flush()

    def gather_ddp(self, data: N) -> list[N]:
        if self.use_ddp:
            outputs: list[N] = [data]*self.config.world_size
            dist.all_gather_object(outputs, data)
            if isinstance(
                    data, (torch.Tensor, metrics.Metric)) and data.is_cuda:
                outputs = [t.to(data.device) for t in outputs]  # type: ignore
            return outputs
        return [data]

    def gather_metrics(self, metric: M) -> M:
        if self.use_ddp:
            return metrics.sum_metrics(self.gather_ddp(metric))
        else:
            return metric

    def gather_batched_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.use_ddp:
            return torch.concat(self.gather_ddp(tensor))
        else:
            return tensor

    def gather_list[T](
            self, seq: list[T], interleave: bool = False) -> list[T]:
        # TODO: This is not ordered. Introduce a way to order this.
        if self.use_ddp:
            lists = self.gather_ddp(seq)
            if interleave:
                return [  # type: ignore
                    item for tup in itertools.zip_longest(  # type: ignore
                        *lists, fillvalue=Undefined)
                    for item in tup if item != Undefined]

            return [item for seq in lists for item in seq]
        else:
            return seq

    def gather_dict_of_lists[K, T](
            self, mapping: dict[K, list[T]], interleave: bool = False
            ) -> dict[K, list[T]]:
        if self.use_ddp:
            return {
                key: self.gather_list(seq, interleave=interleave)
                for key, seq in mapping.items()}
        else:
            return mapping

    def log_metric(
            self, metric: metrics.LMMetric,
            epoch: int,
            split: Literal["train", "eval", "test"]) -> None:
        if not self.use_ddp or self.config.rank == 0:
            self.writer.add_metric(metric, epoch, split)

    # this is not typed in detail like data.data.get_loader
    def get_loader(self, in_data: (
                data.TokenisedDataset[data.IdsSentence]
                | data.DataLoader[data.IdsSentence, data.IdBatch]),
            mode: Literal["train", "eval", "test"]
            ) -> data.DataLoader[data.IdsSentence, data.IdBatch]:

        if not isinstance(in_data, data.DataLoader):

            bucket = False
            shuffle = False
            droplast = False
            fill_incomplete = True
            seed = 0
            if mode == "train" or mode == "eval":
                bucket = True
                shuffle = True
                assert self.train_config is not None
                seed = self.train_config.seed
            if mode == "train":
                droplast = True
            if mode == "test":
                fill_incomplete = False

            assert self.config.batch_size <= len(in_data), (
                "Batch size larger than dataset. "
                f"dataset size: {len(in_data)}, batch size: "
                f"{self.config.batch_size}")
            return data.get_loader(
                in_data, batch_size=self.config.batch_size,
                bucket=bucket,
                shuffle=shuffle, droplast=droplast,
                world_size=self.config.world_size,
                rank=self.config.rank,
                n_workers=self.config.n_workers,
                seed=seed,
                fill_incomplete=fill_incomplete)
        return in_data
