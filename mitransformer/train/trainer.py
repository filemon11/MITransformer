from .. import models, data, utils
from . import hooks
from ..data.dataloader import (
    DataLoader, get_loader, IDBatch,
    CoNLLUTokenisedBatch, EssentialBatch, D)
from ..data.dataset import (
    DepDataset, TransformMaskHeadChild, CoNLLUDataset, IDSen)
from ..train.metrics import (
    sum_metrics, SupervisedEvalMetric,
    SupervisedMetric, Metric, EvalMetric,
    MetricWriter, M, N)

from ..data.tokeniser import DUMMY, ROOT, EOS
from ..utils.dependencies import (
    mst, merge_head_child_scores,
    dummy_mask_removal, mask_to_headlist, uas_absolute)

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.optim.adam import Adam
from torch.optim import Optimizer
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path
import os

from dataclasses import dataclass, field
from collections import defaultdict

from ..utils import pickle

from typing import (Self, Literal, cast,
                    Container, Iterable, Mapping,
                    Any, Generator, TypedDict, NotRequired)

from ..utils.logmaker import getLogger, info, get_timestr

logger = getLogger(__name__)


class Result(TypedDict):
    train: Metric
    eval: Metric


class TestResult(Result):
    test: NotRequired[Metric]


Mode = Literal["standard", "input", "supervised"]


@dataclass
class GeneralConfig(utils.Params):
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
    masks_setting: Literal["next", "current", "complete"] = "current"


@dataclass
class TrainConfig(GeneralConfig):
    eval_interval: int = 1
    epochs: int = 100
    learning_rate: float = 1e-3
    early_stop_after: int | None = 1
    use_steps: bool = False
    max_steps: int | None = None
    gradient_acc: int | None = None


class LMTrainer():
    model_dir: str = "./models/"

    def __init__(
            self, transformerlm: models.MITransformerLM,
            transformer_config: models.MITransformerConfig,
            config: GeneralConfig):
        self.writer = MetricWriter(
            log_dir=os.path.join("./runs", config.model_name))
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
                    weights_only=True)
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
            self, input: CoNLLUTokenisedBatch | EssentialBatch,
            output: tuple[torch.Tensor, dict[str, list[torch.Tensor]]]
            ) -> None:
        for hook in self.hooks:
            hook(input, output)

    def init_hooks(
            self, dataloader: DataLoader, dataset_name: str,
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
                logits.shape[-2])
            loss = F.binary_cross_entropy(
                probs.swapaxes(-1, -2), one_hot.float(), reduction='none')

            # # mask randomly
            # mask_rate = 0.5
            # rand_mask = torch.rand(
            #    loss.shape, device=mask.device) < mask_rate
            # rand_mask[one_hot] = 0  # unmask gold items
            # loss[rand_mask] = 0

            # false continuation factor
            factor = 0.5
            tensor_factor = torch.full(loss.shape, factor, device=mask.device)
            tensor_factor[one_hot.bool()] = 1  # multiply gold items by 1
            loss *= tensor_factor

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
        # assumes that all sentences have same length
        total_len = B * S
        factor = int((S + 1) / 2 * M)
        num_scores = total_len * factor

        if to_ignore_mask is not None:
            # assumes lens are the same for each M
            lens = (~to_ignore_mask).select(
                seq2_dim, 0).select(masks_dim, 0).float().sum(seq1_dim-1)
            # TODO: make it possbile to give lens as parameter
            # since we compute them already in normal loss calculation
            # and compute the to_ignore_mask here using broadcasting...

            total_len = int(lens.sum().item())
            # divide through M since each head mask contributes 1x total_len

            num_scores = (
                torch.dot((lens+1), lens) * M / 2).item()  # type: ignore
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
            weights = torch.zeros(
                *score_preds.shape,
                device=score_preds.device)
            weights[score_gold] = f_true
            weights[~score_gold] = f_false
            loss *= weights

        if to_ignore_mask is not None:
            loss = (~to_ignore_mask)*loss
        else:
            loss = torch.tril(loss)

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
            [
                torch.stack(sc_list)
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

    def get_metric(
            self,
            num_instances: int,
            lm_loss: torch.Tensor,
            arc_loss: torch.Tensor | None = None,
            perplexity: float | None = None,
            uas: float | None = None,
            att_entropy: pd.DataFrame | None = None) -> Metric:
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
                    _att_entropy=att_entropy,
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

    def train_step(
            self,
            batch: CoNLLUTokenisedBatch | EssentialBatch,
            ignore_index: int,
            perform_opt: bool = True) -> Metric:
        assert self.train_config is not None, "Config missing training params."
        assert self.optimiser is not None
        self.batch_to(batch, device=self.config.device)  # type: ignore
        logits, arc_logits = self.transformerlm(**batch)
        self.run_hooks(batch, (logits, arc_logits))
        # remove from arc_scores those that should not be used...
        lm_loss = self.loss(
            logits, batch["label_ids"],
            ignore_index=ignore_index,
            reduction="sum")
        arc_loss: torch.Tensor | None = None

        if self.train_config.dependency_mode == "supervised":
            score_preds, score_gold = self.prepare_scores(
                arc_logits, batch["masks"])
            score_preds = F.sigmoid(score_preds)
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
        if perform_opt:
            self.optimiser.step()   # update parameters
            self.optimiser.zero_grad(set_to_none=True)

        metric.detach()
        metric.to_("cpu")
        return metric

    def eval_step(
            self,
            batch: CoNLLUTokenisedBatch | EssentialBatch,
            mode: Mode,
            ignore_index: int) -> Metric:
        self.batch_to(batch, device=self.config.device)  # type: ignore

        logits, arc_logits = self.transformerlm(**batch)
        self.run_hooks(batch, (logits, arc_logits))
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
        att_entropy = None
        if mode == "supervised":
            logits_preds, score_gold = self.prepare_scores(
                arc_logits, batch["masks"])

            if logits_preds is not None and score_gold is not None:
                score_preds = F.sigmoid(logits_preds)

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
                # TODO: save these in dictionary
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
                if self.config.masks_setting == "next":
                    zeros = np.zeros((
                        preds_head.shape[0],
                        1, preds_head.shape[2]))
                    preds_head = np.concatenate(
                        (zeros, preds_head[:, :-1]), axis=1)
                    preds_child = np.concatenate(
                        (zeros, preds_child[:, :-1]), axis=1)
                    golds_head = np.concatenate(
                        (zeros, golds_head[:, :-1]), axis=1)
                    golds_child = np.concatenate(
                        (zeros, golds_child[:, :-1]), axis=1)

                preds_arcs = dummy_mask_removal(
                    merge_head_child_scores(preds_head, preds_child))
                golds_arcs = dummy_mask_removal(
                    merge_head_child_scores(golds_head, golds_child))

                # print(preds_arcs.round(2))
                # print(golds_arcs.astype(float))
                not_padding = (
                    batch["label_ids"]
                    != ignore_index).cpu().numpy()[:, 1:]

                uas_abs = sum(map(get_uas_abs, zip(
                    preds_arcs, golds_arcs, not_padding)))

                # TODO: make the model output logits and apply sigmoid later
                head_entropy = get_attention_entropy(
                    logits_preds[:middle],
                    to_ignore[:middle]).detach().cpu()
                child_entropy = get_attention_entropy(
                    logits_preds[middle:],
                    to_ignore[middle:]).detach().cpu()

                att_entropy = pd.DataFrame({"head": head_entropy.tolist(),
                                            "child": child_entropy.tolist()})
                # can make separate list of heads

        metric = self.get_metric(
            num_instances,
            lm_loss=lm_loss,
            arc_loss=arc_loss,
            perplexity=surprisal_sum,
            uas=uas_abs,
            att_entropy=att_entropy)
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
            info(
                self.config.rank, logger,
                f"Aborting training after {evals_without_improvement} "
                "evals without improvement.")
            return True
        return False

    def train_iter(
            self,
            train: DepDataset[IDSen] | DataLoader,
            eval: DepDataset[IDSen] | DataLoader,
            token_mapper: data.TokenMapper | None = None,
            **kwargs) -> Generator[
                Result,
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
        max_epochs = (
            train_config.epochs
            if train_config is not None
            else train_config.max_steps)
        # since we cannot run out of epochs if we use
        # max_steps
        best_epoch: int = 0
        best_step: int = 0
        if self.train_config.max_steps is not None:
            pbar_steps = tqdm(total=self.train_config.max_steps, desc="Steps")
        else:
            pbar_steps = None

        for epoch in tqdm(range(1, max_epochs+1), desc="Epochs"):
            self.init_hooks(train, "train", epoch, token_mapper)
            if break_training:
                epoch -= 1
                break
            info(
                self.config.rank,
                logger, f"Epoch: {epoch}/{max_epochs}")
            if self.use_ddp:
                train.sampler.set_epoch(epoch)  # type: ignore

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

                    self.init_hooks(train, "train", epoch, token_mapper)
                    self.transformerlm.train()

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

        if pbar_steps is not None:
            pbar_steps.close()
        return (epoch, total_steps), (best_epoch, best_step)

    def train(
            self,
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

    def _train(self, loader: DataLoader[IDBatch, D]) -> Iterable[Metric]:
        assert self.train_config is not None, "Config missing training params."
        self.transformerlm.train()

        def iterate():
            assert self.train_config is not None, (
                "Config missing training params.")
            metrics: list[Metric] = list()
            for i, batch in tqdm(enumerate(loader), desc="Batches"):
                metrics.append(self.train_step(
                    batch,
                    loader.dataset.keys_for_padding["label_ids"],
                    perform_opt=(po := check_perform_opt(
                        self.train_config.gradient_acc, i))))
                if po:
                    yield sum_metrics(metrics)
                    metrics = list()

        if self.train_config.use_steps:
            for e in iterate():
                yield self.gather_metrics(e)
        else:
            yield self.gather_metrics(sum_metrics(list(iterate())))

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
                for batch in tqdm(loader, desc="Batches")]
        return self.gather_metrics(sum_metrics(metrics))

    def test(
            self, token_mapper: data.TokenMapper | None = None,
            **datasets: DepDataset | DataLoader | Any
            ) -> dict[str, Metric]:
        metrics: dict[str, Metric] = {}
        for n, ds in datasets.items():
            if isinstance(ds, (DataLoader, DepDataset)):
                ds = self.get_loader(ds)
                self.init_hooks(ds, n, token_mapper=token_mapper)
                metrics[n] = self._eval(self.get_loader(ds))
                info(
                    self.config.rank, logger,
                    f"Test metric for {n} split:\n{metrics[n].info}")
        return metrics

    def predict(
            self, dataset: DepDataset,
            make_prob: bool = False,
            only_true: bool = False,
            dataset_name: str | None = None,
            token_mapper: data.TokenMapper | None = None
            ) -> tuple[list[torch.Tensor], dict[str, list[torch.Tensor]]]:
        """Returns logits and arc scores"""
        # TODO: Does this work with ddp? Batches are distributed but not
        # joined back together.
        loader = get_loader(
            dataset, batch_size=self.config.batch_size,
            bucket=False,
            shuffle=False, droplast=False,
            n_workers=self.config.n_workers)
        self.init_hooks(
            loader, (dataset_name if dataset_name is not None else "ds"),
            token_mapper=token_mapper)

        ignore_index = dataset.keys_for_padding["label_ids"]

        unpadded_logits: list[torch.Tensor] = []
        unpadded_arc_logits: defaultdict[str, list[torch.Tensor]]
        unpadded_arc_logits = defaultdict(list)
        self.transformerlm.eval()
        with torch.no_grad():
            # eval loop: no backprop on this data, to avoid storing
            # all intermediate variable
            logits: torch.Tensor
            arc_logits: dict[str, list[torch.Tensor]]
            for batch in tqdm(loader, desc="Batches"):
                self.batch_to(batch, device=self.config.device)  # type: ignore
                logits, arc_logits = self.transformerlm(**batch)
                self.run_hooks(batch, (logits, arc_logits))
                labels = batch["label_ids"]

                if make_prob:
                    logits = logits_to_probs(
                        logits,
                        not self.config.discriminative)

                if only_true:
                    logits = select_true(logits, labels, ignore_index)

                unpadded_logits.extend(
                    unpad(logits, labels, ignore_index))

                for key in arc_logits.keys():
                    unpadded_arc_logits[key].extend(
                        unpad_masks(torch.stack(
                            arc_logits[key]).swapaxes(0, 1),
                            labels, ignore_index))
        return unpadded_logits, dict(unpadded_arc_logits)

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
            idx[0, 0] = token_mapper.token2id[DUMMY]
            idx[0, 1] = token_mapper.token2id[ROOT]
            g = model.generate(
                idx, max_new_tokens=max_len).tolist()[0]
            # support an initial mask here
        else:
            conllu = data.parse_list_of_words_with_spacy(
                start.split(), min_len=0)

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
            g = model.generate(
                batch["input_ids"][0],
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

    def log_metric(
            self, metric: Metric,
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


def logits_to_true_probs(
        logits: torch.Tensor,
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


def sum_depadded(
        values: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int) -> torch.Tensor:
    values[labels == ignore_index] = 0
    return values.sum(-1)


def mean_depadded(
        values: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int) -> torch.Tensor:
    num_items = (labels != ignore_index).sum(-1)
    sums = sum_depadded(values, labels, ignore_index)
    return sums / num_items


def logits_to_perplexity(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
        softmax: bool = True) -> torch.Tensor:
    # Should we disregard first node (root from dummy?)
    surprisal = logits_to_surprisal(logits, labels, softmax)
    means = mean_depadded(surprisal, labels, ignore_index)
    return torch.exp(means)


def unpad(
        sentences: torch.Tensor, labels: torch.Tensor,
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


def get_attention_entropy(
        logits: torch.Tensor,
        to_ignore: torch.Tensor | None = None) -> torch.Tensor:
    """input shape [H, B, S, S]
    with H: heads, B: batch size, S: sequence length.
    output shape: [H]"""
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs*torch.log(probs))

    if to_ignore is not None:
        entropy[to_ignore] = torch.nan
    entropy[entropy.isnan()] = 0
    entropy = entropy.flatten(1).sum(-1)
    return entropy


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return -torch.log((1-x)/x)


def check_perform_opt(gradient_acc: int | None, i_train: int) -> bool:
    return (gradient_acc is None
            or (i_train+1) % gradient_acc == 0)
