from model import MITransformer, MITransformerLM, MITransformerConfig
from data import (DepDataset, DUMMY, ROOT, EOS,
                  get_transform_mask_head_child, CoNLLUDataset,
                  DataLoader, get_loader, IDSen, IDBatch,
                  CoNNLUTokenisedBatch, EssentialBatch, D)
from tokeniser import TokenMapper
from parse import parse_list_of_words_with_spacy
from dependencies import (plot_tree, mst, merge_head_child_scores,
                          dummy_mask_removal, mask_to_headlist, uas_absolute)

import seaborn as sns   # type: ignore
import matplotlib.pyplot as plt   # type: ignore

import numpy as np
import torch
from torch.optim.adam import Adam
from torch.optim import Optimizer
import torch.nn.functional as F

import math

from dataclasses import dataclass, fields
from functools import total_ordering

from typing import (Self, TypedDict, Literal, cast,
                    Container, Iterable, Mapping,
                    ClassVar, TypeVar)

Mode = Literal["standard", "input", "supervised"]
M = TypeVar("M", bound="Metric")


@total_ordering
@dataclass
class Metric:
    num: int = 0
    _lm_loss: torch.Tensor = torch.tensor(0)

    _to_mean: ClassVar[set[str]] = {"lm_loss"}

    def __getattr__(self, prop: str):
        """Calculate mean for metrics"""
        if prop in self._to_mean:
            return super().__getattribute__(f"_{prop}") / self.num
        else:
            raise AttributeError(
                f"'{self.__class__}' has no attribute '{prop}' or '_{prop}'.")

    def __add__(self, other: Self) -> Self:
        # other must be a lower type or Self

        higher = None
        if isinstance(self, other.__class__):
            higher = other
        elif isinstance(other, self.__class__):
            higher = self
        assert higher is not None, (f"Cannot add metrics "
                                    f"of types {self.__class__} "
                                    f"and {other.__class__}")

        return higher.__class__(
            *[getattr(self, f.name) + getattr(other, f.name)
              for f in fields(higher)])

    def __radd__(self, other: M) -> M:
        return other + self

    def print(self, epoch: int, 
              total_epochs: int, kind: str) -> None:
        strs = [f"{name}: {getattr(self, name):.2f}" for name in self._to_mean]
        print(f"[{epoch}/{total_epochs}] {kind}: " + ", ".join(strs))

    @property
    def loss(self) -> torch.Tensor:
        return getattr(self, "lm_loss")

    def detach(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                setattr(self, f.name, value.detach())

    def to(self, device: str | torch.device) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                setattr(self, f.name, value.to(device))

    @property
    def main(self) -> torch.Tensor:
        return self.loss

    def __gt__(self, other: object) -> bool:
        if isinstance(other, Metric):
            return self.main.item() < other.main.item()
        else:
            try:
                return self.main.item() < float(other)  # type: ignore
            except ValueError:
                return False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Metric):
            return self.main.item() == other.main.item()
        else:
            try:
                return self.main.item() == float(other)  # type: ignore
            except ValueError:
                return False


@dataclass
class SupervisedMetric(Metric):
    _arc_loss: torch.Tensor = torch.tensor(0)
    _to_mean: ClassVar[set[str]] = Metric._to_mean | {"arc_loss"}

    @property
    def loss(self) -> torch.Tensor:
        return (self._lm_loss + self._arc_loss) / self.num


@dataclass
class UASMetric(SupervisedMetric):
    _uas: float = 0
    _to_mean: ClassVar[set[str]] = SupervisedMetric._to_mean | {"arc_loss",
                                                                "uas"}


class GeneralConfig(TypedDict):
    batch_size: int


class TrainConfig(GeneralConfig):
    eval_interval: int
    epochs: int
    learning_rate: float
    mode: Mode
    model_dir: str
    arc_loss_weighted: bool


class LMTrainer():
    def __init__(self, transformerlm: MITransformerLM,
                 transformer_config: MITransformerConfig,
                 train_config: TrainConfig | None,
                 device: str = "cpu"):
        self.transformerlm: MITransformerLM = transformerlm
        self.transformerlm.to(device)
        self.device: str = device
        self.transformer_config: MITransformerConfig = transformer_config

        self.optimiser: Optimizer | None
        self.__train_config: TrainConfig | None
        self.train_config = train_config

    @property
    def train_config(self) -> TrainConfig | None:
        return self.__train_config

    @train_config.setter
    def train_config(self, train_config: TrainConfig | None) -> None:
        self.__train_config = train_config
        if train_config is None:
            self.optimiser = None
        else:
            self.optimiser = Adam(self.transformerlm.parameters(),
                                  lr=train_config["learning_rate"])

    @classmethod
    def load(cls, model_dir: str,
             train_config: TrainConfig | None,
             device: str = "cpu",) -> Self:
        state_dict, config = torch.load(model_dir).values()
        config = cast(MITransformerConfig, config)
        model: MITransformerLM = MITransformerLM(MITransformer(**config))
        model.load_state_dict(state_dict)
        return cls(model, config, train_config, device)

    @classmethod
    def new(cls, config: MITransformerConfig,
            train_config: TrainConfig | None,
            device: str = "cpu") -> Self:
        model: MITransformerLM = MITransformerLM(MITransformer(**config))
        model_parameters = filter(
            lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of parameters: {params}")
        return cls(model, config, train_config, device)

    def save(self, model_dir: str) -> None:
        torch.save(
            {"model": self.transformerlm.state_dict(),
             "config": self.transformer_config},
            model_dir)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor,
             ignore_index: int = -100,
             reduction: Literal["sum", "mean"] = "mean"
             ) -> torch.Tensor:
        logits = torch.swapaxes(logits, 1, 2)
        loss = F.cross_entropy(
            logits, labels.to(torch.long),
            ignore_index=ignore_index,
            reduction=reduction)
        return loss

    def arc_loss(
            self, score_preds: torch.Tensor,
            score_gold: torch.BoolTensor,
            to_ignore_mask: torch.BoolTensor | None,
            reduction: Literal["sum", "mean"] = "mean"
            ) -> torch.Tensor:
        assert self.train_config is not None
        """reduction sum takes a mean across dim 1
        of the mask"""
        batch_size = score_preds.shape[1]
        seq_len = score_preds.shape[2]
        total_len = batch_size * seq_len
        factor = seq_len+1
        num_scores = total_len * factor

        if to_ignore_mask is not None:

            # compute number of non-padding tokens to compute
            # number of arcs
            lens = (~to_ignore_mask[0, :, 0]).sum(1).cpu()  # type: ignore

            num_scores = (torch.dot((lens+1), lens)).item()  # type: ignore
            factor: int = num_scores / lens.sum().item()  # type: ignore
            # divide score/factor by number of tokens to get average
            # per-token loss
            score_gold = cast(torch.BoolTensor,
                              score_gold[~to_ignore_mask])

            score_preds = score_preds[~to_ignore_mask]

        loss = F.binary_cross_entropy(
            score_preds,
            score_gold.to(score_preds.dtype),
            reduction='none')

        if self.train_config["arc_loss_weighted"]:
            total_el = score_gold.numel()
            true_el = torch.sum(score_gold)
            false_el = total_el - true_el
            f_true = 0.5*(total_el)/true_el
            f_false = 0.5*(total_el)/false_el
            weights = torch.zeros(*score_preds.shape,
                                  device=score_preds.device)
            weights[score_gold] = f_true
            weights[~score_gold] = f_false
            loss *= weights

        loss = torch.sum(loss)
        if reduction == "mean":
            loss /= num_scores
        else:
            loss /= factor
            # Each position can be attended to S+1 times
        return loss

    @staticmethod
    def filter_arc_scores(
            arc_scores: Mapping[str, list[torch.Tensor]],
            keep_keys: Container | Iterable) -> dict[str, list[torch.Tensor]]:
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
        if arc_scores is None or masks is None:
            return None, None
        else:
            assert isinstance(score_preds, torch.Tensor)
            assert isinstance(score_gold, torch.Tensor)
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

    def train_step(self,
                   batch: CoNNLUTokenisedBatch | EssentialBatch,
                   ignore_index: int) -> Metric:
        assert self.train_config is not None, "train_config not specified"
        assert self.optimiser is not None

        logits, arc_scores = self.transformerlm(**batch)
        # remove from arc_scores those that should not be used...
        lm_loss = self.loss(
            logits, batch["label_ids"],
            ignore_index=ignore_index,
            reduction="sum")
        arc_loss: torch.Tensor | None = None

        loss = lm_loss
        score_preds, score_gold = self.prepare_scores(
            arc_scores, batch["masks"])
        if (self.train_config["mode"] == "supervised"
                and score_preds is not None
                and score_gold is not None):

            to_ignore = self.get_ignore_mask(
                score_preds,
                batch["label_ids"],
                ignore_index)

            arc_loss = self.arc_loss(
                score_preds,
                score_gold,
                to_ignore,
                reduction="sum")
            loss += arc_loss

        num_instances = (batch["input_ids"] != ignore_index).numel()

        if arc_loss is None:
            metric = Metric(num_instances, lm_loss)
        else:
            metric = SupervisedMetric(num_instances, lm_loss, arc_loss)

        metric.loss.backward()   # backward pass
        self.optimiser.step()   # update parameters
        self.optimiser.zero_grad(set_to_none=True)

        metric.detach()
        metric.to("cpu")
        return metric

    def eval_step(self,
                  batch: CoNNLUTokenisedBatch | EssentialBatch,
                  mode: Mode,
                  ignore_index: int) -> Metric:
        logits, arc_scores = self.transformerlm(**batch)
        # remove from arc_scores those that should not be used...
        lm_loss = self.loss(
            logits, batch["label_ids"],
            ignore_index=ignore_index, reduction="sum")

        loss = lm_loss
        score_preds, score_gold = self.prepare_scores(
            arc_scores, batch["masks"])

        num_instances = batch["input_ids"][batch["input_ids"]
                                           != ignore_index].numel()

        if (mode == "supervised"
                and score_preds is not None
                and score_gold is not None):

            to_ignore = self.get_ignore_mask(
                score_preds,
                batch["label_ids"],
                ignore_index)

            arc_loss = self.arc_loss(
                score_preds,
                score_gold,
                to_ignore,
                reduction="sum")
            loss += arc_loss

            preds_head = score_preds[0].detach().cpu().numpy()
            golds_head = score_gold[0].detach().cpu().numpy()
            preds_child = score_preds[1].detach().cpu().numpy()
            golds_child = score_gold[1].detach().cpu().numpy()

            preds_arcs = dummy_mask_removal(
                merge_head_child_scores(preds_head, preds_child))
            golds_arcs = dummy_mask_removal(
                merge_head_child_scores(golds_head, golds_child))

            not_padding = (batch["input_ids"][:, 1:]
                           != ignore_index).cpu().numpy()
            uas_abs = 0
            for p, g, n in zip(preds_arcs, golds_arcs, not_padding):
                p = p[n][:, n]
                g = g[n][:, n]
                pred_headlist = mst(p)
                gold_headlist = mask_to_headlist(g)
                uas_abs += uas_absolute(pred_headlist, gold_headlist)

            return UASMetric(num_instances,
                             lm_loss.detach().cpu(),
                             arc_loss.detach().cpu(),
                             uas_abs)

        return Metric(num_instances, lm_loss.detach().cpu())

    def train(self,
              train: DepDataset[IDSen],
              eval: DepDataset[IDSen],
              test: DepDataset[IDSen] | None = None) -> None:
        assert self.train_config is not None, "train_config not specified"
        train_config = self.train_config

        train_loader = get_loader(
            train, batch_size=train_config["batch_size"],
            bucket=False, device=self.device)

        eval_loader = get_loader(
            eval, batch_size=train_config["batch_size"],
            bucket=False, device=self.device,
            shuffle=False, droplast=False)

        best: float | Metric = math.inf
        self.transformerlm.train()
        for epoch in range(1, train_config["epochs"]+1):
            metric: Metric = SupervisedMetric()
            for batch in train_loader:
                metric += self.train_step(
                    batch,
                    train.keys_for_padding["label_ids"])
            # print progress
            # metric.print(epoch, self.train_config["epochs"], "train")

            if epoch % train_config["eval_interval"] == 0:
                metric = self._eval(train_loader)
                metric.print(epoch, self.train_config["epochs"], "train")

                metric = self._eval(eval_loader)
                metric.print(epoch, self.train_config["epochs"], "eval")
                self.transformerlm.train()

                if metric > best:       # greater means better
                    best = metric
                    self.save(train_config["model_dir"])
                    print(f"Saving model at epoch {epoch}...")

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
        #        not_padding = (batch["input_ids"][i] != token_mapper.pad_id).cpu().numpy()
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

    def _eval(self, loader: DataLoader[IDBatch, D]) -> Metric:
        assert self.train_config is not None, "train_config not specified"
        self.transformerlm.eval()
        with torch.no_grad():
            # eval loop: no backprop on this data, to avoid storing
            # all intermediatte variable
            metric: Metric = UASMetric()
            for batch in loader:
                metric += self.eval_step(
                    batch,
                    self.train_config["mode"],
                    loader.dataset.keys_for_padding["label_ids"])
        return metric

    def test(self, dataset: DepDataset, test_config: dict) -> None:
        pass

    def predict(
            self, dataset: DepDataset,
            predict_config: dict) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def generate(
            self, token_mapper: TokenMapper,
            start: str | None = None, max_len: int = 40) -> str:

        g: list[int]

        if start is None:
            idx = torch.zeros((1, 2), dtype=torch.long, device=self.device)
            idx[0, 0] = token_mapper.token2id[DUMMY]
            idx[0, 1] = token_mapper.token2id[ROOT]
            g = self.transformerlm.generate(
                idx, max_new_tokens=max_len).tolist()[0]
            # support an initial mask here
        else:
            conllu = parse_list_of_words_with_spacy(start.split(), min_len=0)

            transform = get_transform_mask_head_child(
                keys_for_head={"head"},
                keys_for_child={"child"},
                triangulate=True)

            dataset: CoNLLUDataset = CoNLLUDataset.from_str(
                conllu, transform, max_len=None)

            dataset.map_to_ids(token_mapper)
            dataloader: DataLoader = get_loader(
                dataset, batch_size=1,
                bucket=False, device=self.device,
                shuffle=False, droplast=False)

            for batch in dataloader:
                # take last batch, i.e. last sentence
                # TODO: this is a weird solution
                pass

            # TODO: support mask
            g = self.transformerlm.generate(batch["input_ids"][0],
                                            max_new_tokens=max_len).tolist()[0]

        eos_id = token_mapper.token2id[EOS]
        first_eos = next((i for i, x in enumerate(g) if x == eos_id), len(g))
        g = g[:first_eos + 1]
        return token_mapper.decode([g], to_string=True)[0]
