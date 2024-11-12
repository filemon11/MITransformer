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

from pathlib import Path
import os
import pickle

from dataclasses import dataclass, fields
from functools import total_ordering
from collections import defaultdict

from typing import (Self, TypedDict, Literal, cast,
                    Container, Iterable, Mapping,
                    ClassVar, TypeVar, Callable,
                    NotRequired)

Mode = Literal["standard", "input", "supervised"]
M = TypeVar("M", bound="Metric")
N = TypeVar("N")


@total_ordering
@dataclass
class Metric:
    num: int = 0
    _lm_loss: torch.Tensor = torch.tensor(0)

    _to_mean: ClassVar[set[str]] = {"lm_loss"}

    _convert: ClassVar[dict[str, Callable[[N], N]]] = {}    # type: ignore

    def __getattr__(self, prop: str):
        """Calculate mean for metrics"""
        if prop in self._to_mean:
            val = super().__getattribute__(f"_{prop}") / self.num
            if prop in self._convert:
                val = self._convert[prop](val)
            return val
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
        print(f"[{epoch}/{total_epochs}] {kind}:: " + ", ".join(strs))

    def print_test(self) -> None:
        strs = [f"{name}: {getattr(self, name):.2f}" for name in self._to_mean]
        print("Test results: " + ", ".join(strs))

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


class GeneralConfig(TypedDict):
    batch_size: int
    mode: Mode
    arc_loss_weighted: bool
    device: str


class TrainConfig(GeneralConfig):
    eval_interval: int
    epochs: int
    learning_rate: float
    model_dir: str


class OptionalConfig(TypedDict):
    batch_size: NotRequired[int]
    mode: NotRequired[Mode]
    arc_loss_weighted: NotRequired[bool]
    eval_interval: NotRequired[int]
    epochs: NotRequired[int]
    learning_rate: NotRequired[float]
    model_dir: NotRequired[str]
    device: NotRequired[str]


class LMTrainer():
    def __init__(self, transformerlm: MITransformerLM,
                 transformer_config: MITransformerConfig,
                 config: GeneralConfig):
        self.transformerlm: MITransformerLM = transformerlm
        self.transformerlm.to(config["device"])
        self.transformer_config: MITransformerConfig = transformer_config

        self.optimiser: Optimizer | None
        self.__config: GeneralConfig
        self.config = config

    @property
    def config(self) -> GeneralConfig:
        return self.__config

    @config.setter
    def config(self, config: GeneralConfig) -> None:
        self.__config = config
        if "learning_rate" in config:
            self.optimiser = Adam(
                self.transformerlm.parameters(),
                lr=config["learning_rate"])  # type: ignore
        else:
            self.optimiser = None

    @property
    def train_config(self) -> TrainConfig | None:
        if "learning_rate" in self.config:
            return cast(TrainConfig, self.config)
        else:
            return None

    @classmethod
    def load(cls, model_dir: str,
             optional_config: OptionalConfig | None = None) -> Self:

        state_dict, transformer_config = torch.load(
            os.path.join(model_dir, "model")).values()
        transformer_config = cast(MITransformerConfig, transformer_config)
        model: MITransformerLM = MITransformerLM(
            MITransformer(**transformer_config))

        model.load_state_dict(state_dict)

        with open(os.path.join(model_dir, "config"), 'rb') as handle:
            config = pickle.load(handle)

        if optional_config is not None:
            for key, value in optional_config.items():
                config[key] = value

        return cls(model, transformer_config, config)

    @classmethod
    def new(cls, trainsformer_config: MITransformerConfig,
            config: GeneralConfig) -> Self:
        model: MITransformerLM = MITransformerLM(
            MITransformer(**trainsformer_config))
        model_parameters = filter(
            lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Number of parameters: {params}")
        return cls(model, trainsformer_config, config)

    def save(self, model_dir: str) -> None:
        assert self.train_config is not None

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": self.transformerlm.state_dict(),
             "config": self.transformer_config},
            os.path.join(model_dir, "model"))

        # overwrites config
        with open(os.path.join(model_dir, "config"), 'wb') as handle:
            pickle.dump(self.train_config,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

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

        if self.config["arc_loss_weighted"]:
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

    def train_step(self,
                   batch: CoNNLUTokenisedBatch | EssentialBatch,
                   ignore_index: int) -> Metric:
        assert self.train_config is not None, "Config missing training params."
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

        labels = batch["label_ids"]
        lm_loss = self.loss(
            logits, labels,
            ignore_index=ignore_index, reduction="sum")

        loss = lm_loss
        score_preds, score_gold = self.prepare_scores(
            arc_scores, batch["masks"])

        num_instances = batch["input_ids"][batch["input_ids"]
                                           != ignore_index].numel()

        surprisal_sum = sum_unpadded(logits_to_surprisal(logits, labels),
                                     labels, ignore_index).sum()

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

            return SupervisedEvalMetric(
                num_instances,
                lm_loss.detach().cpu(),
                surprisal_sum.detach().cpu().item(),
                arc_loss.detach().cpu(),
                uas_abs)

        return EvalMetric(num_instances, lm_loss.detach().cpu(),
                          surprisal_sum.detach().cpu().item())

    def train(self,
              train: DepDataset[IDSen],
              eval: DepDataset[IDSen],
              test: DepDataset[IDSen] | None = None,
              **kwargs) -> None:
        assert self.train_config is not None, "Config missing training params."
        assert self.train_config["batch_size"] <= len(train), (
            "Batch size larger than dataset.")
        train_config = self.train_config
        device = train_config["device"]

        train_loader = get_loader(
            train, batch_size=train_config["batch_size"],
            bucket=False, device=device)

        eval_loader = get_loader(
            eval, batch_size=train_config["batch_size"],
            bucket=False, device=device,
            shuffle=False, droplast=False)

        best: float | Metric = math.inf
        self.transformerlm.train()
        for epoch in range(1, train_config["epochs"]+1):
            metric = self._train(train_loader)
            metric.print(epoch, self.train_config["epochs"], "train")

            if epoch % train_config["eval_interval"] == 0:
                metric = self._eval(eval_loader)
                metric.print(epoch, self.train_config["epochs"], "eval")
                self.transformerlm.train()

                if metric > best:       # greater means better
                    best = metric
                    self.save(train_config["model_dir"])
                    print(f"Saving model at epoch {epoch}...")

        if test is not None:
            pass
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

    def _train(self, loader: DataLoader[IDBatch, D]) -> Metric:
        assert self.train_config is not None, "Config missing training params."
        self.transformerlm.train()
        metric: Metric = SupervisedMetric()
        for batch in loader:
            metric += self.train_step(
                batch,
                loader.dataset.keys_for_padding["label_ids"])

        return metric

    def _eval(self, loader: DataLoader[IDBatch, D]) -> Metric:
        self.transformerlm.eval()
        with torch.no_grad():
            # eval loop: no backprop on this data, to avoid storing
            # all intermediatte variable
            metric: Metric = SupervisedEvalMetric()
            for batch in loader:
                metric += self.eval_step(
                    batch,
                    self.config["mode"],
                    loader.dataset.keys_for_padding["label_ids"])
        return metric

    def test(self, dataset: DepDataset) -> None:
        loader = get_loader(
            dataset, batch_size=self.config["batch_size"],
            bucket=False, device=self.config["device"],
            shuffle=False, droplast=False)

        self._eval(loader).print_test()

    def predict(
            self, dataset: DepDataset,
            make_prob: bool = False,
            only_true: bool = False
            ) -> tuple[list[torch.Tensor], dict[str, list[torch.Tensor]]]:
        """Returns logits and arc scores"""
        loader = get_loader(
            dataset, batch_size=self.config["batch_size"],
            bucket=False, device=self.config["device"],
            shuffle=False, droplast=False)

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
                logits, arc_scores = self.transformerlm(**batch)
                print(logits.shape)
                labels = batch["label_ids"]

                if make_prob:
                    logits = logits_to_probs(logits)

                if only_true:
                    logits = select_true(logits, labels, ignore_index)
                    print(logits[0])

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

        if start is None:
            idx = torch.zeros((1, 2),
                              dtype=torch.long,
                              device=self.config["device"])
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


#def logits_to_probs(logits: torch.Tensor,
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
    return torch.gather(preds, -1, labels.to(torch.int64)).squeeze(-1)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def logits_to_true_probs(logits: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
    probs = logits_to_probs(logits)
    return select_true(probs, labels)


def logits_to_surprisal(logits: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
    return -torch.log(logits_to_true_probs(logits, labels))


def sum_unpadded(values: torch.Tensor,
                 labels: torch.Tensor,
                 ignore_index: int) -> torch.Tensor:
    values[labels == ignore_index] = 0
    return values.sum(-1)


def mean_unpadded(values: torch.Tensor,
                  labels: torch.Tensor,
                  ignore_index: int) -> torch.Tensor:
    num_items = (labels != ignore_index).sum(-1)
    sums = sum_unpadded(values, labels, ignore_index)
    return sums / num_items


def logits_to_perplexity(logits: torch.Tensor,
                         labels: torch.Tensor,
                         ignore_index: int) -> torch.Tensor:
    # Should we disregard first node (root from dummy?)
    surprisal = logits_to_surprisal(logits, labels)
    means = mean_unpadded(surprisal, labels, ignore_index)
    return torch.exp(means)


def unpad(sentences: torch.Tensor, labels: torch.Tensor,
          ignore_index: int) -> list[torch.Tensor]:
    unpadded_list: list[torch.Tensor] = []
    for sentence, sen_labels in zip(sentences, labels):
        unpadded_list.append(sentence[sen_labels != ignore_index])
    print(unpadded_list[0].shape)
    return unpadded_list


def unpad_masks(masks: torch.Tensor, labels: torch.Tensor,
                ignore_index: int) -> list[torch.Tensor]:
    unpadded_list: list[torch.Tensor] = []
    for sentence, sen_labels in zip(masks, labels):
        unpadded_list.append(
            sentence[:, sen_labels != ignore_index][
                :, :, sen_labels != ignore_index])
    return unpadded_list
