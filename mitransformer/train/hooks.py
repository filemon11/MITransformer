
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt

from spacy.vocab import Vocab
from spacy import displacy
import spacy
import spacy.tokens
from pathlib import Path
import os

from abc import ABC, abstractmethod

from .. import data
from . import trainer

from typing import Sequence


def get_attention_fig(
        pred_matrix: torch.Tensor, gold_matrix: torch.Tensor,
        tokens: None | Sequence[str] = None):
    fontsize: float = 10
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    cbar_ax = fig.add_axes((.95, .25, .02, .5))
    labels: None | Sequence[str] = None
    if tokens is not None:
        labels = tokens
        fontsize = max(1, 10/(len(labels) / 10))
    g = sns.heatmap(
        pred_matrix.cpu().numpy(), ax=ax[0],
        cbar=False)
    if labels is not None:
        g.set_xticks(np.arange(len(labels)) + 0.5)
        g.set_xticklabels(labels, rotation=90, fontsize=fontsize)
        g.set_yticks(np.arange(len(labels)) + 0.5)
        g.set_yticklabels(labels, rotation=0, fontsize=fontsize)

    ax[0].set(adjustable='box', aspect='equal')
    ax[0].xaxis.tick_top()

    g = sns.heatmap(
        gold_matrix.cpu().numpy(), ax=ax[1], cbar_ax=cbar_ax)
    if labels is not None:
        g.set_xticks(np.arange(len(labels)) + 0.5)
        g.set_xticklabels(labels, rotation=90, fontsize=fontsize)
        g.set_yticks(np.arange(len(labels)) + 0.5)
        g.set_yticklabels(labels, rotation=0, fontsize=fontsize)

    ax[1].collections[0].colorbar.ax.tick_params(  # type: ignore
        labelsize=fontsize)

    ax[1].set(adjustable='box', aspect='equal')
    ax[1].xaxis.tick_top()
    return fig


class Hook(ABC):
    def __init__(self, note: str | None = None) -> None:
        self.note: str | None = note

    @abstractmethod
    def __call__(
            self, input: data.CoNLLUTokenisedBatch | data.EssentialBatch,
            output: tuple[
                torch.Tensor, dict[str, torch.Tensor]]) -> None:
        ...

    def init(
            self, dataloader: data.DataLoader | None = None,
            token_mapper: data.TokenMapper | None = None,
            note: str | None = None) -> None:
        self.note = note


class AttentionPlotHook(Hook):
    def __init__(
            self, directory: str, ignore_idx: int | None = None,
            token_mapper: data.TokenMapper | None = None,
            note: str | None = None) -> None:
        super().__init__(note)
        self.directory: str = directory
        self.ignore_idx: int | None = ignore_idx
        self.token_mapper: data.TokenMapper | None = token_mapper

    def __call__(
            self, input: data.CoNLLUTokenisedBatch | data.EssentialBatch,
            output: tuple[
                torch.Tensor, dict[str, torch.Tensor]]) -> None:
        for head_type, att_preds in output[1].items():
            for i, att_mats in enumerate(att_preds):
                for idx, att_p, att_g, labels, input_ids in zip(
                        input["idx"], att_mats, input["masks"][head_type],
                        input["label_ids"], input["input_ids"]):

                    att_p = F.softmax(att_p, dim=-1)
                    att_g = F.softmax(att_g, dim=-1)

                    if self.ignore_idx is not None:
                        att_p = att_p[:, labels != self.ignore_idx][
                            labels != self.ignore_idx, :]
                        att_g = att_g[:, labels != self.ignore_idx][
                            labels != self.ignore_idx, :]

                    tokens = None
                    if self.token_mapper is not None:
                        input_ids = input_ids[labels != self.ignore_idx]
                        tokens = self.token_mapper.decode(
                            [input_ids.cpu().tolist()])[0]
                    fig = get_attention_fig(att_p, att_g, tokens)
                    Path(self.directory).mkdir(parents=True, exist_ok=True)

                    base_name = tuple() if self.note is None else (self.note,)
                    fig.savefig(
                        os.path.join(
                            self.directory,
                            "att_" + "_".join(
                                base_name + (
                                    str(idx),
                                    f"{head_type}:{i}")) + ".pdf"),
                        bbox_inches='tight')
                    plt.close()

    def init(
            self, dataloader: data.DataLoader | None = None,
            token_mapper: data.TokenMapper | None = None,
            note: str | None = None) -> None:
        super().init(dataloader, token_mapper, note)
        if dataloader is not None:
            self.ignore_idx = dataloader.dataset.keys_for_padding["label_ids"]
        if token_mapper is not None:
            self.token_mapper = token_mapper


class TreePlotHook(Hook):
    def __init__(
            self, directory: str, ignore_idx: int | None = None,
            token_mapper: data.TokenMapper | None = None,
            note: str | None = None,
            masks_setting: data.MasksSetting = "current") -> None:
        super().__init__(note)
        self.directory: str = directory
        self.ignore_idx: int | None = ignore_idx
        self.token_mapper: data.TokenMapper | None = token_mapper
        self.masks_setting = masks_setting

    def __call__(
            self, input: data.CoNLLUTokenisedBatch | data.EssentialBatch,
            output: tuple[
                    torch.Tensor, dict[str, torch.Tensor]]) -> None:
        mode = []
        if self.masks_setting in ("current", "both"):
            mode.append("current")
        if self.masks_setting in ("next", "both"):
            mode.append("next")
        for m in mode:
            gov_key = f"head_{m}"
            dep_key = f"child_{m}"
            for i, (att_mats_head, att_mats_child) in enumerate(
                    zip(output[1][gov_key], output[1][dep_key])):
                att_mats_head = F.sigmoid(att_mats_head)
                att_mats_child = F.sigmoid(att_mats_child)
                for (
                    idx, att_p_h, att_p_c, att_g_h,
                    att_g_c, labels, input_ids) in zip(
                        input["idx"], att_mats_head, att_mats_child,
                        input["masks"][gov_key],
                        input["masks"][dep_key],
                        input["label_ids"],
                        input["input_ids"]):
                    labels = labels[1:].cpu()

                    if m == "next":
                        zeros = torch.zeros((
                            1, att_p_h.shape[1]),
                            device=att_p_h.device)

                        def prepend(tensor: torch.Tensor) -> torch.Tensor:
                            return torch.concatenate(
                                (zeros, tensor[:-1]), dim=0)

                        self.draw(
                            i,
                            prepend(att_p_h),
                            prepend(att_p_c),
                            prepend(att_g_h),
                            prepend(att_g_c),
                            labels=labels, idx=idx, input_ids=input_ids,
                            note2="next" if self.masks_setting == "both"
                            else "")

                    else:
                        self.draw(
                            i,
                            att_p_h, att_p_c, att_g_h, att_g_c,
                            labels, idx, input_ids,
                            "current" if self.masks_setting == "both" else "")

    def draw(
            self, i: int, att_p_h: torch.Tensor, att_p_c: torch.Tensor,
            att_g_h: torch.Tensor, att_g_c: torch.Tensor,
            labels: torch.Tensor, idx: int, input_ids,
            note2: str = ""):
        pred_arcs = trainer.dummy_mask_removal(
            trainer.merge_head_child_scores(
                att_p_h.cpu(), att_p_c.cpu()))
        gold_arcs = trainer.dummy_mask_removal(
            trainer.merge_head_child_scores(
                att_g_h.cpu(), att_g_c.cpu()))

        pred_arcs = pred_arcs[:, labels != self.ignore_idx][
                labels != self.ignore_idx, :]
        gold_arcs = gold_arcs[:, labels != self.ignore_idx][
                labels != self.ignore_idx, :]
        pred_headlist = trainer.mst(pred_arcs)
        gold_headlist = trainer.mask_to_headlist(gold_arcs)
        pred_headlist[0] = 0
        gold_headlist[0] = 0

        if self.token_mapper is not None:
            input_ids = input_ids[1:][labels != self.ignore_idx]
            tokens = self.token_mapper.decode(
                [input_ids.cpu().tolist()])[0]
        else:
            tokens = [str(i) for i in range(len(labels))]

        pred_doc = spacy.tokens.Doc(Vocab(), tokens,
                                    heads=pred_headlist.tolist(),
                                    deps=["dep"]*len(tokens))
        gold_doc = spacy.tokens.Doc(Vocab(), tokens,
                                    heads=gold_headlist.tolist(),
                                    deps=["dep"]*len(tokens))

        base_name = (
            tuple(note2) if self.note is None else (f"{self.note}_{note2}",))
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        path_pred = os.path.join(
            self.directory,
            "tree_pred_" + "_".join(
                base_name + (str(idx), str(i))) + ".svg")
        path_gold = os.path.join(
            self.directory,
            "tree_gold_" + "_".join(
                base_name + (str(idx), str(i))) + ".svg")

        pdf_pred = displacy.render(pred_doc, style='dep')
        with Path(path_pred).open("w", encoding="utf-8") as fh:
            fh.write(pdf_pred)

        pdf_gold = displacy.render(gold_doc, style='dep')
        with Path(path_gold).open("w", encoding="utf-8") as fh:
            fh.write(pdf_gold)

    def init(
            self, dataloader: data.DataLoader | None = None,
            token_mapper: data.TokenMapper | None = None,
            note: str | None = None) -> None:
        super().init(dataloader, token_mapper, note)
        if dataloader is not None:
            self.ignore_idx = dataloader.dataset.keys_for_padding["label_ids"]
        if token_mapper is not None:
            self.token_mapper = token_mapper
