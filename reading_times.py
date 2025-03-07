"""Snippets taken from
https://github.com/weijiexu-charlie/
Linearity-of-surprisal-on-RT/blob/main/Preparing%20Corpora/get_frequency.py"""


import pandas as pd
import wordfreq     # type: ignore
import nltk  # type: ignore
from tokeniser import TokenMapper
from data import CoNLLUDataset, TransformMaskHeadChild
from trainer import LMTrainer, inverse_sigmoid
from parse import parse_list_of_words_with_spacy

from natural_stories import load_natural_stories

import torch
import numpy as np

from typing import Iterable, Literal, Callable, TypeVar
import numpy.typing as npt

import math

import sys
'''
The input files are the meta data (text without RT) of the corpus
that is already parsed and contains logp information. The output
files include one additional column of log-scaled frequency retrieved
from the package wordfreq (https://zenodo.org/records/7199437)
'''


pd.set_option('mode.chained_assignment', None)

LANG = "en"

TOKEN_COL = "WORD"
LOGFREQ_COL = "logfreq"
TEXT_ID_COL = "item"
WNUM_COL = "zone"
SURPRISAL_COL = "logp"
WORD_POSITION_COL = "position"
HEAD_DISTANCE_COL = "headdistance"
INTEGRATION_COST_COL = "incost"
FDD_COL = "fdd"  # first dependency distance
FDDC_COL = "fddc"  # first dependency distance correct?
FDDW_COL = "fddw"  # first dependency distance weight
LDDS_COL = "ldds"  # left dependency distance sum
LDC_COL = "ldc"  # left dependency count
POS_TAG_COL = "POS"
DEPREL_COL = "deprel"
WLEN_COL = "WLEN"

DEVICE = "cpu"
BATCH_SIZE = 8

TAGSET = None  # "universal"  # None

if TAGSET is None:
    CONTENT_POS = {"FW", "MD", "NN", "NNS", "NNP",
                   "NNPS", "VB", "VBD", "VBG", "VBN",
                   "VBP", "VBZ", "JJ", "JJR", "JJS"}
    PUNCTUATION = {"''", "(", "SYM", "POS"}
    MERGE_MAPPING = {"JJS": "JJ", "JJR": "JJ",
                     "PRP$": "PRP",
                     "WP$": "PRP", "WP": "PRP", "WRB": "RB", "WDT": "DT",
                     "VBD": "VB", "VBG": "VB", "VBN": "VB",
                     "VBP": "VB", "VBZ": "VB",
                     "PDT": "DT",
                     "RBR": "RB", "RBS": "RB",
                     "NNS": "NN", "NNPS": "NN", "NNP": "NN",
                     "PRP$": "PRP",
                     "UH": "RP",
                     "SYM": "NN"}
    # in natural stories SYM is assigned e.g. to thirty-two in 1632

else:
    CONTENT_POS = {"NOUN", "VERB"}  # , "JJ", "JJR", "JJS"}
    PUNCTUATION = {"."}
    MERGE_MAPPING = {}


def pos_merge(tag: str) -> str:
    if tag not in MERGE_MAPPING.keys():
        return tag
    else:
        return MERGE_MAPPING[tag]


def get_frequency(token, language: str = LANG):
    return wordfreq.zipf_frequency(token, language)


def add_frequency(input_file, output_file, language: str = LANG,
                  token_col: str = TOKEN_COL,
                  logfreq_col: str = LOGFREQ_COL) -> None:
    df = pd.read_csv(input_file)
    df[logfreq_col] = df.apply(
            lambda df: get_frequency(df[token_col], language), axis=1)
    df.to_csv(output_file, index=False)


def add_word_length(input_file, output_file,
                    token_col: str = TOKEN_COL,
                    wlen_col: str = WLEN_COL) -> None:
    df = pd.read_csv(input_file)
    df[wlen_col] = df.apply(
            lambda df: len(df[token_col]), axis=1)
    df.to_csv(output_file, index=False)


def tsv_to_csv(input_file: str, output_file: str,
               token_col: str = TOKEN_COL,
               text_id_col: str = TEXT_ID_COL,
               wnum_col: str = WNUM_COL,
               token_mapper_dir: str | None = None
               ) -> None:
    tokens, text_ids, wnums = load_natural_stories(
        input_file, token_mapper_dir=token_mapper_dir)
    # making lowercase makes no difference

    df = pd.DataFrame({token_col: tokens,
                       text_id_col: text_ids,
                       wnum_col: wnums})

    df.to_csv(output_file, index=False)


def generate_probs(
        trainer: LMTrainer,
        dataset: CoNLLUDataset,
        untokenise: bool = False,
        surprisal: bool = False) -> tuple[torch.Tensor, npt.NDArray[np.int_],
                                          dict[str, list[torch.Tensor]]]:
    """WARNING: untested"""

    pred_probs, attention = trainer.predict(
        dataset,
        make_prob=True,
        only_true=True)

    # Throw away probs of root and eos token
    probs = torch.cat([p[1:-1] for p in pred_probs])
    indices: npt.NDArray[np.int_]

    if untokenise:
        assert dataset.space_after is not None
        space_after = np.concat(dataset.space_after)

        probs = torch.tensor(
            list(untokenise_probs(
                probs.tolist(),
                space_after,
                mode="mult")))
        indices = np.concat(
            [np.arange(1, sen.sum()+1) for sen in dataset.space_after])

    else:
        indices = np.concat(
            [np.arange(1, sen.shape[0]-1) for sen in pred_probs])

    indices = np.append(indices, indices[-1]+1)

    if surprisal:
        probs = -torch.log(probs)

    return probs, indices, attention


def get_POS_tags(sentence: list[str]) -> list[str]:
    tagged = nltk.tag.pos_tag(sentence, tagset=TAGSET)
    return [pos_merge(tag) for _, tag in tagged]


def generate_POS_tags(
        pos_tag_list: list[str],
        space_after: list[npt.NDArray[np.bool_]] | None,
        untokenise: bool = False
        ) -> list[str]:

    if untokenise:
        assert space_after is not None
        space_after_ = np.concat(space_after)

        new_pos_tag_list: list[str] = []
        current_pos: str | None = None
        found_non_punct: bool = False
        for space_after_tok, new_pos in zip(space_after_, pos_tag_list):
            if current_pos is None:
                current_pos = new_pos
            if not found_non_punct and new_pos not in PUNCTUATION:
                current_pos = new_pos
                found_non_punct = True
            if space_after_tok:
                new_pos_tag_list.append(current_pos)
                found_non_punct = False
                current_pos = None

        if current_pos is not None:
            new_pos_tag_list.append(current_pos)

        pos_tag_list = new_pos_tag_list

    return pos_tag_list


def get_content_word_mask(
        sentence: list[str],
        not_to_mask: set[str] = CONTENT_POS,
        dummies_present: bool = True) -> npt.NDArray[np.bool]:
    pos_tags = get_POS_tags(sentence)
    mask = np.array([tag in not_to_mask for tag in pos_tags], dtype=bool)
    if dummies_present:
        mask[0:2] = False
    return mask


def get_content_word_mask_by_POS_tags(
        pos_tags: list[str],
        not_to_mask: set[str] = CONTENT_POS,
        dummies_present: bool = True) -> npt.NDArray[np.bool]:
    mask = np.array([tag in not_to_mask for tag in pos_tags], dtype=bool)
    if dummies_present:
        mask[0:2] = False
    return mask


def get_content_word_cost(
        content_word_mask: npt.NDArray[np.bool],
        mask_gov: npt.NDArray[np.bool],
        mask_dep: npt.NDArray[np.bool]
        ) -> npt.NDArray[np.int_]:

    mask = np.logical_or(mask_gov, mask_dep)

    # do not take into account connections to function words
    mask[:, ~content_word_mask] = 0
    mask[0, :] = 0

    first_connection: npt.NDArray[np.int_] = np.argmax(np.tril(mask), axis=1)

    upper_triang = np.triu(
        np.ones((content_word_mask.shape[0], content_word_mask.shape[0])), 1)
    cw_cumul: np.ndarray = content_word_mask @ upper_triang

    cw_before_first = cw_cumul[first_connection]  # @ mask_gov.T
    cw_intermediate = cw_cumul - cw_before_first
    # assume no crossing arcs
    # left_cw_child_nums = content_word_mask @ mask_dep.T
    # cw_intermediate -= left_cw_child_nums
    cw_intermediate[first_connection == 0] = 0  # if dummy or root

    # cw_intermediate[mask_gov[:, 0]] = 1  # for dummy set 1
    # cw_intermediate[mask_gov[:, 1]] = 1  # for root set 1
    cost = cw_intermediate + content_word_mask.astype(int)  # new discourse referents
    cost[~content_word_mask] = 0  # integration and instantiation cost for function words is 0
    return cost


def generate_integration_Gibson(
        dataset: CoNLLUDataset,
        untokenise: bool = False,
        dummy_used: bool = True) -> npt.NDArray[np.int_]:
    """WARNING: untested"""

    costs_list: list[npt.NDArray[np.int_]] = []
    for i in range(len(dataset)):
        sentence = dataset[i]
        # assumes that governor and child mask are called head and child
        # these are already triangulated
        mask_gov = sentence["masks"]["head"]
        mask_dep = sentence["masks"]["child"]

        assert mask_gov is not None and mask_dep is not None

        root_idx = 1
        mask_dep[:, root_idx] = 0
        mask_dep[:, 0] = 0      # no cost if no children
        mask_gov[:, root_idx] = 0
        mask_gov[:, 0] = 0

        costs = get_content_word_cost(
            get_content_word_mask(dataset.tokens[i][:-1], CONTENT_POS),
            mask_gov, mask_dep)
        costs_list.append(costs[2:])

    cost_array = np.concat(costs_list)
    if untokenise:
        assert dataset.space_after is not None
        space_after = np.concat(dataset.space_after)

        cost_array = np.array(
            list(untokenise_probs(
                cost_array.tolist(),
                space_after,
                mode="add")))

    return cost_array


def generate_first_dep_distance(
        dataset: CoNLLUDataset,
        pos_tags: list[str],
        untokenise: bool = False,
        only_content_words: bool = False) -> npt.NDArray[np.int_]:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    """WARNING: untested"""

    costs_list: list[npt.NDArray[np.int_]] = []
    elements_in: int = 0
    for i in range(len(dataset)):
        sentence = dataset[i]
        # assumes that governor and child mask are called head and child
        # these are already triangulated
        mask_gov = sentence["masks"]["head"].copy()
        mask_dep = sentence["masks"]["child"].copy()

        if only_content_words:
            content_word_mask = get_content_word_mask_by_POS_tags(
                pos_tags[
                    elements_in:elements_in+mask_gov.shape[0]-2],
                CONTENT_POS,
                dummies_present=False)

            mask_gov[:, 2:][:, ~content_word_mask] = False
            mask_dep[:, 2:][:, ~content_word_mask] = False

        assert mask_gov is not None and mask_dep is not None

        root_idx = 1
        mask_dep[:, root_idx] = 0
        mask_dep[:, 0] = 0      # no cost if no children
        mask_gov[:, root_idx] = 0
        mask_gov[:, 0] = 0

        mask = np.tril(np.logical_or(mask_gov, mask_dep))
        mask = np.logical_or(mask, mask.T)

        first_connection: npt.NDArray[np.int_] = np.argmax(
            mask, axis=1)

        length = mask_gov.shape[0]
        indices: npt.NDArray[np.int_] = np.tile(
            np.flip(np.arange(1, length+1)), length).reshape(length, -1)
        indices = indices - np.flip(
            np.arange(1, length+1))[..., np.newaxis]

        distances = -indices[
            np.arange(0, first_connection.shape[0]),
            first_connection]

        distances[first_connection == 0] = 0    # if root, then distance 0
        costs_list.append(distances[2:])

        elements_in += distances.shape[0]-2

    cost_array = np.concat(costs_list)

    if untokenise:
        assert dataset.space_after is not None
        space_after = np.concat(dataset.space_after)

        # take the first distance to disregard punctuation
        # which may connect over long distances
        cost_list: list[int] = cost_array.tolist()
        new_cost_list: list[int] = []
        current_cost: float | None = None
        found_non_punct: bool = False
        for space_after_tok, pos, new_cost in zip(
                space_after, pos_tags, cost_list):
            if current_cost is None:
                current_cost = new_cost
            if not found_non_punct and pos not in PUNCTUATION:
                current_cost = new_cost
                found_non_punct = True
            if space_after_tok:
                new_cost_list.append(current_cost)
                found_non_punct = False
                current_cost = None

        if current_cost is not None:
            new_cost_list.append(current_cost)

        cost_array = np.array(new_cost_list)

    return cost_array


def generate_fddc(
        dataset: CoNLLUDataset,
        attention_matrices: dict[str, list[torch.Tensor]],
        pos_tags: list[str],
        untokenise: bool = False,
        only_content_words: bool = True) -> npt.NDArray[np.bool]:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    # TODO: Discard punctuation
    """WARNING: untested"""

    fdd = generate_first_dep_distance(
        dataset, pos_tags, untokenise=False,
        only_content_words=only_content_words)

    costs_list: list[npt.NDArray[np.bool]] = []
    elements_in: int = 0
    for i in range(len(
            attention_matrices[next(iter(attention_matrices.keys()))])):
        # assumes that governor and child mask are called head and child
        # these are already triangulated
        mask_gov = attention_matrices["head"][i][0].clone()
        mask_dep = attention_matrices["child"][i][0].clone()

        if only_content_words:
            # disregard connections to non-content words
            content_word_mask = get_content_word_mask_by_POS_tags(
                pos_tags[elements_in:elements_in+mask_gov.shape[0]-2],
                CONTENT_POS,
                dummies_present=False)
            mask_gov[:, 2:][:, ~content_word_mask] = 0
            mask_dep[:, 2:][:, ~content_word_mask] = 0

        # mask_gov_ = np.tril(mask_gov) >= 0.5
        # mask_dep_ = np.tril(mask_dep) >= 0.5
        # mask = np.tril(np.logical_or(mask_gov_, mask_dep_))

        # first_connection: npt.NDArray[np.int_] = np.argmax(
        #    mask, axis=1)
        mask_gov_ = np.tril(mask_gov)
        mask_dep_ = np.tril(mask_dep)
        first_connection_gov: npt.NDArray[np.int_] = np.argmax(
            mask_gov_, axis=1)
        first_connection_dep: npt.NDArray[np.int_] = np.argmax(
            mask_dep_, axis=1)
        first_connection = np.minimum(
            first_connection_gov, first_connection_dep)
        no_left_head = first_connection_gov < 2
        no_left_child = first_connection_dep < 2

        cond = np.logical_and(no_left_head, np.logical_not(no_left_child))
        first_connection[cond] = first_connection_dep[cond]

        cond = np.logical_and(no_left_child, np.logical_not(no_left_head))
        first_connection[cond] = first_connection_gov[cond]

        length = mask_gov.shape[0]
        indices: npt.NDArray[np.int_] = np.tile(
            np.flip(np.arange(1, length+1)), length).reshape(length, -1)
        indices = indices - np.flip(
            np.arange(1, length+1))[..., np.newaxis]

        distances = -indices[
            np.arange(0, first_connection.shape[0]),
            first_connection]
        distances[np.logical_and(no_left_head, no_left_child)] = 0
        # distance where there is no left element is set to 0

        distances = distances[2:]

        fdd_sen = np.copy(fdd[elements_in:elements_in+distances.shape[0]])
        fdd_sen[fdd_sen > 0] = 0
        fddc = fdd_sen == distances

        costs_list.append(fddc)
        elements_in += distances.shape[0]

    cost_array = np.concat(costs_list)

    if untokenise:
        assert dataset.space_after is not None
        space_after = np.concat(dataset.space_after)

        # take the first distance to disregard punctuation
        # which may connect over long distances
        cost_list: list[bool] = cost_array.tolist()
        new_cost_list: list[bool] = []
        current_cost: bool | None = None
        found_non_punct: bool = False
        for space_after_tok, pos, new_cost in zip(
                space_after, pos_tags, cost_list):
            if current_cost is None:
                current_cost = new_cost
            if not found_non_punct and pos not in PUNCTUATION:
                current_cost = new_cost
                found_non_punct = True
            if space_after_tok:
                new_cost_list.append(current_cost)
                found_non_punct = False
                current_cost = None

        if current_cost is not None:
            new_cost_list.append(current_cost)

        cost_array = np.array(new_cost_list)

    return cost_array


def generate_fddw(
        dataset: CoNLLUDataset,
        attention_matrices: dict[str, list[torch.Tensor]],
        pos_tags: list[str],
        untokenise: bool = False) -> npt.NDArray[np.bool]:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    """WARNING: untested"""

    costs_list: list[npt.NDArray] = []
    for i in range(len(
            attention_matrices[next(iter(attention_matrices.keys()))])):
        sentence = dataset[i]
        # assumes that governor and child mask are called head and child
        # these are already triangulated
        assert sentence["masks"]["head"] is not None
        assert sentence["masks"]["child"] is not None
        mask_gov = sentence["masks"]["head"].copy()
        mask_dep = sentence["masks"]["child"].copy()

        assert mask_gov is not None and mask_dep is not None

        # throw root connection and future connection together
        mask_dep[:, 0] = mask_dep[:, 0] + mask_dep[:, 1]
        mask_dep[:, 1] = 0
        mask_gov[:, 0] = mask_gov[:, 0] + mask_gov[:, 1]
        mask_gov[:, 1] = 0

        # first dependent only future dependent
        # if no left gov and dep dependent
        mask = np.tril(np.logical_or(mask_gov, mask_dep))
        mask[:, 0] = np.logical_and(mask_gov[:, 0], mask_dep[:, 0])

        first_connection: npt.NDArray[np.int_] = np.argmax(
            mask, axis=1)

        # assumes that governor and child mask are called head and child
        # these are already triangulated
        mask_gov_pred = attention_matrices["head"][i][0].clone()
        mask_dep_pred = attention_matrices["child"][i][0].clone()

        # TODO to softmax
        mask_gov_pred = inverse_sigmoid(mask_gov_pred)
        mask_gov_pred = mask_gov_pred.softmax(-1)
        mask_dep_pred = inverse_sigmoid(mask_dep_pred).softmax(-1)
        mask_dep_pred = mask_dep_pred * mask_dep.sum(1)

        mask_dep_pred[:, 0] = mask_dep_pred[:, 0] + mask_dep_pred[:, 1]
        mask_dep_pred[:, 1] = 0
        mask_gov_pred[:, 0] = mask_gov_pred[:, 0] + mask_gov_pred[:, 1]
        mask_gov_pred[:, 1] = 0

        weights = torch.maximum(
            mask_gov_pred[
                np.arange(0, first_connection.shape[0]),
                first_connection],
            torch.minimum(
                mask_dep_pred[
                    np.arange(0, first_connection.shape[0]),
                    first_connection],
                torch.ones(
                    (1,), device=first_connection.device)))
        # right weight
        costs_list.append(weights[2:].detach().numpy())

    cost_array = np.concat(costs_list)

    if untokenise:
        assert dataset.space_after is not None
        space_after = np.concat(dataset.space_after)

        # take the first distance to disregard punctuation
        # which may connect over long distances
        cost_list: list[float] = cost_array.tolist()
        new_cost_list: list[float] = []
        current_cost: float | None = None
        found_non_punct: bool = False
        for space_after_tok, pos, new_cost in zip(
                space_after, pos_tags, cost_list):
            if current_cost is None:
                current_cost = new_cost
            if not found_non_punct and pos not in PUNCTUATION:
                current_cost = new_cost
                found_non_punct = True
            if space_after_tok:
                new_cost_list.append(current_cost)
                found_non_punct = False
                current_cost = None

        if current_cost is not None:
            new_cost_list.append(current_cost)

        cost_array = np.array(new_cost_list)

    return cost_array


def generate_ldds(
        dataset: CoNLLUDataset,
        pos_tags: list[str],
        untokenise: bool = False) -> npt.NDArray[np.int_]:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    """WARNING: untested"""

    costs_list: list[npt.NDArray[np.int_]] = []
    for i in range(len(dataset)):
        sentence = dataset[i]
        # assumes that governor and child mask are called head and child
        # these are already triangulated
        mask_gov = sentence["masks"]["head"]
        mask_dep = sentence["masks"]["child"]

        assert mask_gov is not None and mask_dep is not None

        root_idx = 1
        mask_dep[:, root_idx] = 0
        mask_dep[:, 0] = 0          # no cost if no children
        mask_gov[:, root_idx] = 0
        mask_gov[:, 0] = 0

        mask = np.tril(np.logical_or(mask_gov, mask_dep))

        length = mask_gov.shape[0]
        indices: npt.NDArray[np.int_] = np.tile(
            np.flip(np.arange(1, length+1)), length).reshape(length, -1)
        indices = indices - np.flip(
            np.arange(1, length+1))[..., np.newaxis]

        costs = (mask * indices).sum(1)

        costs_list.append(costs[2:])

    cost_array = np.concat(costs_list)

    if untokenise:
        assert dataset.space_after is not None
        cost_array = np.array(untokenise_by_POS(
            cost_array.tolist(),
            np.concat(dataset.space_after),
            pos_tags
        ))

    return cost_array


def generate_ldc(
        dataset: CoNLLUDataset,
        pos_tags: list[str],
        untokenise: bool = False) -> npt.NDArray[np.int_]:
    # TODO: all left ones distance sum
    # TODO: all left ones number
    """WARNING: untested"""

    costs_list: list[npt.NDArray[np.int_]] = []
    for i in range(len(dataset)):
        sentence = dataset[i]
        # assumes that governor and child mask are called head and child
        # these are already triangulated
        mask_gov = sentence["masks"]["head"]
        mask_dep = sentence["masks"]["child"]

        assert mask_gov is not None and mask_dep is not None

        root_idx = 1
        mask_dep[:, root_idx] = 0
        mask_dep[:, 0] = 0      # no cost if no children
        mask_gov[:, root_idx] = 0
        mask_gov[:, 0] = 0

        mask = np.tril(np.logical_or(mask_gov, mask_dep))
        costs = mask.sum(1)

        costs_list.append(costs[2:])

    cost_array = np.concat(costs_list)
    if untokenise:
        assert dataset.space_after is not None

        cost_array = np.array(untokenise_by_POS(
            cost_array.tolist(),
            np.concat(dataset.space_after),
            pos_tags
        ))

    return cost_array


def generate_integration_costs(
        dataset: CoNLLUDataset,
        untokenise: bool = False) -> npt.NDArray[np.int_]:
    """WARNING: untested"""

    costs_list: list[npt.NDArray[np.int_]] = []
    for i in range(len(dataset)):
        sentence = dataset[i]
        # assumes that governor and child mask are called head and child
        # these are already triangulated
        mask_gov = sentence["masks"]["head"]
        mask_dep = sentence["masks"]["child"]

        assert mask_gov is not None and mask_dep is not None

        root_idx = 1
        mask_dep[:, root_idx] = 0
        mask_dep[:, 0] = 0      # no cost if no children
        mask_gov[:, root_idx] = 0
        mask_gov[:, 0] = 0

        # restrict this to new content words OR IDEA: do not add if
        # right head (all intermediate elements have the same head)
        # has a head left of this element

        mask = mask_gov  # np.logical_or(mask_gov, mask_dep)

        length = mask_gov.shape[0]
        indices: npt.NDArray[np.int_] = np.tile(
            np.flip(np.arange(1, length+1)), length).reshape(length, -1)
        indices = indices - np.flip(
            np.arange(1, length+1))[..., np.newaxis]
        #
        # # change: if no connection at all, add distance + 5
        # heads = heads_list[i][:-1]
        # for j, row in zip(range(len(heads)), mask):
        #     if row[0]:
        #         delete = True
        #         current = j+1
        #         while current < len(heads) and heads[current] == j:
        #             current += 1
        #         if current != heads[j]:
        #             delete = False
        #         elif heads[current] > j:
        #             delete = False
        #
        #         if delete:
        #             indices[j, 0] = 1
        #         else:
        #             indices[j, 0] = 5  # -= 1  # skip over root node for distance
        #
        # # add cost dependent on intermediate nodes without left governor
        # extra_cost = np.array([0]*len(heads))
        # num = 0
        # for j in range(len(heads)):
        #     if heads[j] > j:
        #         extra_cost[j] += num
        #         num += 1
        #     else:
        #         num = 0
        #
        # # extra_cost += mask_dep[:, 2:].sum(1)
        #
        costs = (mask * indices).sum(1) + (mask_gov + mask_dep).sum(-1)
        costs_list.append(costs[2:])

    cost_array = np.concat(costs_list)
    if untokenise:
        assert dataset.space_after is not None
        space_after = np.concat(dataset.space_after)

        new_cost_list = []
        current_cost = cost_array[0]
        for i, (_, space_after_tok) in enumerate(
                zip(cost_array, space_after)):
            if not space_after_tok:
                pass
            else:
                new_cost_list.append(current_cost)
                current_pos = cost_array[i+1]
        new_cost_list.append(current_pos)
        cost_array = np.array(new_cost_list)

    return cost_array


def generate_head_distance(
        dataset: CoNLLUDataset,
        pos_tags: list[str],
        untokenise: bool = False
        ) -> npt.NDArray[np.int_]:
    """WARNING: untested"""

    costs_list: list[npt.NDArray[np.int_]] = []
    for i in range(len(dataset)):
        sentence = dataset[i]
        # assumes that governor and child mask are called head and child
        # these are already triangulated
        mask_gov = sentence["masks"]["head"]
        mask_dep = sentence["masks"]["child"]

        assert mask_gov is not None and mask_dep is not None

        mask_gov[:, 0] = 0
        mask_dep[:, 0] = 0      # no cost if no children
        mask = mask_gov + mask_dep.T
        mask[np.arange(mask.shape[0]),
             np.arange(mask.shape[0])] = mask_gov[:, 1]
        mask[:, 1] = 0

        # restrict this to new content words OR IDEA: do not add if
        # right head (all intermediate elements have the same head)
        # has a head left of this element

        length = mask_gov.shape[0]
        indices: npt.NDArray[np.int_] = np.tile(
            np.flip(np.arange(1, length+1)), length).reshape(length, -1)
        indices = indices - np.flip(
            np.arange(1, length+1))[..., np.newaxis]
        indices = indices.T

        costs = (mask * indices).sum(1)
        costs_list.append(costs[2:])

    cost_array: npt.NDArray[np.int_] = np.concat(costs_list)

    if untokenise:
        assert dataset.space_after is not None

        cost_array = np.array(untokenise_by_POS(
            cost_array.tolist(),
            np.concat(dataset.space_after),
            pos_tags
        ))

    return cost_array


def generate_deprels(
        dataset: CoNLLUDataset,
        pos_tags: list[str],
        untokenise: bool = False
        ) -> list[str]:
    """WARNING: untested"""

    deprels: list[str] = []
    for deprels_sen in dataset.deprels:
        deprels.extend(deprels_sen[2:-1])

    if untokenise:
        assert dataset.space_after is not None

        deprels = untokenise_by_POS(
            deprels,
            np.concat(dataset.space_after),
            pos_tags
        )

    return deprels


T = TypeVar("T")


def untokenise_by_POS(
        items: Iterable[T],
        space_after: list[bool] | npt.NDArray[np.bool_],
        pos_tags: list[str]
        ) -> list[T]:
    new_list: list[T] = []
    current_item: T | None = None
    found_non_punct: bool = False
    for space_after_tok, pos, new_item in zip(
            space_after, pos_tags, items):
        if current_item is None:
            current_item = new_item
        if not found_non_punct and pos not in PUNCTUATION:
            current_item = new_item
            found_non_punct = True
        if space_after_tok:
            new_list.append(current_item)
            found_non_punct = False
            current_item = None
    if current_item is not None:
        new_list.append(current_item)

    return new_list


def untokenise_probs(
        probs: Iterable[float],
        space_after: Iterable[bool],
        mode: Literal["mult", "add"] = "mult"
        ) -> Iterable[float]:
    base: float
    func: Callable[[float, float], float]
    match mode:
        case "add":
            base = 0
            func = lambda x, y: x+y  # noqa: E731
        case _:
            base = 1
            func = lambda x, y: x*y  # noqa: E731

    current_prob: float = base
    for prob_tok, space_after_tok in zip(probs, space_after):
        current_prob = func(current_prob, prob_tok)
        if not space_after_tok:
            pass
        else:
            yield current_prob
            current_prob = base
    yield current_prob


def untokenise_tokens(
        tokens: Iterable[str],
        space_after: Iterable[bool],
        ) -> Iterable[str]:
    current_token: str = ""
    for token, space_after_tok in zip(tokens, space_after):
        current_token += token
        if not space_after_tok:
            pass
        else:
            yield current_token
            current_token = ""
    yield current_token


def prob_to_surprisal(
        probs: Iterable[float]
        ) -> Iterable[float]:
    for prob in probs:
        yield -math.log(prob)


def add_surprisal(input_file: str, output_file: str,
                  model_dir: str, token_mapper_dir: str,
                  token_col: str = TOKEN_COL,
                  surprisal_col: str = SURPRISAL_COL,
                  word_position_col: str = WORD_POSITION_COL,
                  integration_cost_col: str = INTEGRATION_COST_COL,
                  pos_tag_col: str = POS_TAG_COL,
                  deprel_col: str = DEPREL_COL,
                  fdd_col: str = FDD_COL,
                  fddc_col: str = FDDC_COL,
                  fddw_col: str = FDDW_COL,
                  ldds_col: str = LDDS_COL,
                  ldc_col: str = LDC_COL,
                  device: str = DEVICE,
                  batch_size: int = BATCH_SIZE) -> None:
    df = pd.read_csv(input_file)
    words = df[token_col]
    conllu = parse_list_of_words_with_spacy(words, min_len=0)

    # TODO: load these params from somewhere
    transform = TransformMaskHeadChild(
        keys_for_head={"head"},
        keys_for_child={"child"})

    dataset: CoNLLUDataset = CoNLLUDataset.from_str(
        conllu, transform, max_len=None)

    token_mapper: TokenMapper = TokenMapper.load(token_mapper_dir)
    dataset.map_to_ids(token_mapper)

    trainer = LMTrainer.load(model_dir,
                             batch_size=batch_size,
                             device=device,
                             use_ddp=False,
                             world_size=1)
    surprisal, indices, attention = generate_probs(
        trainer, dataset,
        untokenise=True, surprisal=True)

    pos_tag_list: list[str] = []
    for sentence in dataset.tokens:
        pos_tag_list.extend(get_POS_tags(sentence[2:-1]))

    integration_costs = generate_head_distance(
        dataset, pos_tag_list, untokenise=True)

    fdd = generate_first_dep_distance(dataset, pos_tag_list, untokenise=True,
                                      only_content_words=True)
    fddc = generate_fddc(
        dataset, attention, pos_tag_list, untokenise=True).astype(int)
    fddw = generate_fddw(
        dataset, attention, pos_tag_list, untokenise=True)
    deprels = generate_deprels(dataset, pos_tag_list, untokenise=True)
    ldds = generate_ldds(dataset, pos_tag_list, untokenise=True)
    ldc = generate_ldc(dataset, pos_tag_list, untokenise=True)

    pos_tag_list = generate_POS_tags(
        pos_tag_list, dataset.space_after,
        untokenise=True)

    df[surprisal_col] = surprisal
    df[word_position_col] = indices
    df[integration_cost_col] = integration_costs
    df[fdd_col] = fdd
    df[fddc_col] = fddc
    df[fddw_col] = fddw
    df[ldds_col] = ldds
    df[ldc_col] = ldc
    df[pos_tag_col] = pos_tag_list
    df[deprel_col] = deprels
    df.to_csv(output_file, index=False)


def process_tsv(
        input_file: str, output_file: str,
        model_dir: str, token_mapper_dir: str,
        raw: bool = True,
        token_col: str = TOKEN_COL,
        text_id_col: str = TEXT_ID_COL,
        wnum_col: str = WNUM_COL,
        logfreq_col: str = LOGFREQ_COL,
        surprisal_col: str = SURPRISAL_COL,
        word_position_col: str = WORD_POSITION_COL,
        integration_cost_col: str = INTEGRATION_COST_COL,
        fdd_col: str = FDD_COL,
        fddc_col: str = FDDC_COL,
        fddw_col: str = FDDW_COL,
        ldds_col: str = LDDS_COL,
        ldc_col: str = LDC_COL,
        pos_tag_col: str = POS_TAG_COL,
        deprel_col: str = DEPREL_COL,
        wlen_col: str = WLEN_COL,
        language: str = LANG,
        device: str = DEVICE,
        batch_size: int = BATCH_SIZE
        ) -> None:
    tsv_to_csv(input_file, output_file,
               token_col, text_id_col,
               wnum_col, None if raw else token_mapper_dir)
    add_frequency(output_file, output_file,
                  language, token_col, logfreq_col)
    add_word_length(output_file, output_file,
                    token_col, wlen_col)
    add_surprisal(output_file, output_file, model_dir,
                  token_mapper_dir, token_col,
                  surprisal_col, word_position_col,
                  integration_cost_col, pos_tag_col,
                  deprel_col, fdd_col, fddc_col, fddw_col,
                  ldds_col, ldc_col,
                  device, batch_size)


if __name__ == "__main__":
    # Compute probabilities for natural stories corpus
    # based on a model trianed on Wikitext_processed
    model_name = sys.argv[1]
    in_file = "naturalstories-master/words.tsv"
    out_file = f"RT/data/words_processed_{model_name}.csv"
    mapper = "processed/Wikitext_processed/mapper"  # TODO set to processed
    process_tsv(in_file, out_file, model_name, mapper,
                raw=True)
