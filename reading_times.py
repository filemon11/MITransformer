"""Snippets taken from
https://github.com/weijiexu-charlie/
Linearity-of-surprisal-on-RT/blob/main/Preparing%20Corpora/get_frequency.py"""


import pandas as pd
import wordfreq     # type: ignore
import nltk  # type: ignore
from tokeniser import TokenMapper
from data import CoNLLUDataset, TransformMaskHeadChild
from trainer import LMTrainer
from parse import parse_list_of_words_with_spacy

from natural_stories import load_natural_stories

import torch
import numpy as np

from typing import Iterable, Literal, Callable
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
INTEGRATION_COST_COL = "incost"
POS_TAG_COL = "POS"
WLEN_COL = "WLEN"

DEVICE = "cpu"
BATCH_SIZE = 64

CONTENT_POS = {"FW", "MD", "NN", "NNS", "NNP",
               "NNPS", "VB", "VBD", "VBG", "VBN",
               "VBP", "VBZ"}  # , "JJ", "JJR", "JJS"}


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
        surprisal: bool = False) -> tuple[torch.Tensor, npt.NDArray[np.int_]]:
    """WARNING: untested"""

    pred_probs, _ = trainer.predict(
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

    return probs, indices


def get_POS_tags(sentence: list[str]) -> list[str]:
    tagged = nltk.tag.pos_tag(sentence)
    return [tag for _, tag in tagged]


def generate_POS_tags(
        dataset: CoNLLUDataset,
        untokenise: bool = False
        ) -> list[str]:
    pos_tag_list = []
    for sentence in dataset.tokens:
        pos_tags = get_POS_tags(sentence[2:-1])
        pos_tag_list.extend(pos_tags)

    if untokenise:
        assert dataset.space_after is not None
        space_after = np.concat(dataset.space_after)

        new_pos_tag_list = []
        current_pos = pos_tag_list[0]
        for i, (_, space_after_tok) in enumerate(
                zip(pos_tag_list, space_after)):
            if not space_after_tok:
                pass
            else:
                new_pos_tag_list.append(current_pos)
                current_pos = pos_tag_list[i+1]
        new_pos_tag_list.append(current_pos)
        pos_tag_list = new_pos_tag_list

    return pos_tag_list


def get_content_word_mask(
        sentence: list[str],
        not_to_mask: set[str] = CONTENT_POS) -> npt.NDArray[np.bool]:
    pos_tags = get_POS_tags(sentence)
    print(pos_tags)
    mask = np.array([tag in not_to_mask for tag in pos_tags], dtype=bool)
    mask[0:2] = False
    print(mask)
    return mask


def get_content_word_cost(
        content_word_mask: npt.NDArray[np.bool],
        mask_gov: npt.NDArray[np.bool],
        mask_dep: npt.NDArray[np.bool]
        ) -> npt.NDArray[np.int_]:
    print(content_word_mask)

    mask = np.logical_or(mask_gov, mask_dep)

    # do not take into account connections to function words
    mask[:, ~content_word_mask] = 0
    mask[0, :] = 0

    print(mask.astype(int))
    first_connection: npt.NDArray[np.int_] = np.argmax(np.tril(mask), axis=1)
    print(first_connection)

    upper_triang = np.triu(
        np.ones((content_word_mask.shape[0], content_word_mask.shape[0])), 1)
    cw_cumul: np.ndarray = content_word_mask @ upper_triang

    cw_before_first = cw_cumul[first_connection]  # @ mask_gov.T
    cw_intermediate = cw_cumul - cw_before_first
    print("first connection:\n", first_connection)
    print("cw_cumul:\n", cw_cumul)
    print("cw_before_first:\n", cw_before_first)
    # assume no crossing arcs
    # left_cw_child_nums = content_word_mask @ mask_dep.T
    # cw_intermediate -= left_cw_child_nums
    print("intermediate:\n", cw_intermediate)
    cw_intermediate[first_connection == 0] = 0  # if dummy or root

    # cw_intermediate[mask_gov[:, 0]] = 1  # for dummy set 1
    # cw_intermediate[mask_gov[:, 1]] = 1  # for root set 1
    print("intermediate:\n", cw_intermediate)
    print("content words:\n", content_word_mask.astype(int))
    cost = cw_intermediate + content_word_mask.astype(int)  # new discourse referents
    cost[~content_word_mask] = 0  # integration and instantiation cost for function words is 0
    return cost


def generate_integration_Gibson(
        dataset: CoNLLUDataset,
        untokenise: bool = False,
        dummy_used: bool = True) -> npt.NDArray[np.int_]:
    """WARNING: untested"""

    costs_list: list[npt.NDArray[np.int_]] = []
    heads_list = dataset.heads
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
        print(heads_list[i][:-1])
        print(costs)
        print(dataset.tokens[i][:-1])
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


def generate_integration_costs(
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

        cost_array = np.array(
            list(untokenise_probs(
                cost_array.tolist(),
                space_after,
                mode="add")))

    return cost_array


def untokenise_probs(
        probs: Iterable[float],
        space_after: Iterable[bool],
        mode: Literal["mult", "add"] = "mult"
        ) -> Iterable[float]:
    base: float = 0
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
                  dummy_used: bool = True,
                  token_col: str = TOKEN_COL,
                  surprisal_col: str = SURPRISAL_COL,
                  word_position_col: str = WORD_POSITION_COL,
                  integration_cost_col: str = INTEGRATION_COST_COL,
                  pos_tag_col: str = POS_TAG_COL,
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
    surprisal, indices = generate_probs(
        trainer, dataset,
        untokenise=True, surprisal=True)

    integration_costs = generate_integration_costs(
        dataset, untokenise=True, dummy_used=dummy_used)

    df[surprisal_col] = surprisal
    df[word_position_col] = indices
    df[integration_cost_col] = integration_costs
    df[pos_tag_col] = generate_POS_tags(dataset, untokenise=True)
    df.to_csv(output_file, index=False)


def process_tsv(
        input_file: str, output_file: str,
        model_dir: str, token_mapper_dir: str,
        raw: bool = True,
        dummy_used: bool = True,
        token_col: str = TOKEN_COL,
        text_id_col: str = TEXT_ID_COL,
        wnum_col: str = WNUM_COL,
        logfreq_col: str = LOGFREQ_COL,
        surprisal_col: str = SURPRISAL_COL,
        word_position_col: str = WORD_POSITION_COL,
        integration_cost_col: str = INTEGRATION_COST_COL,
        pos_tag_col: str = POS_TAG_COL,
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
                  token_mapper_dir, dummy_used, token_col,
                  surprisal_col, word_position_col,
                  integration_cost_col, pos_tag_col,
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
