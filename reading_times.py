"""Snippets taken from
https://github.com/weijiexu-charlie/
Linearity-of-surprisal-on-RT/blob/main/Preparing%20Corpora/get_frequency.py"""


import pandas as pd
import wordfreq     # type: ignore
from tokeniser import TokenMapper
from data import (CoNLLUDataset, DataLoader,
                  get_transform_mask_head_child, get_loader)
from model import MITransformerLM, MITransformer
from parse import parse_list_of_words_with_spacy

from natural_stories import load_natural_stories

import torch

from typing import Iterable

import math

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
SURPRISAL_COL = "surprisal"
WLEN_COL = "WLEN"

DEVICE = "cpu"
BATCH_SIZE = 64


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
               wnum_col: str = WNUM_COL
               ) -> None:
    tokens, text_ids, wnums = load_natural_stories(input_file)
    # making lowercase makes no difference

    df = pd.DataFrame({token_col: tokens,
                       text_id_col: text_ids,
                       wnum_col: wnums})

    df.to_csv(output_file, index=False)


def generate_probs(
        model: MITransformerLM,
        dataloader,
        padding_label: int,
        untokenise: bool = False,
        surprisal: bool = False) -> torch.Tensor:
    """WARNING: untested"""

    pred_prob_list: list[torch.Tensor] = []
    detok_list = []

    space_after_list = []
    for batch in dataloader:

        logits, _ = model(**batch)
        probs = torch.softmax(logits, dim=-1)

        token_mapper: TokenMapper = TokenMapper.load("./tokmap.pickle")

        space_after_list.extend(batch["space_after"])
        for probs_sen, label_ids_sen in zip(
                probs, batch["label_ids"]):

            select_entries = label_ids_sen != padding_label
            unpadded_labels = label_ids_sen[select_entries][1:-1]

            detok = token_mapper.decode([unpadded_labels.tolist()])[0]
            detok_list.extend(detok)

            probs_sen = probs_sen[1:-1]
            unpadded_indices = torch.arange(
                label_ids_sen.shape[0])[select_entries][1:-1] - 1
            pred_prob = probs_sen[unpadded_indices,
                                  unpadded_labels.long()]
            pred_prob_list.append(pred_prob)

    probs = torch.cat(
            [p for p in pred_prob_list])

    if untokenise:
        space_after = torch.cat(
            [torch.tensor(e) for e in space_after_list]).tolist()
        probs = torch.tensor(list(untokenise_probs(
            probs.tolist(),
            space_after)))

    if surprisal:
        probs = -torch.log(probs)

    return probs


def untokenise_probs(
        probs: Iterable[float],
        space_after: Iterable[bool],
        ) -> Iterable[float]:
    current_prob: float = 1
    for prob_tok, space_after_tok in zip(probs, space_after):
        current_prob *= prob_tok
        if not space_after_tok:
            pass
        else:
            yield current_prob
            current_prob = 1
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


def add_surprisal(input_file: str, output_file,
                  model_dir: str, token_mapper_dir: str,
                  token_col: str = TOKEN_COL,
                  surprisal_col: str = SURPRISAL_COL,
                  device: str = DEVICE,
                  batch_size: int = BATCH_SIZE) -> None:
    # TODO: generate preditions with model
    # TODO: save
    df = pd.read_csv(input_file)
    words = df[token_col]
    conllu = parse_list_of_words_with_spacy(words, min_len=0)

    transform = get_transform_mask_head_child(
        keys_for_head={"head"},
        keys_for_child={"child"},
        triangulate=True)

    dataset: CoNLLUDataset = CoNLLUDataset.from_str(
        conllu, transform, max_len=None)

    token_mapper: TokenMapper = TokenMapper.load(token_mapper_dir)
    dataset.map_to_ids(token_mapper)
    dataloader: DataLoader = get_loader(dataset, batch_size=batch_size,
                                        bucket=False, device=device,
                                        shuffle=False, droplast=False)

    state_dict, config = torch.load(model_dir).values()
    model: MITransformerLM = MITransformerLM(MITransformer(**config))
    model.load_state_dict(state_dict)

    surprisal = generate_probs(
        model, dataloader, padding_label=dataset.keys_for_padding["label_ids"],
        untokenise=True, surprisal=True)

    df[surprisal_col] = surprisal
    df.to_csv(output_file, index=False)


def process_tsv(
        input_file: str, output_file: str,
        model_dir: str, token_mapper_dir: str,
        token_col: str = TOKEN_COL,
        text_id_col: str = TEXT_ID_COL,
        wnum_col: str = WNUM_COL,
        logfreq_col: str = LOGFREQ_COL,
        surprisal_col: str = SURPRISAL_COL,
        wlen_col: str = WLEN_COL,
        language: str = LANG,
        device: str = DEVICE,
        batch_size: int = BATCH_SIZE
        ) -> None:
    tsv_to_csv(input_file, output_file,
               token_col, text_id_col,
               wnum_col)
    add_frequency(output_file, output_file,
                  language, token_col, logfreq_col)
    add_word_length(output_file, output_file,
                    token_col, wlen_col)
    add_surprisal(output_file, output_file, model_dir,
                  token_mapper_dir, token_col,
                  surprisal_col, device, batch_size)
