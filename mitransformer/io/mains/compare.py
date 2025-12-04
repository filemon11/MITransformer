import torch

from . import functions

from ...data import DataProvider
from ...train import LMTrainer
from .. import parsing

import numpy as np
from collections import Counter, defaultdict

from mitransformer.utils.logmaker import (
    getLogger, info)

logger = getLogger(__name__)


def main_compare(
        arguments: "parsing.CompareParserArgs",
        world_size: int,
        data_provider: DataProvider | None = None
        ) -> None:
    """Calculates the mean and standard deviation of several
    runs."""
    data_provider_given: bool = data_provider is not None
    window = 20
    highest_num = 1000

    # device: where to execute computation
    if world_size > 1:
        assert arguments.rank is not None, (
            "Rank cannot be None if word_size > 1.")

    extra_arguments = arguments.to_dict()

    # model1
    trainer = LMTrainer.load(
        model_name=arguments.model1_name,
        world_size=world_size,
        **extra_arguments)

    arguments.update_from_kwargs(**trainer.config.to_dict())

    if not data_provider_given:
        data_provider = functions._load_data_provider(
            arguments,
            memmaped=True,
            model_num=1)

    assert data_provider is not None
    assert "eval" in data_provider.datasets
    dataset = data_provider.datasets["eval"]
    token_mapper = data_provider.datasets["token_mapper"]

    logprobs1, _, _ = trainer.predict(
        dataset,
        make_prob=True,
        only_true=True)

    # model2
    del trainer
    trainer = LMTrainer.load(
        model_name=arguments.model2_name,
        world_size=world_size,
        **extra_arguments)

    if not data_provider_given:
        data_provider = functions._load_data_provider(
            arguments,
            memmaped=True,
            model_num=2)

    logprobs2, _, _ = trainer.predict(
        dataset,
        make_prob=True,
        only_true=True)
    del trainer

    diffs: list[tuple[float, int, int]] = []  # diff, sen, pos
    for sen_num, (probs1, probs2) in enumerate(zip(logprobs1, logprobs2)):
        diff_tensor = (probs2 - probs1).tolist()
        sen_nums = [sen_num] * probs1.shape[0]
        positions = list(range(0, probs1.shape[0]))
        diffs.extend(list(zip(diff_tensor, sen_nums, positions)))

    highest = list(sorted(
        diffs,
        key=lambda x: abs(x[0]),
        reverse=True))[:highest_num]

    assert dataset.id_hl is not None
    highest_tokens = [token_mapper.id2word[
        dataset.id_hl[x[1]][0][x[2]]] for x in highest]
    highest_tokens_count = Counter(highest_tokens)

    info(
        arguments.rank, logger,
        f"{highest_num} tokens with the highest "
        f"difference: {highest_tokens_count}")

    info(arguments.rank, logger, "Tokens with window:")

    for c_diff, c_sen_num, c_pos in highest:
        tokens = token_mapper.decode([dataset.id_hl[c_sen_num][0]])[0]
        left = max(0, c_pos-window)
        right = min(len(tokens), c_pos+window)
        info(
            arguments.rank, logger,
            (
                f"diff={round(c_diff, 2)}: {' '.join(tokens[left:c_pos])} "
                f"[{tokens[c_pos]}] {' '.join(tokens[c_pos+1:right])}"))

    token_to_diffs: defaultdict[str, list[tuple[float, int, int]]]
    token_to_diffs = defaultdict(list)
    for token, diff in zip(highest_tokens, highest):
        token_to_diffs[token].append(diff)

    # Concordances
    info(arguments.rank, logger, "Concordances:")
    for token, _ in sorted(
            highest_tokens_count.items(), key=lambda x: x[1], reverse=True):
        info(arguments.rank, logger, f"\nToken: {token}\n")
        for c_diff, c_sen_num, c_pos in token_to_diffs[token]:
            tokens = token_mapper.decode([dataset.id_hl[c_sen_num][0]])[0]
            left = max(0, c_pos-window)
            right = min(len(tokens), c_pos+window)
            info(
                arguments.rank, logger,
                (
                    f"diff={round(c_diff, 2)}: "
                    f"{' '.join(tokens[left:c_pos])[-80:]:>80} "
                    f"[{tokens[c_pos]}] "
                    f"{' '.join(tokens[c_pos+1:right])[:81]}"))

    # Mean differences
    total_tokens = [token_mapper.id2word[
        dataset.id_hl[x[1]][0][x[2]]] for x in diffs]
    total_tokens_count = Counter(total_tokens)

    perplexity_diffs: list[tuple[float, int, int]] = []  # diff, sen, pos
    ppl1: list[float] = []
    ppl2: list[float] = []
    for sen_num, (probs1, probs2) in enumerate(zip(logprobs1, logprobs2)):
        ppl1.extend((-torch.log(probs1)).tolist())
        ppl2.extend((-torch.log(probs2)).tolist())
        diff_tensor = (-torch.log(probs2) - -torch.log(probs1)).tolist()
        sen_nums = [sen_num] * probs1.shape[0]
        positions = list(range(0, probs1.shape[0]))
        perplexity_diffs.extend(list(zip(diff_tensor, sen_nums, positions)))

    summed_differences: defaultdict[str, float] = defaultdict(float)
    for token, (c_diff, _, _) in zip(total_tokens, diffs):
        summed_differences[token] += c_diff

    mean_differences = {token: diff_sum/total_tokens_count[token]
                        for token, diff_sum in summed_differences.items()}

    info(
        arguments.rank, logger,
        "\nProbability diffs ordered by improvement contribution:\n"
        + "\n".join(f"'{tup[0]}': {tup[1]}" for tup in sorted(
            mean_differences.items(),
            key=lambda x: x[1]*total_tokens_count[x[0]],
            reverse=True)))

    info(
        arguments.rank, logger,
        "\nProbability diffs ordered by worsening contribution:\n"
        + "\n".join(f"'{tup[0]}': {tup[1]}" for tup in sorted(
            mean_differences.items(),
            key=lambda x: x[1]*total_tokens_count[x[0]],
            reverse=False)))

    info(
        arguments.rank, logger,
        f"Perplexity 1: {np.exp(np.mean(ppl1))}")

    info(
        arguments.rank, logger,
        f"Perplexity 2: {np.exp(np.mean(ppl2))}")

    info(
        arguments.rank, logger,
        "Change of perplexity in total: "
        f"{np.exp(np.mean(ppl2)) - np.exp(np.mean(ppl1))}")
