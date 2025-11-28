import torch
import torch.distributed as dist

from ..data import (
    get_loader, DataProvider, DataConfig)
from ..train import (
    LMMetric, Result, LMTrainer, TrainConfig)
from ..models import (description_builder, MITransformerConfig)
from ..train.metrics import (
    MetricWriter, metric_writer, sum_and_std_metrics, minimise)
from ..utils.params import dict_info, is_undef
from ..train.hooks import TreePlotHook, AttentionPlotHook
from . import args, argtypes

from tqdm import tqdm
import optuna
import os
import numpy as np
import pandas as pd
from contextlib import contextmanager
from copy import copy
from collections import Counter, defaultdict

from mitransformer.utils.logmaker import (
    getLogger, info)

from typing import (
    Any, Iterable, cast, Iterator)

import time

logger = getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


"""
TODO:
- establish dataset naming and loading by name
- save loading config with dataset
- if no name: save as Wikitext_1, _2 etc.
- when loading look for all datasets in dataset folder
and check if there is a config that fits;
then load this dataset
and if no, search for dataset having this name
if no, search for preset datasets and load new
if not, search on huggingface and parse and load new.
"""


def _load_data_provider(
        arguments: (
            "args.ParserArgs "
            "| args.TestParserArgs | args.CompareParserArgs"),
        memmaped: bool = False,
        model_num: int | None = None
        ) -> DataProvider:
    try:
        if isinstance(arguments, args.TestParserArgs):
            provider = DataProvider.load(
                os.path.join(
                    LMTrainer.model_dir, arguments.model_name,
                    "data_config.json"),
                **arguments.to_dict())
        elif isinstance(arguments, args.CompareParserArgs):
            assert isinstance(model_num, int)
            provider = DataProvider.load(
                os.path.join(
                    LMTrainer.model_dir,
                    arguments.model1_name if model_num == 1
                    else arguments.model2_name,
                    "data_config.json"),
                **arguments.to_dict())
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        config = DataConfig.from_kwargs(
            include_test=False,
            memmapped=memmaped,
            **arguments.to_dict())
        provider = DataProvider(config, arguments.rank)
    return provider


def main_dataprep(arguments: "args.ParserArgs") -> None:
    _load_data_provider(arguments, memmaped=False)


def main_train(
        arguments: "args.TrainParserArgs",
        world_size: int,
        iterate: bool = False,
        data_provider: DataProvider | None = None
        ) -> Iterator[Result]:

    # device: where to execute computation
    if world_size > 1:
        assert arguments.rank is not None, (
            "Rank cannot be None if word_size > 1.")

    train_config = TrainConfig.from_kwargs(
        **arguments.to_dict(),
        world_size=world_size)

    if data_provider is None:
        # TODO: do this entirely in the load dataset method
        # load memmap
        data_provider = _load_data_provider(arguments, memmaped=True)

    # Model
    # 24 heads, one layer approximately matches CBR-RRN
    # TODO: Make this a proper config
    arguments.vocab_size = data_provider.datasets["token_mapper"].vocab_size

    # make proper transformer description
    if (arguments.transformer_description is None
            and arguments.masks_setting == "both"):
        arguments.transformer_description = (
            (('head_current', 'child_current'), 1),
            (('head_next', 'child_next'), 1))
    elif arguments.transformer_description is None:
        if arguments.layer_design is None:
            if arguments.masks_setting == "current":
                arguments.layer_design = (
                    "head_current", "child_current")
            elif arguments.masks_setting == "next":
                arguments.layer_design = (
                    "head_next", "child_next")
        arguments.transformer_description = description_builder(
            arguments.layer_design,
            arguments.use_standard,
            arguments.width,
            arguments.depth,
            arguments.unrestricted_before,
            arguments.unrestricted_after
        )
    n_heads: list[int] = [
        len(layer[0])*layer[1] for layer in arguments.transformer_description]

    # make n_embd divisible by number of heads in each layer
    for n_h_l in n_heads:
        arguments.n_embd = arguments.n_embd // n_h_l * n_h_l

    transformer_config = MITransformerConfig.from_kwargs(
        **arguments.to_dict(),
        use_input_mask=(arguments.dependency_mode == "input"),
        return_proj_states=arguments.distr_mode == "att-n")

    trainer = LMTrainer.new(transformer_config, train_config)
    if isinstance(data_provider, DataProvider):
        data_provider.save(os.path.join(trainer.model_dir,
                                        train_config.model_name,
                                        "data_config.json"))
    metrics: Result
    # Training setting
    if not iterate:
        metrics = trainer.train(**data_provider.datasets)
        if arguments.rank is None or arguments.rank == 0:
            generated = []
            for _ in range(20):
                generated.append(
                    trainer.generate(data_provider.datasets["token_mapper"]))
            info(
                arguments.rank, logger,
                f"Generated model output sample: {generated}")
        yield metrics  # type: ignore

    # Hyperopt setting
    else:
        for metrics in trainer.train_iter(**data_provider.datasets):
            yield metrics

    del trainer
    del data_provider


MeanStdDict = dict[str, tuple[float, float]]


def main_train_multiple(
        arguments: "args.TrainParserArgs",
        world_size: int,
        data_provider: DataProvider | None = None
        ) -> (
            tuple[MeanStdDict, MeanStdDict]
            | tuple[MeanStdDict, MeanStdDict, MeanStdDict]):
    start = time.time()
    """Calculates the mean and standard deviation of several
    runs."""
    # How to keep track of results? We are logging them
    # but shouldn't we also forward them to tensorboard?
    # but we only have the final results or should we compute
    # the mean over the models for each step?
    if data_provider is None:
        data_provider = _load_data_provider(arguments, memmaped=True)

    assert arguments.n_runs != 0, "--n_runs cannot be 0"

    metrics_list: (
        list[tuple[LMMetric, LMMetric]]
        | list[tuple[LMMetric, LMMetric, LMMetric]]) = []
    for n_run in tqdm(range(arguments.n_runs), desc="Runs"):
        run_arguments = copy(arguments)
        run_arguments.model_name = f"{arguments.name}_{n_run}"
        run_arguments.seed = arguments.seed + n_run  # offset seed
        args.args_logic(run_arguments)  # also sets seed

        metrics_list.append(
            tuple(next(main_train(
                run_arguments,
                world_size,
                iterate=False,
                data_provider=data_provider)).values()))  # type: ignore

    means_and_stds = tuple(
        sum_and_std_metrics(seq)
        for seq in zip(*metrics_list))

    info(
        arguments.rank, logger,
        f"Performed {arguments.n_runs} training runs.")
    info(
        arguments.rank, logger,
        f"Final mean and std train: {dict_info(means_and_stds[0])}")
    info(
        arguments.rank, logger,
        f"Final mean and std dev: {dict_info(means_and_stds[1])}")
    if len(means_and_stds) > 2:
        info(
            arguments.rank, logger,
            f"Final mean and std test: {dict_info(means_and_stds[2])}")

    end = time.time()
    info(arguments.rank, logger, f"Took {end - start} seconds!")
    return means_and_stds  # type: ignore


def main_test(
        arguments: "args.TestParserArgs",
        world_size: int,
        data_provider: DataProvider | None = None
        ) -> tuple[LMMetric, LMMetric, LMMetric]:
    """Calculates the mean and standard deviation of several
    runs."""

    # device: where to execute computation
    if world_size > 1:
        assert arguments.rank is not None, (
            "Rank cannot be None if word_size > 1.")

    if is_undef(arguments.dependency_mode):
        trainer = LMTrainer.load(
            world_size=world_size,
            **arguments.to_dict())
    else:
        trainer = LMTrainer.load(
            world_size=world_size,
            use_input_mask=arguments.dependency_mode == "input",
            **arguments.to_dict())

    arguments.update_from_kwargs(**trainer.config.to_dict())

    if data_provider is None:
        data_provider = _load_data_provider(
            arguments,
            memmaped=True)

    # TODO: Hooks do not save dataset name or number.
    # Idea: add counter to trainer that counts number of received
    # datasets and give this number to hook

    model_dir = os.path.join(trainer.model_dir, arguments.model_name)

    if arguments.att_plot:
        trainer.add_hook(AttentionPlotHook(
            os.path.join(model_dir, "hooks", "att_plots")
        ))

    if arguments.tree_plot:
        trainer.add_hook(TreePlotHook(
            os.path.join(model_dir, "hooks", "tree_plots"),
            masks_setting=arguments.masks_setting
        ))

    # Training setting
    metrics = trainer.test(**data_provider.datasets)

    generated = []
    for _ in range(20):
        generated.append(
            trainer.generate(data_provider.datasets["token_mapper"]))
    info(
        arguments.rank, logger,
        f"Generated model output sample: {generated}")

    del trainer
    del data_provider

    return metrics  # type: ignore


def main_compare(
        arguments: "args.CompareParserArgs",
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
        data_provider = _load_data_provider(
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
        data_provider = _load_data_provider(
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


USE_LOG = {"learning_rate"}


def hyperopt_arguments_sampler(
        name: str,
        arg: argtypes.T | list[argtypes.T] | tuple[argtypes.T, argtypes.T],
        trial
        ) -> argtypes.T:
    if isinstance(arg, list):
        assert len(arg) > 0, f"Provided an empty selection for {name}!"
        return trial.suggest_categorical(name, arg)
    elif (
            isinstance(arg, tuple)
            and len(arg) == 2
            and isinstance(arg[0], (int, float))):
        if isinstance(arg[0], float) and isinstance(arg[1], float):
            return trial.suggest_float(
                name, arg[0], arg[1],
                log=name in USE_LOG)
        elif isinstance(arg[0], int) and isinstance(arg[1], int):
            return trial.suggest_int(
                name, arg[0], arg[1],
                log=name in USE_LOG)
        else:
            raise Exception(
                f"Range {arg} for arg {name} inconsistently typed!")
    else:
        return cast(argtypes.T, arg)


class Objective:
    def __init__(
            self, n_devices: int,
            arguments: "args.HyperoptParserArgs",
            writer: MetricWriter,
            pg):
        self.n_devices = n_devices
        self.arguments = arguments
        self.writer = writer
        self.pg = pg

        self.data_provider = None
        self.datasets = None

        # TODO: do not use try but check if any of the relevant arguments are
        # Hyperopt spaces
        try:
            self.data_provider = _load_data_provider(arguments, memmaped=True)
            # Since pin_memory=True, persistent_workers=True lead
            # to too many files
            # error when creating a lot of dataloaders, we need to construct
            # dataloaders here
            # Remove this if https://github.com/pytorch/pytorch/issues/91252
            # is resolved
            self.data_provider.datasets["train"] = get_loader(  # type: ignore
                    self.data_provider.datasets["train"],  # type: ignore
                    batch_size=self.arguments.batch_size,
                    bucket=False,
                    shuffle=True, droplast=True,
                    world_size=self.n_devices,
                    rank=self.arguments.rank,
                    n_workers=self.arguments.n_workers)
            self.data_provider.datasets["eval"] = get_loader(  # type: ignore
                    self.data_provider.datasets["eval"],  # type: ignore
                    batch_size=self.arguments.batch_size,
                    bucket=False,
                    shuffle=False, droplast=False,
                    world_size=self.n_devices,
                    rank=self.arguments.rank,
                    n_workers=self.arguments.n_workers)
        except TypeError:
            self.data_provider = None

    def __call__(self, trial) -> float:
        if self.n_devices > 1:
            trial = optuna.integration.TorchDistributedTrial(
                trial, self.pg)  # type: ignore

        arguments = args.TrainParserArgs.from_kwargs(**{
            name: hyperopt_arguments_sampler(name, arg, trial) for
            name, arg in self.arguments.to_dict().items()},
            model_name=f"{self.arguments.name}_{trial.number}",
            n_runs=1)
        arguments.seed = arguments.seed + trial.number
        args.args_logic(arguments)

        train_iterator = main_train(
            arguments, self.n_devices,
            iterate=True,
            data_provider=self.data_provider)
        assert train_iterator is not None

        should_prune = False
        metrics = None
        for step, metrics in enumerate(train_iterator, start=1):
            # Handle pruning based on the intermediate value.
            opt_metric = getattr(
                metrics["eval"], self.arguments.optimise.lower())
            if isinstance(opt_metric, pd.DataFrame):
                opt_metric = float(opt_metric.to_numpy().sum())
            trial.report(
                opt_metric,
                step)

            if trial.should_prune():
                should_prune = True
                break

        assert metrics is not None, (
            "eval_interval is larger than total number of steps")
        if self.writer is not None:
            self.writer.add_params(
                arguments.to_dict(),
                metrics["eval"],
                run_name=str(trial.number),
                global_step=arguments.eval_interval*step)

        if should_prune:
            raise optuna.exceptions.TrialPruned()
        # trial.set_user_attr("metric_dicts", metric_dicts)

        opt_metric = getattr(metrics["eval"], self.arguments.optimise.lower())
        if isinstance(opt_metric, pd.DataFrame):
            opt_metric = opt_metric.to_numpy().sum()
        loss: float = float(opt_metric)
        return loss


def main_hyperopt(
        arguments: "args.HyperoptParserArgs",
        world_size: int) -> None:
    direction = (
        "minimize" if minimise[arguments.optimise.lower().split(":")[0]]
        else "maximize")

    ld = os.path.join("./runs", f"{arguments.name}_hyperopt")
    with new_pg(world_size, "gloo") as pg, metric_writer(log_dir=ld) as writer:
        objective: Objective = Objective(world_size, arguments, writer, pg)
        if arguments.rank == 0 or arguments.rank is None:
            study = optuna.create_study(
                study_name=arguments.name,
                direction=direction,
                sampler=optuna.samplers.RandomSampler(
                    seed=arguments.seed),  # TODO: normal sampler
                pruner=optuna.pruners.MedianPruner(
                    n_warmup_steps=arguments.n_warmup_steps,
                    n_startup_trials=arguments.n_startup_trials))
            study.optimize(
                objective, n_trials=arguments.n_trials)

        else:
            for _ in range(arguments.n_trials):
                try:
                    objective(None)
                except optuna.TrialPruned:
                    pass

    if arguments.rank == 0 or arguments.rank is None:
        assert study is not None
        pruned_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        info(
            arguments.rank, logger,
            (
                f"Pruned {len(pruned_trials)}, "
                f"completed {len(complete_trials)} trials"))

        info(
            arguments.rank, logger,
            f"Best trial: {study.best_trial.number}\n"
            f"with results: {study.best_value}\n"
            f"with params: {study.best_params}")

    return None


def options(seq_of_items: list[tuple[str, Iterable[Any]]]
            ) -> Iterable[dict[str, Any]]:
    name, values = seq_of_items[0]
    for value in values:
        if len(seq_of_items[1:]) == 0:
            yield {name: value}
        else:
            for d in options(seq_of_items[1:]):
                d[name] = value
                yield d


def setup_group(world_size, backend: str = "gloo") -> dist.ProcessGroup | None:
    if world_size > 1:
        info(
            None, logger,
            f"Initialising process group with backend {backend}")
        pg = dist.new_group(backend=backend)
        return pg
    else:
        return None


def clean_group(world_size, pg) -> None:
    if world_size > 1:
        dist.destroy_process_group(pg)


@contextmanager
def new_pg(
        world_size, backend: str = "gloo") -> Iterator[
            dist.ProcessGroup
            | None]:
    try:
        pg = setup_group(world_size, backend)
        yield pg
    finally:
        clean_group(world_size, pg)


def setup_ddp(rank, world_size, backend: str = "nccl") -> bool:
    if world_size > 1:
        info(
            None, logger,
            (
                f"Initialising process group with backend {backend}, "
                f"world size {world_size} and rank {rank}."))
        dist.init_process_group(backend, world_size=world_size, rank=rank)
        torch.cuda.set_device(torch.distributed.get_rank())
        return True
    else:
        return False


def clean_ddp(world_size, pg=None) -> None:
    """if None then destroy dist.group.WORLD"""
    if world_size > 1:
        pg = dist.group.WORLD if pg is None else pg
        clean_group(world_size, pg)


@contextmanager
def ddp(rank: int | None, world_size: int) -> Iterator[bool]:
    try:
        yield setup_ddp(rank, world_size)
    finally:
        clean_ddp(world_size)


def main(arguments: "args.ParserArgs") -> None:
    if arguments.mode == "dataprep":
        main_dataprep(arguments)
    else:
        try:
            n_devices = (
                int(os.environ["WORLD_SIZE"]) if arguments.use_ddp else 1)
        except ValueError:
            n_devices = torch.cuda.device_count() if arguments.use_ddp else 1
        assert not ((n_devices == 1 or not arguments.use_ddp) and
                    (arguments.rank is not None and arguments.rank > 0)), (
            "Rank cannot be larger than 0 if only having one device"
            "/not using ddp. "
            f"Received --local-rank {arguments.rank} "
            f"--use_ddp {arguments.use_ddp} "
            f"and number of recognised CUDA devices is {n_devices}.")
        info(arguments.rank, logger, f"Running on {n_devices} devices.")
        with ddp(arguments.rank, n_devices) as ddp_status:
            info(arguments.rank, logger, f"Using DDP: {ddp_status}")
            mode = arguments.mode
            match mode:
                case "train":
                    assert isinstance(arguments, args.TrainParserArgs)
                    info(arguments.rank, logger, "Launching model training.")
                    main_train_multiple(arguments, n_devices)
                case "test":
                    assert isinstance(arguments, args.TestParserArgs)
                    info(arguments.rank, logger, "Launching model testing.")
                    main_test(arguments, n_devices)
                case "hyperopt":
                    assert isinstance(arguments, args.HyperoptParserArgs)
                    info(
                        arguments.rank, logger,
                        "Launching hyperparameter tuning.")
                    main_hyperopt(arguments, n_devices)
                case _:
                    assert isinstance(arguments, args.CompareParserArgs)
                    info(arguments.rank, logger, "Launching model comparison.")
                    main_compare(arguments, n_devices)
