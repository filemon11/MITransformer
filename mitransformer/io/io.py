import torch
import torch.distributed as dist

from ..data import (
    get_loader, MasksSetting, DataProvider, DataConfig)
from ..train import (
    Metric, Result, LMTrainer, TrainConfig)
from ..models import (
    TransformerDescription, description_builder, MITransformerConfig)
from ..train.metrics import (
    MetricWriter, metric_writer, sum_and_std_metrics, minimise)
from ..utils.params import Params, dict_info, Undefined, is_undef
from ..train.hooks import TreePlotHook, AttentionPlotHook

from tqdm import tqdm
import optuna
import random
import os
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from copy import copy
from collections import Counter, defaultdict

from mitransformer.utils.logmaker import (
    getLogger, info)

from typing import (
    Literal, Any, Iterable, cast, Iterator, TypeVar, Generic,
    Sequence, Callable)

import time

logger = getLogger(__name__)
optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


def seed_everything(seed: int):
    """There might be nondeterministic torch algorithms.
    We're not making them deterministic here."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # according to
    # https://pytorch.org/docs/stable/data.html#data-loading-randomness
    # each dataloader worker will have its PyTorch seed set to
    # base_seed + worker_id. Thus, with the same number
    # of workers, the process is deterministic


T = TypeVar("T")


class StrToLiteral(Generic[T]):
    def __init__(self, *selection: T):
        self.selection: tuple[T, ...] = selection

    def __call__(self, string: str) -> T:
        if string in self.selection:
            return cast(T, string)
        else:
            raise Exception(
                f"Argument value not allowed."
                f"Given: {string}, allowed: {self.selection}")


def str_to_bool(string: str) -> bool:
    try:
        if (string.lower() == "true"
                or int(string) == 1):
            return True
    except ValueError:
        pass
    try:
        if (string.lower() == "false"
                or int(string) == 0):
            return False
    except ValueError:
        pass
    raise Exception((
        f"argument value {string} cannot"
        "be parsed as a string!"))


class OptNone(Generic[T]):
    def __init__(self, type: Callable[[Any], T]):
        self.type = type

    def __call__(self, value: str) -> None | T:
        if value.lower() == 'none':
            return None
        try:
            return self.type(value)  # type: ignore
        except TypeError:
            raise Exception(
                f"Constructor for {self.type} does not accept an argument")


class HyperoptSpace(Generic[T]):
    def __init__(
            self, type: Callable[[Any], T],
            choices: Sequence[T] | None = None):
        self.type = type
        self.choices = choices

    def __call__(self, value: str) -> tuple[T, T] | list[T] | T:
        # try to split via :
        split = value.split(":")
        if len(split) == 2:
            try:
                h_range = (
                    self.type(split[0]),
                    self.type(split[1]))  # type: ignore
                assert (isinstance(h_range[0], (int, float, complex))
                        and not isinstance(h_range[0], bool)), (
                        f"Non-numeric range detected: {h_range}")
                return h_range
            except TypeError:
                raise Exception(
                    f"Constructor for {self.type} does not accept an argument")

        assert len(split) < 2, (
            f"Range must have one starting and one end point. Given: {value}")

        split = value.split(";")
        if len(split) == 1:
            try:
                return self.type(value)  # type: ignore
            except TypeError:
                raise Exception(
                    f"Constructor for {self.type} does not accept an argument")

        try:
            options = [self.type(v) for v in split]  # type: ignore
            if self.choices is not None:
                for o in options:
                    assert o in self.choices, (
                        f"{o} must be one of {self.choices}")
            return options
        except TypeError:
            raise Exception(
                f"Constructor for {self.type} does not accept an argument")


def make_device_str(string: str) -> str:
    try:
        return "cuda:" + str(int(string))
    except ValueError:
        return string


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
        args: "ParserArgs | TestParserArgs | CompareParserArgs",
        memmaped: bool = False,
        model_num: int | None = None
        ) -> DataProvider:
    try:
        if isinstance(args, TestParserArgs):
            provider = DataProvider.load(
                os.path.join(
                    LMTrainer.model_dir, args.model_name, "data_config.json"),
                **args.to_dict())
        elif isinstance(args, CompareParserArgs):
            assert isinstance(model_num, int)
            provider = DataProvider.load(
                os.path.join(
                    LMTrainer.model_dir,
                    args.model1_name if model_num == 1 else args.model2_name,
                    "data_config.json"),
                **args.to_dict())
        else:
            raise FileNotFoundError
    except FileNotFoundError:
        config = DataConfig.from_kwargs(
            include_test=False,
            memmapped=memmaped,
            **args.to_dict())
        provider = DataProvider(config, args.rank)
    return provider


def main_dataprep(args: "ParserArgs") -> None:
    _load_data_provider(args, memmaped=False)


def main_train(
        args: "TrainParserArgs",
        world_size: int,
        iterate: bool = False,
        data_provider: DataProvider | None = None
        ) -> Iterator[Result]:

    # device: where to execute computation
    if world_size > 1:
        assert args.rank is not None, "Rank cannot be None if word_size > 1."

    train_config = TrainConfig.from_kwargs(
        **args.to_dict(),
        world_size=world_size)

    if data_provider is None:
        # TODO: do this entirely in the load dataset method
        # load memmap
        data_provider = _load_data_provider(args, memmaped=True)

    # Model
    # 24 heads, one layer approximately matches CBR-RRN
    # TODO: Make this a proper config
    args.vocab_size = data_provider.datasets["token_mapper"].vocab_size

    # make proper transformer description
    if args.transformer_description is None:
        if args.layer_design is None:
            if args.masks_setting == "current":
                args.layer_design = (
                    "head_current", "child_current")
            elif args.masks_setting == "next":
                args.layer_design = (
                    "head_next", "child_next")
            else:
                raise Exception(
                    "Define transformer_description "
                    "explicitly for masks_setting='both'.")
        args.transformer_description = description_builder(
            args.layer_design,
            args.use_standard,
            args.width,
            args.depth,
            args.unrestricted_before,
            args.unrestricted_after
        )
        n_heads = len(args.layer_design) * args.width
    else:
        n_heads = sum(
            [len(layer[0])*layer[1] for layer in args.transformer_description])
    # make n_embd divisible by number of heads
    args.n_embd = args.n_embd // n_heads * n_heads

    transformer_config = MITransformerConfig.from_kwargs(
        **args.to_dict(),
        use_input_mask=(args.dependency_mode == "input"))

    trainer = LMTrainer.new(transformer_config, train_config)
    if isinstance(data_provider, DataProvider):
        data_provider.save(os.path.join(trainer.model_dir,
                                        train_config.model_name,
                                        "data_config.json"))
    metrics: Result
    # Training setting
    if not iterate:
        metrics = trainer.train(**data_provider.datasets)
        if args.rank is None or args.rank == 0:
            generated = []
            for _ in range(20):
                generated.append(
                    trainer.generate(data_provider.datasets["token_mapper"]))
            info(
                args.rank, logger,
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
        args: "TrainParserArgs",
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
        data_provider = _load_data_provider(args, memmaped=True)

    assert args.n_runs != 0, "--n_runs cannot be 0"

    metrics_list: (
        list[tuple[Metric, Metric]]
        | list[tuple[Metric, Metric, Metric]]) = []
    for n_run in tqdm(range(args.n_runs), desc="Runs"):
        run_args = copy(args)
        run_args.model_name = f"{args.name}_{n_run}"
        run_args.seed = args.seed + n_run  # offset seed
        args_logic(run_args)  # also sets seed

        metrics_list.append(
            tuple(next(main_train(
                run_args,
                world_size,
                iterate=False,
                data_provider=data_provider)).values()))  # type: ignore

    means_and_stds = tuple(
        sum_and_std_metrics(seq)
        for seq in zip(*metrics_list))

    info(
        args.rank, logger,
        f"Performed {args.n_runs} training runs.")
    info(
        args.rank, logger,
        f"Final mean and std train: {dict_info(means_and_stds[0])}")
    info(
        args.rank, logger,
        f"Final mean and std dev: {dict_info(means_and_stds[1])}")
    if len(means_and_stds) > 2:
        info(
            args.rank, logger,
            f"Final mean and std test: {dict_info(means_and_stds[2])}")

    end = time.time()
    info(args.rank, logger, f"Took {end - start} seconds!")
    return means_and_stds  # type: ignore


def main_test(
        args: "TestParserArgs",
        world_size: int,
        data_provider: DataProvider | None = None
        ) -> tuple[Metric, Metric, Metric]:
    """Calculates the mean and standard deviation of several
    runs."""

    # device: where to execute computation
    if world_size > 1:
        assert args.rank is not None, "Rank cannot be None if word_size > 1."

    if is_undef(args.dependency_mode):
        trainer = LMTrainer.load(
            world_size=world_size,
            **args.to_dict())
    else:
        trainer = LMTrainer.load(
            world_size=world_size,
            use_input_mask=args.dependency_mode == "input",
            **args.to_dict())

    args.update_from_kwargs(**trainer.config.to_dict())

    if data_provider is None:
        data_provider = _load_data_provider(
            args,
            memmaped=True)

    # TODO: Hooks do not save dataset name or number.
    # Idea: add counter to trainer that counts number of received
    # datasets and give this number to hook

    model_dir = os.path.join(trainer.model_dir, args.model_name)

    if args.att_plot:
        trainer.add_hook(AttentionPlotHook(
            os.path.join(model_dir, "hooks", "att_plots")
        ))

    if args.tree_plot:
        trainer.add_hook(TreePlotHook(
            os.path.join(model_dir, "hooks", "tree_plots"),
            masks_setting=args.masks_setting
        ))

    # Training setting
    metrics = trainer.test(**data_provider.datasets)

    generated = []
    for _ in range(20):
        generated.append(
            trainer.generate(data_provider.datasets["token_mapper"]))
    info(
        args.rank, logger,
        f"Generated model output sample: {generated}")

    del trainer
    del data_provider

    return metrics  # type: ignore


def main_compare(
        args: "CompareParserArgs",
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
        assert args.rank is not None, "Rank cannot be None if word_size > 1."

    extra_args = args.to_dict()

    # model1
    trainer = LMTrainer.load(
        model_name=args.model1_name,
        world_size=world_size,
        **extra_args)

    args.update_from_kwargs(**trainer.config.to_dict())

    if not data_provider_given:
        data_provider = _load_data_provider(
            args,
            memmaped=True,
            model_num=1)

    assert data_provider is not None
    assert "eval" in data_provider.datasets
    dataset = data_provider.datasets["eval"]
    token_mapper = data_provider.datasets["token_mapper"]

    logprobs1, _ = trainer.predict(
        dataset,
        make_prob=True,
        only_true=True)

    # model2
    del trainer
    trainer = LMTrainer.load(
        model_name=args.model2_name,
        world_size=world_size,
        **extra_args)

    if not data_provider_given:
        data_provider = _load_data_provider(
            args,
            memmaped=True,
            model_num=2)

    logprobs2, _ = trainer.predict(
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
        args.rank, logger,
        f"{highest_num} tokens with the highest "
        f"difference: {highest_tokens_count}")

    info(args.rank, logger, "Tokens with window:")

    for c_diff, c_sen_num, c_pos in highest:
        tokens = token_mapper.decode([dataset.id_hl[c_sen_num][0]])[0]
        left = max(0, c_pos-window)
        right = min(len(tokens), c_pos+window)
        info(
            args.rank, logger,
            (
                f"diff={round(c_diff, 2)}: {' '.join(tokens[left:c_pos])} "
                f"[{tokens[c_pos]}] {' '.join(tokens[c_pos+1:right])}"))

    token_to_diffs: defaultdict[str, list[tuple[float, int, int]]]
    token_to_diffs = defaultdict(list)
    for token, diff in zip(highest_tokens, highest):
        token_to_diffs[token].append(diff)

    # Concordances
    info(args.rank, logger, "Concordances:")
    for token, _ in sorted(
            highest_tokens_count.items(), key=lambda x: x[1], reverse=True):
        info(args.rank, logger, f"\nToken: {token}\n")
        for c_diff, c_sen_num, c_pos in token_to_diffs[token]:
            tokens = token_mapper.decode([dataset.id_hl[c_sen_num][0]])[0]
            left = max(0, c_pos-window)
            right = min(len(tokens), c_pos+window)
            info(
                args.rank, logger,
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
        args.rank, logger,
        "\nProbability diffs ordered by improvement contribution:\n"
        + "\n".join(f"'{tup[0]}': {tup[1]}" for tup in sorted(
            mean_differences.items(),
            key=lambda x: x[1]*total_tokens_count[x[0]],
            reverse=True)))

    info(
        args.rank, logger,
        "\nProbability diffs ordered by worsening contribution:\n"
        + "\n".join(f"'{tup[0]}': {tup[1]}" for tup in sorted(
            mean_differences.items(),
            key=lambda x: x[1]*total_tokens_count[x[0]],
            reverse=False)))

    info(
        args.rank, logger,
        f"Perplexity 1: {np.exp(np.mean(ppl1))}")

    info(
        args.rank, logger,
        f"Perplexity 2: {np.exp(np.mean(ppl2))}")

    info(
        args.rank, logger,
        "Change of perplexity in total: "
        f"{np.exp(np.mean(ppl2)) - np.exp(np.mean(ppl1))}")


USE_LOG = {"learning_rate"}


def hyperopt_args_sampler(
        name: str,
        arg: T | list[T] | tuple[T, T],
        trial
        ) -> T:
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
        return cast(T, arg)


class Objective:
    def __init__(
            self, n_devices: int,
            args: "HyperoptParserArgs",
            writer: MetricWriter,
            pg):
        self.n_devices = n_devices
        self.args = args
        self.writer = writer
        self.pg = pg

        self.data_provider = None
        self.datasets = None

        # TODO: do not use try but check if any of the relevant args are
        # Hyperopt spaces
        try:
            self.data_provider = _load_data_provider(args, memmaped=True)
            # Since pin_memory=True, persistent_workers=True lead
            # to too many files
            # error when creating a lot of dataloaders, we need to construct
            # dataloaders here
            # Remove this if https://github.com/pytorch/pytorch/issues/91252
            # is resolved
            self.data_provider.datasets["train"] = get_loader(  # type: ignore
                    self.data_provider.datasets["train"],  # type: ignore
                    batch_size=self.args.batch_size,
                    bucket=False,
                    shuffle=True, droplast=True,
                    world_size=self.n_devices,
                    rank=self.args.rank,
                    n_workers=self.args.n_workers)
            self.data_provider.datasets["eval"] = get_loader(  # type: ignore
                    self.data_provider.datasets["eval"],  # type: ignore
                    batch_size=self.args.batch_size,
                    bucket=False,
                    shuffle=False, droplast=False,
                    world_size=self.n_devices,
                    rank=self.args.rank,
                    n_workers=self.args.n_workers)
        except TypeError:
            self.data_provider = None

    def __call__(self, trial) -> float:
        if self.n_devices > 1:
            trial = optuna.integration.TorchDistributedTrial(
                trial, self.pg)  # type: ignore

        args = TrainParserArgs.from_kwargs(**{
            name: hyperopt_args_sampler(name, arg, trial) for
            name, arg in self.args.to_dict().items()},
            model_name=f"{self.args.name}_{trial.number}",
            n_runs=1)
        args.seed = args.seed + trial.number
        args_logic(args)

        train_iterator = main_train(
            args, self.n_devices,
            iterate=True,
            data_provider=self.data_provider)
        assert train_iterator is not None

        should_prune = False
        metrics = None
        for step, metrics in enumerate(train_iterator, start=1):
            # Handle pruning based on the intermediate value.
            trial.report(
                getattr(metrics["eval"], self.args.optimise),
                step)

            if trial.should_prune():
                should_prune = True
                break

        assert metrics is not None, (
           "eval_interval is larger than total number of steps")
        if self.writer is not None:
            self.writer.add_params(
                args.to_dict(),
                metrics["eval"],
                run_name=str(trial.number),
                global_step=args.eval_interval*step)

        if should_prune:
            raise optuna.exceptions.TrialPruned()
        # trial.set_user_attr("metric_dicts", metric_dicts)

        loss: float = getattr(metrics["eval"], self.args.optimise)
        return loss


def main_hyperopt(
        args: "HyperoptParserArgs",
        world_size: int) -> None:
    direction = "minimize" if minimise[args.optimise.lower()] else "maximize"

    ld = os.path.join("./runs", f"{args.name}_hyperopt")
    with new_pg(world_size, "gloo") as pg, metric_writer(log_dir=ld) as writer:
        objective: Objective = Objective(world_size, args, writer, pg)
        if args.rank == 0 or args.rank is None:
            study = optuna.create_study(
                study_name=args.name,
                direction=direction,
                sampler=optuna.samplers.RandomSampler(
                    seed=args.seed),  # TODO: normal sampler
                pruner=optuna.pruners.MedianPruner(
                    n_warmup_steps=args.n_warmup_steps,
                    n_startup_trials=args.n_startup_trials))
            study.optimize(
                objective, n_trials=args.n_trials)

        else:
            for _ in range(args.n_trials):
                try:
                    objective(None)
                except optuna.TrialPruned:
                    pass

    if args.rank == 0 or args.rank is None:
        assert study is not None
        pruned_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        info(
            args.rank, logger,
            (
                f"Pruned {len(pruned_trials)}, "
                f"completed {len(complete_trials)} trials"))

        info(
            args.rank, logger,
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


def main(args: "ParserArgs") -> None:
    if args.mode == "dataprep":
        main_dataprep(args)
    else:
        try:
            n_devices = int(os.environ["WORLD_SIZE"]) if args.use_ddp else 1
        except ValueError:
            n_devices = torch.cuda.device_count() if args.use_ddp else 1
        assert not ((n_devices == 1 or not args.use_ddp) and
                    (args.rank is not None and args.rank > 0)), (
            "Rank cannot be larger than 0 if only having one device"
            "/not using ddp. "
            f"Received --local-rank {args.rank} --use_ddp {args.use_ddp} "
            f"and number of recognised CUDA devices is {n_devices}.")
        info(args.rank, logger, f"Running on {n_devices} devices.")
        with ddp(args.rank, n_devices) as ddp_status:
            info(args.rank, logger, f"Using DDP: {ddp_status}")
            mode = args.mode
            match mode:
                case "train":
                    assert isinstance(args, TrainParserArgs)
                    info(args.rank, logger, "Launching model training.")
                    main_train_multiple(args, n_devices)
                case "test":
                    assert isinstance(args, TestParserArgs)
                    info(args.rank, logger, "Launching model testing.")
                    main_test(args, n_devices)
                case "hyperopt":
                    assert isinstance(args, HyperoptParserArgs)
                    info(args.rank, logger, "Launching hyperparameter tuning.")
                    main_hyperopt(args, n_devices)
                case _:
                    assert isinstance(args, CompareParserArgs)
                    info(args.rank, logger, "Launching model comparison.")
                    main_compare(args, n_devices)


@dataclass
class ParserArgs(Params):
    mode: Literal[
        "train", "hyperopt", "dataprep", "test",
        "compare"]
    rank: int | None
    n_workers: int
    name: str
    device: str
    use_ddp: bool
    dataset_name: str
    max_len_train: None | int
    max_len_eval_test: None | int
    vocab_size: int | None
    triangulate: int
    first_k: int | None
    first_k_eval_test: int | None
    connect_with_dummy: bool
    connect_with_self: bool
    masks_setting: MasksSetting
    seed: int


@dataclass
class TrainParserArgs(ParserArgs):
    n_runs: int

    model_name: str
    dependency_mode: Literal["supervised", "input", "standard"]
    batch_size: int
    use_steps: bool
    max_steps: int | None
    eval_interval: int
    early_stop_after: int | None
    early_stop_metric: str | None
    epochs: int
    gradient_acc: int | None
    learning_rate: float
    loss_alpha: float | None
    arc_loss_weighted: bool
    discriminative: bool

    transformer_description: TransformerDescription | None
    layer_design: tuple[str, ...]
    use_standard: bool
    width: int
    depth: int
    unrestricted_before: int
    unrestricted_after: int
    d_ff_factor: int
    dropout: float | None
    dropout_attn: float | None
    dropout_resid: float | None
    dropout_ff: float | None
    dropout_embd: float | None
    dropout_lstm: float | None
    use_lstm: bool
    block_size: int
    n_embd: int
    overlay_causal: bool
    use_dual_fixed: bool
    bias: bool
    pos_enc: Literal["embedding", "sinusoidal"]


@dataclass
class HyperoptParserArgs(ParserArgs):
    optimise: Literal["perplexity", "uas", "loss", "lm_loss", "arc_loss"]
    n_warmup_steps: int
    n_startup_trials: int
    n_trials: int

    dependency_mode: Literal["supervised", "input", "standard"]
    batch_size: int
    use_steps: bool
    max_steps: int | None
    eval_interval: int
    early_stop_after: int | None
    early_stop_metric: str | None
    epochs: int
    gradient_acc: int | None

    learning_rate: float | tuple[float, float] | list[float]
    loss_alpha: float | tuple[float, float] | list[float | None] | None
    arc_loss_weighted: bool | list[bool]
    discriminative: bool | list[bool]

    block_size: int
    overlay_causal: bool

    transformer_description: (
        TransformerDescription
        | list[TransformerDescription])
    layer_design: tuple[str, ...] | list[tuple[str, ...]]
    use_standard: bool | list[bool]
    width: int | tuple[int, int] | list[int]
    depth: int | tuple[int, int] | list[int]
    unrestricted_before: int | tuple[int, int] | list[int]
    unrestricted_after: int | tuple[int, int] | list[int]
    d_ff_factor: int | tuple[int, int] | list[int]
    dropout: float | tuple[int, int] | list[int | None] | None
    dropout_attn: float | tuple[int, int] | list[int | None] | None
    dropout_resid: float | tuple[int, int] | list[int | None] | None
    dropout_ff: float | tuple[int, int] | list[int | None] | None
    dropout_embd: float | tuple[int, int] | list[int | None] | None
    dropout_lstm: float | tuple[int, int] | list[int | None] | None
    use_lstm: bool | list[bool]
    n_embd: int | tuple[int, int] | list[int]
    use_dual_fixed: bool | list[bool]
    bias: bool | list[bool]
    pos_enc: (
        Literal["embedding", "sinusoidal"]
        | list[Literal["embedding", "sinusoidal"]])


@dataclass
class TestParserArgs(ParserArgs):
    model_name: str
    dependency_mode: str | Undefined
    batch_size: int | Undefined
    loss_alpha: float | None | Undefined
    arc_loss_weighted: bool | Undefined

    att_plot: bool
    tree_plot: bool


@dataclass
class DataprepParserArgs(ParserArgs):
    pass


@dataclass
class CompareParserArgs(ParserArgs):
    model1_name: str
    model2_name: str
    batch_size: int


def args_logic(args: (
        TrainParserArgs | HyperoptParserArgs
        | DataprepParserArgs | TestParserArgs
        | CompareParserArgs)
        ) -> None:
    seed_everything(args.seed)
    args.device = make_device_str(args.device)
    if isinstance(args, TrainParserArgs):
        # parse transformer description

        # override dropout that was not set specifically
        for specific_dropout in (
                "dropout_attn", "dropout_resid",
                "dropout_ff", "dropout_embd",
                "dropout_lstm"):
            if getattr(args, specific_dropout) == -1:
                setattr(args, specific_dropout, args.dropout)

    if isinstance(args, (TrainParserArgs, TestParserArgs)):
        if args.model_name is None:
            args.model_name = args.name

    if args.rank is None and args.use_ddp:
        try:
            args.rank = int(os.environ["LOCAL_RANK"])
        except AttributeError:
            pass
