import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from data import (load_dataset, dataset_details_full, DatasetDictTrain,
                  dataset_details_full_memmaped)
from trainer import (LMTrainer, TrainConfig, MITransformerConfig,
                     Metric, MetricWriter, metric_writer,
                     sum_metrics)
import pandas as pd
import os.path
import sys

import optuna
from optuna.trial import TrialState

from typing import cast, Iterable, Any


def test_subset(rank,
                n_devices,
                batch_size: int = 100,
                first_k: int | None = 1_000,
                depth: int = 1,
                width: int = 1,
                stand: bool = False,
                lstm: bool = False,
                unr_bef: int = 0,
                unr_aft: int = 0,
                alpha: float = 1.0,
                dropout: float = 0.0,
                lr: float = 1e-3,
                n_embd=400,
                datasets: DatasetDictTrain | None = None,
                num_tries: int = 1,
                device: str | None = None,
                writer: MetricWriter | None = None
                ) -> dict[str, dict[str, Any]]:
    info_dict = dict(
        alpha=alpha,
        batch_size=batch_size,
        sentences=first_k,
        depth=depth,
        width=width,
        stand=stand,
        lstm=lstm,
        unr_bef=unr_bef,
        unr_aft=unr_aft,
        n_embd=n_embd,
        learning_rate=lr,
        dropout=dropout,
    )

    metrics_tries: list[tuple[Metric, ...]] = []

    for _ in range(num_tries):
        layer_design = (("head", "child", "standard")
                        if stand else ("head", "child"))
        n_embd = int(n_embd // (len(layer_design)*width)
                     // 2 * len(layer_design)*width * 2)
        block_size = 500
        core = tuple([(layer_design, width)
                      ] * depth)
        before = tuple([(("standard",), len(layer_design)*width)
                        ] * unr_bef)
        after = tuple([(("standard",), len(layer_design)*width)
                       ] * unr_aft)
        transformer_description = before + core + after

        train_config = TrainConfig(
            batch_size=batch_size,
            eval_interval=5,
            abort_after=3,
            epochs=100,
            learning_rate=lr,
            mode="supervised",
            loss_alpha=alpha,
            model_name="experiment",
            arc_loss_weighted=False,
            device=device if device is not None else (
                "cuda" if torch.cuda.is_available() else "cpu"),
            rank=rank,
            world_size=n_devices
            )

        if datasets is None:
            ds_det = dataset_details_full["Wikitext"]
            ds_det["dirs"] = ds_det["dirs"][0:2]  # type: ignore
            datasets = load_dataset(ds_det,
                                    max_len_train=40,
                                    max_len_eval_test=40,
                                    vocab_size=50_000,
                                    first_k=first_k,
                                    first_k_eval_test=None,
                                    triangulate=0,
                                    connect_with_dummy=True,
                                    connect_with_self=False)

        transformer_config = MITransformerConfig(
            transformer_description=transformer_description,
            d_ff=4*n_embd,
            attn_dropout=dropout,
            resid_dropout=dropout,
            dropout_ff=dropout,
            embd_dropout=dropout,
            block_size=block_size,
            n_embd=n_embd,
            dropout_embd=dropout,
            vocab_size=datasets["token_mapper"].vocab_size,
            overlay_causal=True, use_input_mask=False,
            use_dual_fixed=False, bias=False,
            use_lstm=lstm)

        # Experiments
        trainer = LMTrainer.new(transformer_config, train_config)
        metrics = trainer.train(**datasets)
        metrics_tries.append(metrics)

        del trainer

    metrics_mean: list[Metric] = []
    for split_metrics in zip(*metrics_tries):
        summed = split_metrics[0]
        for m in split_metrics[1:]:
            summed = summed + m
        metrics_mean.append(summed)

    if n_devices > 1:
        outputs_0 = [metrics_mean[0]]*n_devices
        dist.all_gather_object(outputs_0, metrics_mean[0])
        metrics_mean[0] = sum_metrics(outputs_0)
        outputs_1 = [metrics_mean[1]]*n_devices
        dist.all_gather_object(outputs_1, metrics_mean[0])
        metrics_mean[1] = sum_metrics(outputs_1)

    if writer is not None:
        writer.add_params(
            info_dict,
            metrics_mean[1])

    metric_dicts: dict[str, dict[str, Any]] = {}
    for d, s in zip([cast(dict[str, Any], metric.to_dict())
                     for metric in metrics_mean],
                    ("train", "eval", "test")):
        d["split"] = s
        metric_dicts[s] = d | info_dict
    return metric_dicts


class Objective:
    def __init__(self, rank, n_devices,
                 params, hyperopt_metric,
                 writer):
        self.rank = rank
        self.n_devices = n_devices
        self.params = params
        self.hyperopt_metric = hyperopt_metric
        self.writer = writer

    def __call__(self, trial) -> float:
        trial = optuna.integration.TorchDistributedTrial(
            trial, torch.cuda.current_device())

        n_embd = trial.suggest_int("n_embd", 300, 600, log=True)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0, 0.6)

        metric_dicts = test_subset(self.rank, self.n_devices,
                                   **self.params, n_embd=n_embd,
                                   lr=lr, dropout=dropout,
                                   writer=self.writer)

        trial.set_user_attr("metric_dicts", metric_dicts)
        loss: float = metric_dicts["eval"][self.hyperopt_metric]
        return loss


def log_trial(study, trial):
    # TODO
    pass
    # logger.info('to achieve objective function score of {}\n'.format(study.best_trial.value))


def hyperopt(
        rank: int | None, n_devices: int,
        objective_col: str,
        seed, datasets, n_trials: int, n_tries: int,
        config):
    maximise = {"UAS"}
    direction = "maximize" if objective_col in maximise else "minimize"

    with metric_writer() as writer:
        objective: Objective = Objective(rank, n_devices,
                                         config, objective_col,
                                         writer)
        if rank == 0 or rank is None:
            study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(
                    seed=seed),  # TODO: normal sampler
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=3, reduction_factor=3))
            study.optimize(
                objective, n_trials=n_trials, callbacks=[log_trial])

        else:
            for _ in range(options.n_trials):
                try:
                    objective(None)
                except optuna.TrialPruned:
                    pass

    if rank == 0 or rank is None:
        assert study is not None
        pruned_trials = study.get_trials(
            deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE])

        trial = study.best_trial
        return trial.user_attrs["metric_dicts"]
    else:
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


def hyperopt_multiple(
        rank,
        world_size: int,
        result_file: str = "./resultsUAS.csv",
        tries: int = 3,
        max_evals: int = 50,
        objective_col: str = "uas"):
    # TODO: make usable with DDP
    if world_size > 1:
        assert rank is not None
        device = rank % torch.cuda.device_count()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 100
    sent_num_and_bs = [(100_000, 256)]
    params = dict(
        depth=[1],
        width=[1],
        stand=[False],
        lstm=[True],
        unr_bef=[0],
        unr_aft=[0],
        alpha=[0.0])

    for sent_num, batch_size in sent_num_and_bs:
        ds_det = dataset_details_full["Wikitext"]
        ds_det["dirs"] = ds_det["dirs"][0:2]  # type: ignore
        # TODO: remove restriction 40 for eval; restrict at 500
        # -> switch from uint8 to uint16 for memmap
        # TODO: Ensure DDP-compatible dataset loading
        datasets = load_dataset(ds_det,
                                max_len_train=40,
                                max_len_eval_test=40,
                                vocab_size=50_000,
                                first_k=sent_num,
                                first_k_eval_test=None,
                                triangulate=0,
                                connect_with_dummy=True,
                                connect_with_self=False)

        for kwargs in options(list(params.items())):  # type: ignore
            attachments = hyperopt(rank, world_size, objective_col,
                                   seed, datasets,
                                   max_evals, tries,
                                   kwargs | {"batch_size": batch_size,
                                             "device": device,
                                             "n_tries": tries,
                                             "datasets": datasets},)

            old_df = (pd.read_csv(result_file) if os.path.exists(result_file)
                      else pd.DataFrame())

            pd.concat([
                old_df,
                pd.DataFrame([attachments['train'],
                              attachments['eval']])
                ]).to_csv(
                    result_file, index=False,
                    mode="w")