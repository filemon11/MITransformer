import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from data import (load_dataset, dataset_details_full, DatasetDictTrain,
                  dataset_details_full_memmaped)
from trainer import (LMTrainer, TrainConfig, MITransformerConfig,
                     Metric, MetricWriter)
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK  # type: ignore
import os.path
import sys
from logmaker import getLogger, log, basicConfig, INFO, info

from typing import Literal, Any, Iterable, cast

logger = getLogger(__name__)

# set the random seed, for reproducibility
torch.manual_seed(42)


def train(rank: int | None, world_size: int, n_workers: int,
          mode: Literal["standard", "input", "supervised"], dataset_name: str):
    # device: where to execute computation
    loss_alpha: float | None
    if mode == "supervised":
        loss_alpha = 0.5
    else:
        loss_alpha = None
    if world_size > 1:
        assert rank is not None
        device = rank % torch.cuda.device_count()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train_config = TrainConfig(
        batch_size=10,
        eval_interval=5,
        abort_after=1,
        epochs=100,
        learning_rate=1e-3,
        mode=mode,
        loss_alpha=loss_alpha,
        model_name="experiment",
        arc_loss_weighted=False,
        device=device,
        rank=rank,
        world_size=world_size,
        n_workers=n_workers
        )

    # dropout rate (variable p) fquit(or dropout units
    dropout = 0.0002
    n_embd = 500
    block_size = 500

    datasets: DatasetDictTrain
    # TODO: do this entirely in the load dataset method
    # load memmap
    details = dataset_details_full_memmaped[dataset_name]
    details["memmaped"] = details["memmaped"][0:2]  # type: ignore
    datasets = load_dataset(details,
                            max_len_train=40,
                            max_len_eval_test=40,
                            vocab_size=50_000,
                            triangulate=0,
                            first_k=100_000,
                            first_k_eval_test=None,
                            connect_with_dummy=True,
                            connect_with_self=False)
    assert isinstance(datasets, dict)

    # Model
    transformer_description = ((("head", "child"), 1),)
    # 24 heads, one layer approximately matches CBR-RRN

    # TODO: Make this a proper config
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
        overlay_causal=True, use_input_mask=(mode == "input"),
        use_dual_fixed=True, bias=False,
        use_lstm=False)

    trainer = LMTrainer.new(transformer_config, train_config)
    trainer.train(**datasets)

    if rank is None or rank == 0:
        generated = []
        for _ in range(20):
            generated.append(trainer.generate(datasets["token_mapper"]))
        info(rank, logger, f"Generated model output sample: {generated}")

    del trainer


def test_subset(batch_size: int = 100,
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
            rank=0,
            world_size=1,
            n_workers=0
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


def test_multiple(result_file: str = "./resultsUAS.csv",
                  tries: int = 3,
                  max_evals: int = 50,
                  objective_col: str = "uas",
                  device: str | None = None):
    # TODO: make usable with DDP
    sent_num_and_bs = [(100_000, 256)]
    params = dict(
        depth=[1],
        width=[1],
        stand=[False],
        lstm=[True],
        unr_bef=[0],
        unr_aft=[0],
        alpha=[0.0]
    )
    hyperopt_params = dict(
        n_embd=hp.quniform("n_embd", 300, 600, q=100),
        lr=hp.uniform("lr", 1e-5, 1e-2),
        dropout=hp.quniform("dropout", 0, 0.6, q=0.1)
    )
    objective_maximize = (False if objective_col in ("perplexity", "loss",
                                                     "arc_loss", "lm_loss")
                          else True)
    factor = -1 if objective_maximize else 1

    for sent_num, batch_size in sent_num_and_bs:
        ds_det = dataset_details_full["Wikitext"]
        ds_det["dirs"] = ds_det["dirs"][0:2]  # type: ignore
        # TODO: remove restriction 40 for eval; restrict at 500
        # -> switch from uint8 to uint16 for memmap
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
            writer = MetricWriter()

            def objective(hyperopt_params):
                d = test_subset(
                    batch_size=batch_size,
                    first_k=sent_num,
                    datasets=datasets,
                    num_tries=tries,
                    device=device,
                    writer=writer,
                    **kwargs,
                    **hyperopt_params)
                return {"loss": factor*d["eval"][objective_col],
                        "attachments": d,
                        'status': STATUS_OK}

            trials = Trials()
            fmin(
                objective,
                space=hyperopt_params,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)
            writer.flush()

            old_df = (pd.read_csv(result_file) if os.path.exists(result_file)
                      else pd.DataFrame())
            tid = trials.best_trial["tid"]  # type: ignore
            pd.concat([
                old_df,
                pd.DataFrame([trials.attachments[f"ATTACH::{tid}::train"],
                              trials.attachments[f"ATTACH::{tid}::eval"]])
                ]).to_csv(
                    result_file, index=False,
                    mode="w")
            writer.close()


def setup_ddp(rank, world_size) -> tuple[bool, int | None]:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '65035'
    if world_size > 1:
        print("initialising process", world_size, rank)
        dist.init_process_group("nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(torch.distributed.get_rank())
        return True, torch.distributed.get_rank()
    else:
        return False, None


def clean_ddp(world_size) -> None:
    if world_size > 1:
        dist.destroy_process_group()


def main(rank, n_devices) -> None:
    use_ddp, rank = setup_ddp(rank, n_devices)
    mode = sys.argv[1]
    n_workers = int(sys.argv[2])
    if mode == "train":
        info(rank, logger, "Launching model training.")
        train(rank, n_devices, n_workers, "supervised", "Wikitext",)
    else:
        info(rank, logger, "Launching hyperparameter tuning.")
        test_multiple()

    clean_ddp(n_devices)


if __name__ == "__main__":
    basicConfig(stream=sys.stdout, level=INFO)

    # tokenise dataset
    # TODO: do this automatically
    # problem: cannot do in parallel and
    # therefore leads to timeout when doing
    # on only one rank.
    # details = dataset_details_full["Wikitext"]
    # details["dirs"] = details["dirs"][0:2]  # type: ignore
    #
    # datasets = load_dataset(details,
    #                         max_len_train=40,
    #                         max_len_eval_test=40,
    #                         vocab_size=50_000,
    #                         triangulate=0,
    #                         first_k=100_000,
    #                         first_k_eval_test=None,
    #                         connect_with_dummy=True,
    #                         connect_with_self=False)

    n_devices = torch.cuda.device_count()
    info(None, logger, "Recognised {n_devices} CUDA devices.")

    if n_devices > 1:
        processes = []
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        info(None, logger, "Starting {n_devices} processes.")
        for rank in range(n_devices):
            p = mp.Process(target=main, args=(rank, n_devices))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        info(None, logger,
             "Running one process without multiprocessing module.")
        main(None, n_devices)
    # if n_devices > 1:
    #     mp.spawn(main, args=(n_devices,), nprocs=n_devices)  # type: ignore
    # else:
    #     main(None, n_devices)
