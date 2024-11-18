import torch

from data import load_dataset, dataset_details_full, DatasetDictTrain
from trainer import LMTrainer, TrainConfig, MITransformerConfig, Metric
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import pickle
import os.path


from typing import Literal, Any, Iterable, cast

# set the random seed, for reproducibility
torch.manual_seed(42)


def train(mode: Literal["standard", "input", "supervised"], dataset_name: str):
    # device: where to execute computation
    loss_alpha: float | None
    if mode == "supervised":
        loss_alpha = 0.5
    else:
        loss_alpha = None

    train_config = TrainConfig(
        batch_size=528,
        eval_interval=5,
        epochs=5,
        learning_rate=1e-3,
        mode=mode,
        loss_alpha=loss_alpha,
        model_dir="./model2",
        arc_loss_weighted=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # dropout rate (variable p) for dropout units
    dropout = 0.1
    n_embd = 400
    block_size = 500

    datasets = load_dataset(dataset_details_full[dataset_name],
                            max_len_train=40,
                            vocab_size=50_000,
                            triangulate=0,
                            connect_with_dummy=True,
                            connect_with_self=False)

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

    generated = []
    for _ in range(20):
        generated.append(trainer.generate(datasets["token_mapper"]))

    logits, arc_scores = trainer.predict(datasets["eval"])
    return trainer, generated, logits, arc_scores


def test_subset(batch_size: int = 100,
                first_k: int | None = 1_000,
                depth: int = 1,
                width: int = 1,
                add_standard: bool = False,
                use_lstm: bool = False,
                unrestricted_before: int = 0,
                unrestricted_after: int = 0,
                alpha: float = 1.0,
                dropout: float = 0.0,
                learning_rate: float = 1e-3,
                n_embd=400,
                datasets: DatasetDictTrain | None = None,
                num_tries: int = 1,
                device: str | None = None) -> dict[str, dict[str, Any]]:

    metrics_tries: list[tuple[Metric, ...]] = []

    for _ in range(num_tries):
        layer_design = (("head", "child", "standard")
                        if add_standard else ("head", "child"))
        n_embd = int(n_embd // (len(layer_design)*width)
                     // 2 * len(layer_design)*width * 2)
        block_size = 500
        core = tuple([(layer_design, width)
                      ] * depth)
        before = tuple([(("standard",), len(layer_design)*width)
                        ] * unrestricted_before)
        after = tuple([(("standard",), len(layer_design)*width)
                       ] * unrestricted_after)
        transformer_description = before + core + after

        train_config = TrainConfig(
            batch_size=batch_size,
            eval_interval=100,
            epochs=100,
            learning_rate=learning_rate,
            mode="supervised",
            loss_alpha=None,
            model_dir="placeholder",
            arc_loss_weighted=False,
            device=device if device is not None else (
                "cuda" if torch.cuda.is_available() else "cpu")
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
            use_lstm=use_lstm)

        # Experiments
        train_config["model_dir"] = "experiment"
        train_config["loss_alpha"] = alpha
        trainer = LMTrainer.new(transformer_config, train_config)
        metrics = trainer.train(**datasets)
        metrics_tries.append(metrics)

    metrics_mean: list[Metric] = []
    for split_metrics in zip(*metrics_tries):
        summed = split_metrics[0]
        for m in split_metrics[1:]:
            summed = summed + m
        metrics_mean.append(summed)

    metric_dicts: dict[str, dict[str, Any]] = {}
    for d, s in zip([cast(dict[str, Any], metric.to_dict())
                     for metric in metrics_mean],
                    ("train", "eval", "test")):
        d["split"] = s
        d["alpha"] = alpha
        d["batch_size"] = batch_size
        d["sentences"] = first_k
        d["depth"] = depth
        d["width"] = width
        d["add_standard"] = add_standard
        d["use_lstm"] = use_lstm
        d["unrestricted_before"] = unrestricted_before
        d["unrestricted_after"] = unrestricted_after
        d["n_embd"] = n_embd
        d["learning_rate"] = learning_rate
        d["dropout"] = dropout

        metric_dicts[s] = d
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


def test_multiple(result_file: str = "./results.csv",
                  tries: int = 3,
                  max_evals: int = 50,
                  objective_col: str = "perplexity",
                  device: str | None = None):
    sent_num_and_bs = [(100_000, 512)]
    params = dict(
        depth=[1, 0],
        width=[1],
        add_standard=[False],
        use_lstm=[True],
        unrestricted_before=[0],
        unrestricted_after=[0],
        alpha=[1.0, 0.5]
    )
    hyperopt_params = dict(
        n_embd=hp.quniform("n_embd", 300, 600, q=100),
        learning_rate=hp.uniform("learning_rate", 1e-5, 1e-2),
        dropout=hp.quniform("dropout", 0, 0.6, q=0.1)
    )
    objective_col = "perplexity"
    objective_maximize = (False if objective_col in ("perplexity", "loss",
                                                     "arc_loss", "lm_loss")
                          else True)
    factor = -1 if objective_maximize else 1

    for sent_num, batch_size in sent_num_and_bs:
        ds_det = dataset_details_full["Wikitext"]
        ds_det["dirs"] = ds_det["dirs"][0:2]  # type: ignore
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
            print("New kwargs choice:")

            def objective(hyperopt_params):
                d = test_subset(
                    batch_size=batch_size,
                    first_k=sent_num,
                    datasets=datasets,
                    num_tries=tries,
                    device=device,
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

            old_df = (pd.read_csv(result_file) if os.path.exists(result_file)
                      else pd.DataFrame())
            tid = trials.best_trial["tid"]
            pd.concat([
                old_df,
                pd.DataFrame([trials.attachments[f"ATTACH::{tid}::train"],
                              trials.attachments[f"ATTACH::{tid}::eval"]])]
                ).to_csv(
                    result_file, index=False,
                    mode="w")
