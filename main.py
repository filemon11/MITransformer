import torch

from data import load_dataset, dataset_details_full
from trainer import LMTrainer, TrainConfig, MITransformerConfig
import pandas as pd

from typing import Literal, Any, TypedDict, Iterable

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
                alphas: tuple[float, ...] = (0.0, 0.5, 1.0)):

    layer_design = (("head", "child", "standard")
                    if add_standard else ("head", "child"))
    # dropout rate (variable p) for dropout units
    dropout = 0.0
    n_embd = 400 // len(layer_design) * len(layer_design)
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
        learning_rate=1e-3,
        mode="supervised",
        loss_alpha=None,
        model_dir="placeholder",
        arc_loss_weighted=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
        )

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
    dataframe_list = []
    for i, a in enumerate(alphas):
        i += 1
        print(f"Performing experiment {i}")
        train_config["model_dir"] = f"experiment{i}"
        train_config["loss_alpha"] = a

        trainer = LMTrainer.new(transformer_config, train_config)
        metrics = trainer.train(**datasets)

        metric_dicts: list[dict[str, Any]] = [
            dict(metric.to_dict()) for metric in metrics]
        for d, s in zip(metric_dicts, ("train", "eval", "test")):
            d["split"] = s
            d["alpha"] = a
            d["batch_size"] = batch_size
            d["sentences"] = first_k
            d["depth"] = depth
            d["width"] = width
            d["add_standard"] = add_standard
            d["use_lstm"] = use_lstm
            d["unrestricted_before"] = unrestricted_before
            d["unrestricted_after"] = unrestricted_after
        dataframe_list.append(pd.DataFrame(metric_dicts))

    return pd.concat(dataframe_list)


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


def test_multiple():
    sent_num_and_bs = [(1, 1), (100, 10), (1_000, 100)]
    params = dict(
        depth=[1, 2, 5],
        width=[1, 2, 5],
        add_standard=[False, True],
        use_lstm=[False, True],
        unrestricted_before=[0, 1],
        unrestricted_after=[0, 1],
    )

    for sent_num, batch_size in sent_num_and_bs:
        for kwargs in options(list(params.items())):
            df = test_subset(batch_size=batch_size,
                             first_k=sent_num,
                             **kwargs)
            df.to_csv("./results.csv", index=False,
                      mode="a")
