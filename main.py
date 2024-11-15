import torch

from data import load_dataset, dataset_details_full
from trainer import LMTrainer, TrainConfig, MITransformerConfig


from typing import Literal

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
        use_dual_fixed=True, bias=False)

    trainer = LMTrainer.new(transformer_config, train_config)

    trainer.train(**datasets)

    generated = []
    for _ in range(20):
        generated.append(trainer.generate(datasets["token_mapper"]))

    logits, arc_scores = trainer.predict(datasets["eval"])
    return trainer, generated, logits, arc_scores


def test_subset():
    # device: where to execute computation
    alphas = (1.0, 0.5, 0.0)

    # dropout rate (variable p) for dropout units
    dropout = 0.0
    n_embd = 400
    block_size = 500
    transformer_description = ((("head", "child"), 1),
                               )

    # Experiment 1
    train_config = TrainConfig(
        batch_size=100,
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
                            first_k=10_000,
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
        use_dual_fixed=False, bias=False)

    # Experiment 2
    for i, a in enumerate(alphas):
        i += 1
        print(f"Performing experiment {i}")
        train_config["model_dir"] = f"experiment{i}"
        train_config["loss_alpha"] = a

        trainer = LMTrainer.new(transformer_config, train_config)
        trainer.train(**datasets)
