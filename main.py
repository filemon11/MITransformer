# -*- coding: future_typing -*-
import torch

from data import load_dataset, dataset_details
from trainer import LMTrainer, TrainConfig, MITransformerConfig


from typing import Literal

# set the random seed, for reproducibility
torch.manual_seed(42)


def train(mode: Literal["standard", "input", "supervised"], dataset_name: str):
    # device: where to execute computation
    train_config = TrainConfig(
        batch_size=528,
        eval_interval=5,
        epochs=5,
        learning_rate=1e-3,
        mode=mode,
        model_dir="./model2",
        arc_loss_weighted=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # dropout rate (variable p) for dropout units
    dropout = 0.1
    n_embd = 400
    block_size = 500

    datasets = load_dataset(dataset_details[dataset_name],
                            max_len_train=40,
                            vocab_size=50_000)

    # Model
    transformer_description = ((("head", "child"), 1),
                               )
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
