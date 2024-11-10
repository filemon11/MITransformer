import torch

from data import (MemMapDataset, MemMapWindowDataset,
                  get_transform_mask_head_child)
from tokeniser import TokenMapper
from trainer import LMTrainer, TrainConfig, MITransformerConfig

import os

from typing import Literal


# set the random seed, for reproducibility
torch.manual_seed(42)


def load_dataset_and_token_mapper(
        ) -> tuple[MemMapDataset, MemMapDataset, TokenMapper]:
    memmap_dir = "./memmap"
    tokmap_dir = "./tokmap.pickle"

    ud = "./Universal Dependencies 2.14/ud-treebanks-v2.14"
    # path = os.path.join(ud, "UD_English-Atis/en_atis-ud-train.conllu")
    # path = os.path.join(ud, "UD_English-EWT/en_ewt-ud-train.conllu")
    # path = "./sample.conllu"
    # path = "./naturalstories-master/parses/ud/stories-aligned.conllx"
    path = "./wikitext_spacy_train.conllu"

    # path_val = os.path.join(ud, "UD_English-EWT/en_ewt-ud-dev.conllu")
    path_val = "./wikitext_spacy_dev.conllu"

    transform = get_transform_mask_head_child(
        keys_for_head={"head"},
        keys_for_child={"child"},
        triangulate=True)
    dataset = MemMapDataset.from_file(path, transform, max_len=40)
    val_dataset = MemMapDataset.from_file(path_val, transform)

    token_mapper = TokenMapper.train(
        dataset.tokens,
        keep_top_k=50_000)

    token_mapper.save(tokmap_dir)
    print("Vocab size", token_mapper.vocab_size)

    dataset.map_to_ids(token_mapper, memmap_dir)
    print("Number of sentences:", len(dataset))

    val_dataset.map_to_ids(token_mapper, f'{memmap_dir}_val')

    return dataset, val_dataset, token_mapper


def train(mode: Literal["standard", "input", "supervised"]):
    # device: where to execute computation
    train_config = TrainConfig(
        batch_size=528,
        eval_interval=5,
        epochs=100,
        learning_rate=1e-3,
        mode=mode,
        model_dir="./model",
        arc_loss_weighted=True
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dropout rate (variable p) for dropout units
    dropout = 0.1
    n_embd = 400
    block_size = 500

    dataset, val_dataset, token_mapper = load_dataset_and_token_mapper()

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
        vocab_size=token_mapper.vocab_size,
        overlay_causal=True, use_input_mask=(mode == "input"),
        use_dual_fixed=True, bias=False)

    trainer = LMTrainer.new(transformer_config, train_config, device)

    trainer.train(dataset, val_dataset)

    generated = []
    for _ in range(20):
        generated.append(trainer.generate(token_mapper))

    return trainer, generated
