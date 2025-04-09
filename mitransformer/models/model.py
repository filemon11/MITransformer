"""
Snippets taken from
https://github.com/karpathy/nanoGPT/blob/master/model.py"""

from .. import utils
from . import pe

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import math

from collections import defaultdict
from dataclasses import dataclass

from typing import (TypedDict, NotRequired, Sequence, Mapping, Literal)


def combine_scores(
        score_dicts: Sequence[Mapping[str, list[torch.Tensor]]]
        ) -> dict[str, list[torch.Tensor]]:
    """Collects all scores for each tag
    category separately in a list.

    Parameters
    ----------
    score_dicts : Sequence[dict[str, list[torch.Tensor]]]
        Seqence of mappings of attention scores.

    Returns
    -------
    dict[str, list[torch.Tensor]]
        Dictionary of attention scores.
    """
    # Do we need to distinguish different layers (vertical heights) here?

    # maybe combination should not be taking the mean
    # if we are taking the mean and the number of heads wrt to tag i is
    # different on different layers then the scores would receive different
    # weights
    # maybe only append and take mean afterwards
    combined_dicts = defaultdict(list)
    for sc_dict in score_dicts:
        for key, scores in sc_dict.items():
            combined_dicts[key].extend(scores)

    return combined_dicts


class BatchInput(TypedDict):
    input_ids: torch.Tensor
    masks: NotRequired[dict[str, torch.Tensor | None]]


class FeedForward(nn.Module):
    def __init__(
            self, n_embd: int,
            d_ff_factor: int, dropout: float,
            bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, d_ff_factor*n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(d_ff_factor*n_embd, n_embd, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DualFixedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        # TODO: make possible to use when having other keys
        # => then head, child must be first pair
        # in addition to head and child
        # num_keys; saves 1/3 keys * proportion
        # dim: out_dim - (out_dim/3)*(2/num_keys)
        self.w_qkv = nn.Linear(in_dim, 2 * out_dim // 3, bias=bias)
        # -> 2*(M * H/2 * (E/3)*2)

    def forward(self, x: torch.Tensor):
        # x : [B, S, E]
        # -> B x (M/2 * H * E/3)
        qk, v = self.w_qkv(x).chunk(2, dim=-1)
        # take v from end
        # take q, k and chunk rest into two
        # cat q, k, r1, k, q, r2, v
        q, k = qk.chunk(2, dim=-1)
        m = torch.cat((q, k, k, q, v), dim=-1)
        return m


LayerDescription = tuple[tuple[str, ...], int]  # tag : num_heads


class MIAttention(nn.Module):
    # NOTE: This layer design only works for descriptions with
    # multihead-attention modules of the same size
    def __init__(
            self, n_embd: int, layer_description: LayerDescription,
            block_size: int, attn_dropout: float,
            resid_dropout: float, overlay_causal: bool = False,
            use_dual_fixed: bool = False, bias: bool = False):
        # n_embd: embedding dimensionType[DependencyMultiHeadAttention]
        # n_heads : the number of heads we'd like to use
        super().__init__()

        self.tags: tuple[str, ...] = layer_description[0]
        self.n_multihead: int = len(self.tags)
        self.n_head: int = layer_description[1]
        self.head_size: int = n_embd // (self.n_multihead * self.n_head)
        assert n_embd % (self.n_multihead * self.n_head) == 0

        #####
        self.scale = self.head_size ** -0.5

        self.w_qkv: nn.Linear | DualFixedLinear
        if use_dual_fixed:
            self.w_qkv = DualFixedLinear(n_embd, n_embd * 3, bias=bias)
        else:
            self.w_qkv = nn.Linear(n_embd, n_embd * 3, bias=bias)

        self.proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        self.overlay_causal: bool = overlay_causal
        if overlay_causal:
            self.register_buffer(
                'tril',
                torch.tril(torch.ones(block_size, block_size)))
        # The diagonal argument of torch.tril refers to
        # shifting the diagonal to consider,
        # diagonal = 0 means that the diagonal will be
        # filled with ones; -1 would mean leaving it out.

        self.register_buffer(
            'ones',
            torch.ones(block_size, block_size, dtype=torch.bool))

    def forward(
            self,
            x: torch.Tensor,
            masks: dict[str, torch.Tensor | None]
            ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """Mask shape: [M, B, S, S]
        with M number of multiheads,
        B batch size, S sequence length"""
        B, S, E = x.shape
        # B = batch size, S = sequence length, E = embedding dimensionality
        # H = head number, M = multihead number

        qkv = self.w_qkv(x).chunk(3, dim=-1)
        mh = self.n_head * self.n_multihead
        q, k, v = map(
            lambda t: einops.rearrange(
                t, 'b s (mh e) -> b mh s e',
                mh=mh), qkv)

        att: torch.Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # (B, MH, S, S)

        att = einops.rearrange(
            att, 'b (m h) s1 s2 -> b m h s1 s2',
            m=self.n_multihead)
        att *= (1.0 / math.sqrt(k.shape[-1]))

        # (B, M, H, S, S)
        if self.overlay_causal:
            att = att.masked_fill(
                self.tril[:S, :S] == 0, float('-inf'))  # type: ignore

        # TODO: get an empty mask for all tags not occurring in the
        # mask argument; then stack all masks on dim 0 and fill masks

        if masks is not None:
            mask_list: list[torch.Tensor] = []
            for tag in self.tags:
                if tag in masks.keys() and masks[tag] is not None:
                    mask_list.append(masks[tag])    # type: ignore
                else:
                    mask_list.append(
                        self.ones[:S, :S].unsqueeze(0))  # type: ignore

            masks_stacked = torch.stack(mask_list, dim=1)
            masks_stacked = masks_stacked.unsqueeze(2)  # (B, M, 1, S, S)

            att = att.masked_fill(
                masks_stacked.logical_not(),  # type: ignore
                float('-inf'))
            # unclear why type checker complains

        att_logits = att

        att = F.softmax(att, dim=-1)
        att = einops.rearrange(att, 'b m h s1 s2 -> b (m h) s1 s2')

        out = torch.matmul(att, v)

        # reassamble all head outputs side by side
        out = einops.rearrange(out, 'b mh s e -> b s (mh e)')

        # output projection
        out = self.proj(out)
        out = self.resid_dropout(out)

        out_logits = {
            tag: [att_logits[:, m, h] for h in range(self.n_head)]
            for tag, m in zip(self.tags, range(self.n_multihead))}
        return out, out_logits


class MILayer(nn.Module):
    """ Transformer design: comunication (attention) followed
    by computation (FFN) """
    # NOTE: This layer design only works for descriptions with
    # multihead-attention modules of the same size

    def __init__(
            self, n_embd: int, layer_description: LayerDescription,
            d_ff_factor: int, block_size: int, attn_dropout: float,
            resid_dropout: float,
            dropout_ff: float, overlay_causal: bool = False,
            use_dual_fixed: bool = False, bias: bool = False):
        # n_embd: embedding dimensionType[DependencyMultiHeadAttention]
        # n_heads : the number of heads we'd like to use
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = MIAttention(n_embd, layer_description,
                                block_size, attn_dropout,
                                resid_dropout, overlay_causal,
                                use_dual_fixed)
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        self.ff = FeedForward(n_embd, d_ff_factor, dropout_ff, bias)

    def forward(
            self,
            x: torch.Tensor,
            masks: dict[str, torch.Tensor | None]
            ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """Mask shape: [M, B, S, S]
        with M number of multiheads,
        B batch size, S sequence length"""

        x_attn, out_logits = self.attn(self.ln_1(x), masks)
        x = x + x_attn
        x = x + self.ff(self.ln_2(x))

        return x, out_logits


TransformerDescription = tuple[LayerDescription, ...]


@dataclass
class MITransformerConfig(utils.Params):
    transformer_description: TransformerDescription = ((("head", "child"), 1),)
    d_ff_factor: int = 4
    dropout_attn: float = 0.3
    dropout_resid: float = 0.3
    dropout_ff: float = 0.3
    dropout_embd: float = 0.3
    dropout_lstm: float = 0.3
    block_size: int = 500
    n_embd: int = 400
    vocab_size: int = 50_000
    overlay_causal: bool = True
    use_input_mask: bool = False
    use_dual_fixed: bool = False
    bias: bool = False
    use_lstm: bool = True
    pos_enc: Literal["embedding", "sinusoidal"] = "embedding"


class MITransformer(nn.Module):
    def __init__(self, config: MITransformerConfig):
        super().__init__()

        n_embd = config.n_embd
        transformer_description = config.transformer_description

        self.layers = nn.ModuleList([MILayer(
            n_embd, layer_description, config.d_ff_factor,
            config.block_size, config.dropout_attn,
            config.dropout_resid, config.dropout_ff,
            config.overlay_causal, config.use_dual_fixed,
            config.bias)
            for layer_description in transformer_description])
        self.block_size = config.block_size

        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, n_embd)
        # self.wpe = PositionalEncoding(n_embd, 0, self.block_size)
        self.pos_enc_type = config.pos_enc
        self.wpe: nn.Embedding | pe.PositionalEncoding
        if config.pos_enc == "embedding":
            self.wpe = nn.Embedding(self.block_size, n_embd)
        else:
            self.wpe = pe.PositionalEncoding(
                config.n_embd, 0,
                config.block_size)

        self.embd_dropout = nn.Dropout(config.dropout_embd)

        self.use_input_mask: bool = config.use_input_mask

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections,
        # per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=0.02/math.sqrt(2 * len(transformer_description)))

        self.lstm = None
        if config.use_lstm:
            self.lstm = torch.nn.LSTM(
                n_embd, n_embd, batch_first=True, dropout=config.dropout_lstm)
            self.ln_1 = nn.LayerNorm(n_embd)
            self.ln_2 = nn.LayerNorm(n_embd)
            self.ff = FeedForward(
                n_embd, config.d_ff_factor, config.dropout_ff, config.bias)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self, input_ids: torch.Tensor,
            masks: dict[str, torch.Tensor | None] | None = None,
            **kwargs
            ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """
            Input:
            x : [B, S, E]
            masks : dictionary mapping from tags to masks
            mask shape: [B, S, S]
            masks should be a boolean mask with True being the elements not
            to mask and False being the entries to mask.

            Output : Shape[B, S, E]
        call the model with idx and targets (training)
        or without targets (generation)"""
        S = input_ids.shape[1]

        tok_emb = self.wte(input_ids.long())
        if self.pos_enc_type == "embedding":
            pos_emb = self.wpe(torch.arange(0, S, device=tok_emb.device))
        else:
            pos_emb = self.wpe(S)

        x = self.embd_dropout(tok_emb + pos_emb)

        if self.lstm is not None:
            x = x + self.lstm(self.ln_1(x))[0]
            x = x + self.ff(self.ln_2(x))

        att_logits = []
        for layer in self.layers:
            x, al = layer(x, masks if self.use_input_mask else None)
            att_logits.append(al)

        return x, combine_scores(att_logits)


class MITransformerLM(nn.Module):

    def __init__(self, mi_transformer: MITransformer):
        super().__init__()
        self.mi_transformer = mi_transformer

        n_embd = mi_transformer.wte.weight.shape[1]

        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(
            n_embd,
            mi_transformer.vocab_size, bias=False)

        self.lm_head.weight = mi_transformer.wte.weight

        self._init_weights(self.lm_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self, input_ids: torch.Tensor,
            masks: dict[str, torch.Tensor | None] | None = None, **kwargs
            ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """
            Input:
            x : [B, S, E]
            masks : dictionary mapping from tags to masks
            mask shape: [B, S, S]
            masksshould be a boolean mask with True being the elements not
            to mask and False being the entries to mask.

            Output : Shape[B, S, E]
        """

        x, att_logits = self.mi_transformer(input_ids, masks)

        x = self.ln(x)
        logits = self.lm_head(x)

        # shape (B,T,C)  B : batch, T : sequence length, C : embedding dim

        # logits = x @ self.embds
        return logits, att_logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs):
        """ given a context idx, generate max_new_tokens tokens
        and append them to idx """
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.mi_transformer.block_size:]
            # we can never have any idx longer than block_size
            logits = self(idx_cond, {"head": None, "child": None})[0]
            # call fwd without targets
            logits = logits[:, -1, :]
            # take last token. from shape (B, C, T) to (B, C)
            # convert logits to probabilities
            probs = F.softmax(logits, dim=-1)   # shape (B, C)
            # randomly sample the next tokens, 1 for each of
            # the previous probability distributions
            # (one could take instead the argmax, but that would
            # be deterministic and boring)
            input_ids_next = torch.multinomial(probs, num_samples=1)
            # shape (B, 1)
            # append next token ix to the solution sequence so far
            input_ids = torch.cat([input_ids, input_ids_next], dim=-1)
            # shape (B, T+1)
        return input_ids


def description_builder(
        layer_design: tuple[str, ...] = ("head", "child"),
        use_standard: bool = False,
        width: int = 1,
        depth: int = 1,
        unrestricted_before: int = 0,
        unrestricted_after: int = 0
        ):
    if use_standard:
        layer_design += ("standard",)
    core = tuple([(layer_design, width)
                  ] * depth)
    before = tuple([(("standard",), len(layer_design)*width)
                    ] * unrestricted_before)
    after = tuple([(("standard",), len(layer_design)*width)
                   ] * unrestricted_after)
    return before + core + after
