"""
Snippets taken from
https://github.com/karpathy/nanoGPT/blob/master/model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import math

from collections import defaultdict

from typing import (TypedDict, NotRequired, Sequence, Unpack,
                    Literal)


def combine_scores(
        score_dicts: Sequence[dict[str, list[torch.Tensor]]]
        ) -> dict[str, list[torch.Tensor]]:
    """Currently: collects all scores for each tag
    category separately in a list.
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


class PositionalEncoding(nn.Module):

    def __init__(
            self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        begin_emb = torch.zeros((2, d_model))
        self.register_buffer('begin_emb', begin_emb)

    def forward(self, len: int) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = torch.cat((self.begin_emb, self.pe[:len-2]))
        # do not positionally encode dummy and root
        return self.dropout(x).unsqueeze(0)


class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids: torch.Tensor):
        # Input: (B, S)
        # Output: (B, S, E)

        # Need (S, B) format for encoder.
        src = self.encoder(input_ids) * math.sqrt(self.ninp)
        return self.pos_encoder(src)


class FeedForward(nn.Module):
    def __init__(self, n_embd, d_ff, dropout, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, d_ff, bias=bias),
            nn.GELU(),
            nn.Linear(d_ff, n_embd, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DualFixedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias: bool = False):
        super().__init__()
        # TODO: make possible to use when having other keys
        # => then head, child must be first pair
        # in addition to head and child
        # num_keys; saves 1/3 keys * proportion
        # dim: out_dim - (out_dim/3)*(2/num_keys)
        self.w_qkv = nn.Linear(in_dim, 2 * out_dim // 3, bias=bias)
        # -> 2*(M * H/2 * (E/3)*2)

    def forward(self, x):
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
            self, n_embd, layer_description: LayerDescription,
            block_size, attn_dropout,
            resid_dropout, overlay_causal: bool = False,
            use_dual_fixed: bool = False, bias: bool = False):
        # n_embd: embedding dimensionType[DependencyMultiHeadAttention]
        # n_heads : the number of heads we'd like to use
        super().__init__()

        self.tags: tuple[str, ...] = layer_description[0]
        self.n_multihead: int = len(self.tags)
        self.n_head: int = layer_description[1]
        self.head_size: int = n_embd // (self.n_multihead * self.n_head)
        assert n_embd % self.n_head == 0

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
            lambda t: einops.rearrange(t, 'b s (mh e) -> b mh s e',
                                       mh=mh), qkv)

        att: torch.Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # (B, MH, S, S)

        att = einops.rearrange(att, 'b (m h) s1 s2 -> b m h s1 s2',
                               m=self.n_multihead)
        att *= (1.0 / math.sqrt(k.shape[-1]))

        # (B, M, H, S, S)
        if self.overlay_causal:
            att = att.masked_fill(self.tril[:S, :S] == 0, float('-inf'))

        # TODO: get an empty mask for all tags not occurring in the
        # mask argument; then stack all masks on dim 0 and fill masks
        if masks is not None:
            mask_list: list[torch.Tensor] = []
            for tag in self.tags:
                if tag in masks.keys() and masks[tag] is not None:
                    mask_list.append(masks[tag])    # type: ignore
                else:
                    mask_list.append(self.ones[:S, :S].unsqueeze(0))

            masks_stacked = torch.stack(mask_list, dim=1)
            masks_stacked = masks_stacked.unsqueeze(2)  # (B, M, 1, S, S)

            att = att.masked_fill(
                masks_stacked.logical_not(),  # type: ignore
                float('-inf'))
            # unclear why type checker complains

        arc_scores = F.sigmoid(att)

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        att = einops.rearrange(att, 'b m h s1 s2 -> b (m h) s1 s2')

        out = torch.matmul(att, v)

        # reassamble all head outputs side by side
        out = einops.rearrange(out, 'b mh s e -> b s (mh e)')

        # output projection
        out = self.proj(out)
        out = self.resid_dropout(out)

        out_scores = {
            tag: [arc_scores[:, m, h] for h in range(self.n_head)]
            for tag, m in zip(self.tags, range(self.n_multihead))}
        return out, out_scores


class MILayer(nn.Module):
    """ Transformer design: comunication (attention) followed
    by computation (FFN) """
    # NOTE: This layer design only works for descriptions with
    # multihead-attention modules of the same size

    def __init__(
            self, n_embd, layer_description: LayerDescription,
            d_ff, block_size, attn_dropout, resid_dropout,
            dropout_ff, overlay_causal: bool = False,
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
        self.ff = FeedForward(n_embd, d_ff, dropout_ff, bias)

    def forward(
            self,
            x: torch.Tensor,
            masks: dict[str, torch.Tensor | None]
            ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """Mask shape: [M, B, S, S]
        with M number of multiheads,
        B batch size, S sequence length"""

        x_attn, out_scores = self.attn(self.ln_1(x), masks)
        x = x + x_attn
        x = x + self.ff(self.ln_2(x))

        return x, out_scores


TransformerDescription = tuple[LayerDescription, ...]

ProbMethod = Literal["sigmoid", "softmax"]


class TransformerDescription2(TypedDict):
    layers: tuple[LayerDescription, ...]
    prob_methods: dict[tuple[str, str] | str, ProbMethod]


class MITransformerConfig(TypedDict):
    transformer_description: TransformerDescription
    d_ff: int
    attn_dropout: float
    resid_dropout: float
    dropout_ff: float
    embd_dropout: float
    block_size: int
    n_embd: int
    dropout_embd: float
    vocab_size: int
    overlay_causal: bool
    use_input_mask: bool
    use_dual_fixed: bool
    bias: bool
    # I would like to make these optional
    # in the class constructor but still
    # type them; dataclass is not unpackable
    # and NotRequired necessitates conditional
    # assignment inside constructor which the
    # type checker does not recognise.


class MITransformer(nn.Module):
    def __init__(self, **kwargs: Unpack[MITransformerConfig]):
        super().__init__()

        n_embd = kwargs["n_embd"]
        transformer_description = kwargs["transformer_description"]

        self.layers = nn.ModuleList([MILayer(
            n_embd, layer_description, kwargs["d_ff"],
            kwargs["block_size"], kwargs["attn_dropout"],
            kwargs["resid_dropout"], kwargs["dropout_ff"],
            kwargs["overlay_causal"], kwargs["use_dual_fixed"],
            kwargs["bias"])
            for layer_description in transformer_description])

        self.block_size = kwargs["block_size"]

        self.vocab_size = kwargs["vocab_size"]

        # vocabulary embedding and positional embedding
        self.token_embedder = Encoder(self.vocab_size, n_embd,
                                      kwargs["dropout_embd"])

        self.wte = nn.Embedding(kwargs["vocab_size"], n_embd)
        # self.wpe = PositionalEncoding(n_embd, 0, self.block_size)
        self.wpe = nn.Embedding(self.block_size, n_embd)

        self.embd_dropout = nn.Dropout(kwargs["embd_dropout"])

        self.use_input_mask: bool = kwargs["use_input_mask"]

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections,
        # per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0,
                    std=0.02/math.sqrt(2 * len(transformer_description)))

        self.lstm = torch.nn.LSTM(n_embd, n_embd, batch_first=True)
        # self.transformer = torch.nn.Transformer(400, 1, 1, 1, 4*400, 0, batch_first=True)

    def _init_weights(self, module):
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
        pos_emb = self.wpe(torch.arange(0, S, device=tok_emb.device))

        x = self.embd_dropout(tok_emb + pos_emb)

        x = self.lstm(x)[0]
        # return x, {}
        # return self.transformer(x, x), {}
        scores = []
        for layer in self.layers:
            x, sc = layer(x, masks if self.use_input_mask else None)
            scores.append(sc)

        return x, combine_scores(scores)


class MITransformerLM(nn.Module):

    def __init__(self, mi_transformer: MITransformer):
        super().__init__()
        self.mi_transformer = mi_transformer

        n_embd = mi_transformer.wte.weight.shape[1]

        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd,
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

            Output : Shape[B, S, E]"""
        x, scores = self.mi_transformer(input_ids, masks)

        x = self.ln(x)
        logits = self.lm_head(x)

        # shape (B,T,C)  B : batch, T : sequence length, C : embedding dim

        # logits = x @ self.embds
        return logits, scores

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
