"""
Custom GPT models

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

from typing import (
    Sequence, Mapping, Literal, TypedDict, NotRequired)


AdditionalKeys = Literal[
    "proj_state_norms", "att", "activations", "embeddings"]


class AdditionalResults(TypedDict):
    proj_state_norms: NotRequired[torch.Tensor]
    att: NotRequired[torch.Tensor]
    embeddings: NotRequired[torch.Tensor]
    activations: NotRequired[torch.Tensor]


def combine_scores(
        score_dicts: Sequence[Mapping[str, list[torch.Tensor]]]
        ) -> dict[str, torch.Tensor]:
    """Collects all scores for each tag
    category separately in a list.

    Parameters
    ----------
    score_dicts : Sequence[dict[str, list[torch.Tensor]]]
        Seqence of mappings of attention scores.

    Returns
    -------
    dict[str, torch.Tensor[shape[M, B, S, S]]]
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

    return {key: torch.stack(scores) for key, scores in combined_dicts.items()}


class FeedForward(nn.Module):
    """Feed forward network with two
    linear networks and GELU activation function.
    """
    def __init__(
            self, n_embd: int,
            d_ff_factor: int, dropout: float,
            bias: bool = True):
        """Initialise feed forward network.
        Parameters
        ----------
        n_embd : int
            Input and output dimensionality.
        d_ff_factor : int
            Factor for inner dimensionality.
            Size is `n_embd*d_ff_factor`.
        dropout : float
            Dropout to apply after feed forward network.
        bias : bool, default=True
            Include bias in linear networks.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, d_ff_factor*n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(d_ff_factor*n_embd, n_embd, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed forward network
        to an input.

        Parameters
        ----------
        x : torch.Tensor
            Input.

        Returns
        -------
        torch.Tensor
            Feed forward network output.
        """
        return self.net(x)


class DualFixedLinear(nn.Module):
    """Dual fixed layer for fixing
    governor head Q and dependent K
    as well as governor head K and dependent Q.
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        """Initialise dual fixed layer.

        Parameters
        ----------
        in_dim : int
            Input dimensionality.
        out_dim : int
            Output dimensionality.
        bias : bool, default=False
            Whether to include bias in linear layer.
        """
        super().__init__()
        # TODO: make possible to use when having other keys
        # in addition to head and child
        # => then head, child must be first pair
        # num_keys; saves 1/3 keys * proportion
        # dim: out_dim - (out_dim/3)*(2/num_keys)
        self.w_qkv = nn.Linear(in_dim, 2 * out_dim // 3, bias=bias)
        # -> 2*(M * H/2 * (E/3)*2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dual fixed linear module.

        Parameters
        ----------
        x : torch.Tensor, shape[B, S, in_dim]
            Input.

        Returns
        -------
        torch.Tensor, shape[B, S, out_dim]
            Output.
        """
        out: torch.Tensor = self.w_qkv(x)
        qk, v = out.chunk(2, dim=-1)
        # take v from end
        # take q, k and chunk rest into two
        q, k = qk.chunk(2, dim=-1)
        m = torch.cat((q, k, k, q, v), dim=-1)
        return m


LayerDescription = tuple[tuple[str, ...], int]
"""Description for transformer layer:
tuple of head names and a dimensionality."""


class Attention(nn.Module):
    def __init__(
            self, n_embd: int, layer_description: tuple[tuple[str], int],
            block_size: int, attn_dropout: float,
            resid_dropout: float, overlay_causal: bool = False,
            use_dual_fixed: bool = False, bias: bool = False):
        """Initialise mask-informed attention module.

        Parameters
        ----------
        n_embd : int
            Module layer width.
        layer_description : LayerDescription
            Layer description.
        block_size : int
            Maximum sequence length.
        attn_dropout : float
            Dropout for attention weights.
        resid_dropout : float
            Dropout for residual connection.
        overlay_causal : bool, default=False
            Apply causal mask, i.e. make model incremental.
        use_dual_fixed : bool, default=False
            Cross-fix query and key vectors in dual-head
            transformer.
        bias : bool, default=False
            Include bias in linear layers.
        """
        # n_embd: embedding dimensionType[DependencyMultiHeadAttention]
        # n_heads : the number of heads we'd like to use
        super().__init__()
        self.tag: str = layer_description[0][0]
        self.n_head: int = layer_description[1]
        self.head_size: int = n_embd // self.n_head
        assert n_embd % self.n_head == 0

        #####
        self.scale = self.head_size ** -0.5

        self.w_qkv: nn.Linear | DualFixedLinear
        if use_dual_fixed:
            self.w_qkv = DualFixedLinear(n_embd, n_embd * 3, bias=bias)
        else:
            self.w_qkv = nn.Linear(n_embd, n_embd * 3, bias=bias)

        self.proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = attn_dropout
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
            return_arc_logits: bool = False,
            return_proj_state_norms: bool = False,
            return_att: bool = False,
            ) -> tuple[
                torch.Tensor,
                None | dict[str, list[torch.Tensor]],
                AdditionalResults]:
        """
        return_arc_logits: bool, default=False
            Return per-head attention logits.
        return_proj_state_norms: bool, default=False
            Return projected states.
        return_att: bool, default=False
            Return attention distribution."""

        """Mask shape: [M, B, S, S]
        with M number of multiheads,
        B batch size, S sequence length"""
        B, S, E = x.shape
        H = self.n_head
        E = self.head_size
        # B = batch size, S = sequence length, E = embedding dimensionality
        # H = head number, M = multihead number

        qkv = self.w_qkv(x).chunk(3, dim=-1)

        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor

        q, k, v = map(
            lambda t: einops.rearrange(
                t, 'b s (h e) -> b h s e',
                h=H), qkv)

        att: torch.Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # (B, H, S, S)

        att *= (1.0 / math.sqrt(k.shape[-1]))

        # (B, H, S, S)
        if self.overlay_causal:
            att = att.masked_fill(
                self.tril[:S, :S] == 0, float('-inf'))  # type: ignore

        # TODO: get an empty mask for all tags not occurring in the
        # mask argument; then stack all masks on dim 0 and fill masks

        if self.attn_dropout > 0 and self.training:
            att[..., 2:][torch.rand(
                att.shape, device=att.device)[..., 2:]
                < self.attn_dropout] = float('-inf')

        att_logits = att

        att = F.softmax(att, dim=-1)

        out = torch.matmul(att, v)

        # reassamble all head outputs side by side
        out = einops.rearrange(out, 'b h s e -> b s (h e)')

        # output projection
        out = self.proj(out)
        out = self.resid_dropout(out)

        out_logits: None | dict[str, list[torch.Tensor]] = None
        if return_arc_logits:
            out_logits = {
                self.tag: [att_logits[:, h] for h in range(self.n_head)]}

        additional_output: AdditionalResults = {}
        if return_att:
            additional_output["att"] = att

        if return_proj_state_norms:
            # adapted from Goro Kobayashi
            # (https://github.com/gorokoba560/norm-analysis-of-transformer)
            v_layer = v.permute(0, 2, 1, 3).contiguous().unsqueeze(3)
            # b h s e -> b s h 1 e

            # dense weight is converted to
            # (num_heads, head_size, all_head_size)
            output_weight = self.proj.weight.view(H, E, H*E)
            # he he -> h e he

            # create transformed vectors f(x) from value vectors (value_layer)
            # and weight matrix (output_weight).
            # the bias of the output transformation is assumed to be
            # distributed equally among heads
            proj_state_norms = v_layer.matmul(output_weight).squeeze(3)
            # (b s h 1 e) x (h e he) -> b s h he

            proj_state_norms = proj_state_norms.permute(
                0, 2, 1, 3).contiguous()
            if self.proj.bias is not None:
                proj_state_norms += (self.proj.bias / H)
            # (b h s he)

            proj_state_norms = torch.linalg.vector_norm(
                proj_state_norms, dim=-1)  # (b, h, s)

            # Normalise here (TODO: use my normalise function)
            # att: (b, h, s, s)
            proj_state_norms = att * proj_state_norms.unsqueeze(-2)
            # (b, h, s, s)

            proj_state_norms = proj_state_norms / proj_state_norms.sum(
                dim=-1, keepdim=True).clamp_min(1e-12)

            additional_output["proj_state_norms"] = proj_state_norms

        return out, out_logits, additional_output


class MIAttention(nn.Module):
    """Mask-informed attention module.
    """
    # NOTE: This layer design only works for descriptions with
    # multihead-attention modules of the same size
    def __init__(
            self, n_embd: int, layer_description: LayerDescription,
            block_size: int, attn_dropout: float,
            resid_dropout: float, overlay_causal: bool = False,
            use_dual_fixed: bool = False, bias: bool = False):
        """Initialise mask-informed attention module.

        Parameters
        ----------
        n_embd : int
            Module layer width.
        layer_description : LayerDescription
            Layer description.
        block_size : int
            Maximum sequence length.
        attn_dropout : float
            Dropout for attention weights.
        resid_dropout : float
            Dropout for residual connection.
        overlay_causal : bool, default=False
            Apply causal mask, i.e. make model incremental.
        use_dual_fixed : bool, default=False
            Cross-fix query and key vectors in dual-head
            transformer.
        bias : bool, default=False
            Include bias in linear layers.
        """
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

        self.attn_dropout = attn_dropout
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
            masks: dict[str, torch.Tensor | None] | None = None,
            return_arc_logits: bool = False,
            return_proj_state_norms: bool = False,
            return_att: bool = False
            ) -> tuple[
                torch.Tensor,
                None | dict[str, list[torch.Tensor]],
                AdditionalResults]:
        """
        return_arc_logits: bool, default=False
            Return per-head attention logits.
        return_proj_state_norms: bool, default=False
            Return projected states.
        return_att: bool, default=False
            Return attention distribution."""

        tags_l = list(self.tags)
        # just for compatibility
        # for i in range(len(tags_l)):
        #     if (
        #             not tags_l[i].endswith("_next")
        #             and not tags_l[i].endswith("_current")):
        #         tags_l[i] = tags_l[i] + "_current"

        tags = tuple(tags_l)

        """Mask shape: [M, B, S, S]
        with M number of multiheads,
        B batch size, S sequence length"""
        B, S, E = x.shape
        M = self.n_multihead
        H = self.n_head
        E = self.head_size
        # B = batch size, S = sequence length, E = embedding dimensionality
        # H = head number, M = multihead number

        qkv = self.w_qkv(x).chunk(3, dim=-1)

        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor

        q, k, v = map(
            lambda t: einops.rearrange(
                t, 'b s (mh e) -> b mh s e',
                mh=H*M), qkv)

        att: torch.Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # (B, MH, S, S)

        att = einops.rearrange(
            att, 'b (m h) s1 s2 -> b m h s1 s2',
            m=M)
        att *= (1.0 / math.sqrt(k.shape[-1]))

        # (B, M, H, S, S)
        if self.overlay_causal:
            att = att.masked_fill(
                self.tril[:S, :S] == 0, float('-inf'))  # type: ignore

        # TODO: get an empty mask for all tags not occurring in the
        # mask argument; then stack all masks on dim 0 and fill masks

        if masks is not None:
            mask_list: list[torch.Tensor] = []
            for tag in tags:
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

        if self.attn_dropout > 0 and self.training:
            att[..., 2:][torch.rand(
                att.shape, device=att.device)[..., 2:]
                < self.attn_dropout] = float('-inf')

        att_logits = att

        att = F.softmax(att, dim=-1)
        att = einops.rearrange(att, 'b m h s1 s2 -> b (m h) s1 s2')

        out = torch.matmul(att, v)

        # reassamble all head outputs side by side
        out = einops.rearrange(out, 'b mh s e -> b s (mh e)')

        # output projection
        out = self.proj(out)
        out = self.resid_dropout(out)

        out_logits: None | dict[str, list[torch.Tensor]] = None
        if return_arc_logits:
            out_logits = {
                tag: [att_logits[:, m, h] for h in range(self.n_head)]
                for tag, m in zip(tags, range(self.n_multihead))}

        additional_output: AdditionalResults = {}
        if return_att:
            additional_output["att"] = att

        if return_proj_state_norms:
            # adapted from Goro Kobayashi
            # (https://github.com/gorokoba560/norm-analysis-of-transformer)
            v_layer = v.permute(0, 2, 1, 3).contiguous().unsqueeze(3)
            # b mh s e -> b s mh 1 e

            # dense weight is converted to
            # (num_heads, head_size, all_head_size)
            output_weight = self.proj.weight.view(H*M, E, H*M*E)
            # mhe mhe -> mh e mhe

            # create transformed vectors f(x) from value vectors (value_layer)
            # and weight matrix (output_weight).
            # the bias of the output transformation is assumed to be
            # distributed equally among heads
            projected_states = v_layer.matmul(output_weight).squeeze(3)
            # (b s mh 1 e) x (mh e mhe) -> b s mh mhe

            projected_states = projected_states.permute(
                0, 2, 1, 3).contiguous()
            if self.proj.bias is not None:
                projected_states += (self.proj.bias / H*M)
            # (b mh s mhe)
            projected_states = torch.einsum(
                "bhks,bhsd->bhksd", att, projected_states)  # (b mh s s mhe)

            additional_output["proj_state_norms"] = projected_states

        return out, out_logits, additional_output


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
        self.attn: MIAttention | Attention
        if len(layer_description[0]) == 1:
            self.attn = Attention(
                n_embd, ((layer_description[0][0],), layer_description[1]),
                block_size, attn_dropout,
                resid_dropout, overlay_causal,
                use_dual_fixed)
        else:
            self.attn = MIAttention(n_embd, layer_description,
                                    block_size, attn_dropout,
                                    resid_dropout, overlay_causal,
                                    use_dual_fixed)
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        self.ff = FeedForward(n_embd, d_ff_factor, dropout_ff, bias)

        self.forward_mask: bool = len(layer_description[0]) != 1

    def forward(
            self,
            x: torch.Tensor,
            masks: dict[str, torch.Tensor | None],
            return_arc_logits: bool = False,
            return_proj_state_norms: bool = False,
            return_att: bool = False
            ) -> tuple[
                torch.Tensor,
                None | dict[str, list[torch.Tensor]],
                AdditionalResults]:
        """Mask shape: [M, B, S, S]
        with M number of multiheads,
        B batch size, S sequence length"""

        # this is necessary since the simple attention module does
        # not support custom masks
        if self.forward_mask:
            x_attn, out_logits, additional = self.attn(
                self.ln_1(x), masks,
                return_arc_logits=return_arc_logits,
                return_proj_state_norms=return_proj_state_norms,
                return_att=return_att)
        else:
            x_attn, out_logits, additional = self.attn(
                self.ln_1(x),
                return_arc_logits=return_arc_logits,
                return_proj_state_norms=return_proj_state_norms,
                return_att=return_att)
        x = x + x_attn
        x = x + self.ff(self.ln_2(x))

        return x, out_logits, additional


TransformerDescription = tuple[LayerDescription, ...]


@dataclass
class MITransformerConfig(utils.Params):
    transformer_description: TransformerDescription = (
        (("head_current", "child_current"), 1),)
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
            config.overlay_causal, config.use_dual_fixed)
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
                n_embd, n_embd, batch_first=True)
            self.lstm_dropout = nn.Dropout(config.dropout_lstm)
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
            return_arc_logits: bool = False,
            return_proj_state_norms: bool = False,
            return_att: bool = False,
            return_embeddings: bool = False,
            **kwargs
            ) -> tuple[
                torch.Tensor, None | dict[str, torch.Tensor],
                AdditionalResults]:
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
            tok_emb *= tok_emb.shape[-1]**0.5
            # scale to make larger than encodings
            pos_emb = self.wpe(S)

        x = self.embd_dropout(tok_emb + pos_emb)
        del pos_emb

        if self.lstm is not None:
            # x = self.lstm(x)[0]
            x = x + self.lstm_dropout(self.lstm(self.ln_1(x))[0])
            x = x + self.ff(self.ln_2(x))

        att_logits = []
        additional_list: list[AdditionalResults] = []
        for layer in self.layers:
            x, al, additional = layer(
                x, masks if self.use_input_mask else None,
                return_arc_logits=return_arc_logits,
                return_proj_state_norms=return_proj_state_norms,
                return_att=return_att)
            if al is not None:
                att_logits.append(al)
            additional_list.append(additional)

        additional_names: list[AdditionalKeys] = []
        if return_att:
            additional_names.append("att")
        if return_proj_state_norms:
            additional_names.append("proj_state_norms")

        additional_stacked: AdditionalResults = {}
        for key in additional_names:  # type: ignore
            stacked = torch.stack(
                [additional[key]      # type: ignore
                    for additional in additional_list])
            additional_stacked[key] = stacked  # type: ignore

        out_logits = None
        if return_arc_logits:
            out_logits = combine_scores(att_logits)

        if return_embeddings:
            additional_stacked["embeddings"] = tok_emb

        return x, out_logits, additional_stacked


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
            masks: dict[str, torch.Tensor | None] | None = None,
            return_arc_logits: bool = False,
            return_proj_state_norms: bool = False,
            return_att: bool = False,
            return_embeddings: bool = False,
            return_activations: bool = False,
            **kwargs
            ) -> tuple[
                torch.Tensor, None | dict[str, torch.Tensor],
                AdditionalResults]:
        """
            Input:
            x : [B, S, E]
            masks : dictionary mapping from tags to masks
            mask shape: [B, S, S]
            masks should be a boolean mask with True being the elements not
            to mask and False being the entries to mask.

            Output : Shape[B, S, E]
        """

        x, att_logits, additional = self.mi_transformer(
            input_ids, masks,
            return_arc_logits=return_arc_logits,
            return_proj_state_norms=return_proj_state_norms,
            return_att=return_att,
            return_embeddings=return_embeddings)

        x = self.ln(x)
        logits = self.lm_head(x)

        if return_activations:
            additional["activations"] = x

        return logits, att_logits, additional

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **kwargs):
        """ given a context idx, generate max_new_tokens tokens
        and append them to idx """
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.mi_transformer.block_size:]
            # we can never have any idx longer than block_size
            logits = self(idx_cond, {
                "head_current": None, "child_current": None})[0]
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
        layer_design: tuple[str, ...] = ("head_current", "child_current"),
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
