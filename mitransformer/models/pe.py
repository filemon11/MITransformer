
import torch
import torch.nn as nn

import math


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
        x = torch.cat((self.begin_emb, self.pe[:len-2])).requires_grad_(False)
        # do not positionally encode dummy and root
        return self.dropout(x).unsqueeze(0)

