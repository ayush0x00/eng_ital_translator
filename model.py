import torch
import math
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # as mentioned in the paper


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # we need a matrix of shape (seq_len,d_model) ie. for every sequence there will be a
        # vector of d_model size (basically seq_len x 512)

        positional_encoding = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        #apply sin to even pos and cos to odd pos
        positional_encoding[:,0::2] = torch.sin(pos*div_term)
        positional_encoding[:,1::2] = torch.cos(pos*div_term)

        # adding batch dim
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('pe',positional_encoding)
    
    def forward(self,x): # pe have batch size as the first dim. (1,seq_len,d_model)
        x = x + (self.positional_encoding[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)
