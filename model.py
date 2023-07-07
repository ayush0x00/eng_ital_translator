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

        # apply sin to even pos and cos to odd pos
        positional_encoding[:, 0::2] = torch.sin(pos * div_term)
        positional_encoding[:, 1::2] = torch.cos(pos * div_term)

        # adding batch dim
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer("pe", positional_encoding)

    def forward(self, x):  # pe have batch size as the first dim. (1,seq_len,d_model)
        x = x + (self.positional_encoding[:, : x.shape[1], :]).requires_grad(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # multiplied while layer normalizing
        self.beta = nn.Parameter(torch.zeros(1))  # added while layer normalizing

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # IP and OP both dim are (1,seq_len,d_model)
        # inner layer dim is (1,seq_len,d_ff)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MulthiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "Head not divisible"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(
            d_k
        )  # (batch,h,seq_len,seq_len)

        # apply mask before softmax

        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch,h,seq_len,seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # before applying softmax , we mask the values close to 0. Then we apply softmax and multiply it with V
        # to get attention score

        query = self.w_q(
            q
        )  # (batch,seq_len,d_model)---> (batch,seq_len,d_model), basically Q'
        key = self.w_k(
            k
        )  # (batch,seq_len,d_model)---> (batch,seq_len,d_model), basically K'
        value = self.v(
            v
        )  # (batch,seq_len,d_model)---> (batch,seq_len,d_model), basically V'

        # divide query, key, value into smaller matrices for different heads

        # we want each head to see the full sequence so we have to take transpose
        # (batch,seq_len,d_model)-->(batch,seq_len,h,d_k)--->(batch,h,seq_len,d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attention_score = MulthiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )
        # (batch,h,seq_len,d_k) ---> #(batch,seq_len,h,d_k) --> #(batch,seq_len,d_model)
        # refer here for use of contiguous https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], x.shape[1], self.h * self.d_k)
        )

        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = dropout
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # slight different from paper

    # here we first apply add&norm and the apply sublayer (MHA/FeedForward)
    # but in paper first sublayer is applied and then normalization


class EncoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MulthiHeadAttention,
        feed_forward_block: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        self_attention_block: MulthiHeadAttention,
        cross_attention_block: MulthiHeadAttention,
        feed_forward_block: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(
        self, x, encoder_output, src_mask, target_mask
    ):  # src mask for English and target mask for Italian
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, target_mask)
        )  # self attention in decoder
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch,seq_len,d_model)-->(batch,seq_len,vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        target_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        target_pos: PositionalEncoding,
        proj_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.target_embed(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x):
        return self.proj_layer(x)


def build_transformer(
    src_vocab_size: int,
    target_vocab_size: int,
    src_seq_len: int,
    target_seq_len: int,
    d_model: int = 512,
    n: int = 6,  # number of encoder decoder blocks
    h: int = 8,
    dropout: float = 0.1,
    d_ff=2048,
) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    target_embed = InputEmbeddings(d_model, target_embed)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, src_seq_len, dropout)

    encoder_blocks = []
    for _ in range(n):
        encoder_self_attention = MulthiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention, feed_forward_block, dropout
        )
        encoder_block.append(encoder_block)

    decoder_blocks = []
    for _ in range(n):
        decoder_self_attention = MulthiHeadAttention(d_model, h, dropout)
        decoder_cross_attention = MulthiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention, decoder_cross_attention, feed_forward_block, dropout
        )
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    transformer = Transformer(
        encoder, decoder, src_embed, target_embed, src_pos, target_pos, projection_layer
    )

    # intializing params
    for p in transformer.parameters():
        if p.dim() >1:
            nn.init.xavier_uniform_(p)

    return transformer
