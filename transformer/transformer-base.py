import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy, re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
# import torchtext
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')
print(torch.__version__)

BATCH_SIZE = 32
SEQ_LEN = 10
EMB_DIM = 512
HEADS = 8


# Word Embedding
class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        :param x: input vector
        :return: embedding vector
        """
        out = self.embed(x)
        return out


# Positional Encoding
class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        """
        :param max_seq_len: length of input sequence
        :param embed_model_dim: dimension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(max_seq_len):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x:input vector
        :return: output
        """
        x = x * math.sqrt(self.embed_dim)  # for scaling
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        return x


# SelfAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=EMB_DIM, n_heads=HEADS):
        """
        :param embed_dim: dims of embedding vector output
        :param n_heads: num of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim  # 512
        self.n_heads = n_heads  # 8
        self.single_head_dim = int(self.embed_dim / self.n_heads)  # 512/8=64, each QKV will be 64

        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, query, key, value, mask=None):  # batch_size * seq_len * emb_dim = 32 * 10 * 512
        """
        :param query: value vector
        :param key: value vector
        :param value: value vector
        :param mask: mask for decoder
        :return: output vector from multi-head attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)

        # since the query dimension can change in Decoder during inference so can't take general seq_len
        seq_length_query = query.size(1)

        # batch_size * seq_len * n_heads * single_head_dim (32*10*8*64)
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)

        # 32x10x8x64
        k = self.key_matrix(key)
        q = self.query_matrix(query)
        v = self.value_matrix(query)

        # seq_len <-> n_heads = 32*8*10*64
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # compute attention
        k_adjusted = k.transpose(-1, -2)  # seq_len <-> single_head_dim = (32*8*64*10)
        product = torch.matmul(q, k_adjusted)  # (32*8*10*64) * (32*8*64*10) = 32*8*10*10

        # fill those positions with product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        # divising by square root of key dims
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)

        # applying softmax
        scores = F.softmax(product, dim=-1)  # -1 is seq_len
        # applying with value matrix
        scores = torch.matmul(scores, v)  # (32*8*10*10) x (32*8*10*64) = 32*8*10*64
        # concat output: (32*8*10*64) * (32*10*8*64) -> (32*10*512)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_length_query,
                                                          self.single_head_dim * self.n_heads)
        output = self.out(concat)  # (32, 10, 512) -> (32, 10, 512)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        """
        :param embed_dim: dims of the embedding
        :param expansion_factor: factor which determines output dimension of linear layer
        :param n_heads: num of attention heads
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value):
        attention_out = self.attention(key, query, value)  # 32*10*512
        attention_residual_out = attention_out + value  # 32*10*512
        norm1_out = self.dropout1(self.norm1(attention_residual_out))  # 32*10*512

        feed_fwd_out = self.feed_forward(norm1_out)  # 32*10*512 -> 32*10*2048 -> 32*10*512
        feed_fwd_residual_out = feed_fwd_out + norm1_out  # 32*10*512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))  # 32*10*512

        return norm2_out


# Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        """
        :param seq_len: length of input sequence
        :param vocab_size: dims of embedding
        :param embed_dim: num of encoder layers
        :param num_layers: factor which determines number of linear layers in feed forward layer
        :param expansion_factor: num of heads in multihead attention
        :param n_heads: num of heads in multihead attention
        """
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out, out, out)

        return out  # 32*10*512


# Decoder
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        """
        :param embed_dim: dims of embedding
        :param expansion_factor:  factor which determines output dimension of linear layer
        :param n_heads: num of attention heads
        """
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, query, x, mask):
        """
        :param key:
        :param query:
        :param x: value vector
        :param mask: mask to be given for multi head attention
        :return: output of transformer block
        """
        # need to pass mask only for attention
        attention = self.attention(x, x, x, mask=mask)  # 32*10*512
        value = self.dropout(self.norm(attention + x))
        out = self.transformer_block(key, query, value)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        """
        :param target_vocab_size: vocabulary size of target
        :param embed_dim: dim of embedding
        :param seq_len: length of input sequence
        :param num_layers: num of encoder layers
        :param expansion_factor: factor which detemins num of layers in FF layer
        :param n_heads: num of heads in multihead attention
        """
        super(TransformerDecoder, self).__init__()
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):
        """
        :param x: vector for target
        :param enc_out: output from encoder layer
        :param mask: mask for decoder self attention
        :return: output vector
        """
        x = self.word_embedding(x)
        x = self.position_embedding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask)

        out = F.softmax(self.fc_out(x))

        return out


# Transformer Architecture
class Transformer(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size, seq_length, num_layers=2, expansion_factor=4,
                 n_heads=8):
        """
        :param embed_dim: dimension of embedding
        :param src_vocab_size: vocab size of source
        :param target_vocab_size: vocab size of target
        :param seq_length: length of input sequence
        :param num_layers: num of encoder layers
        :param expansion_factor: factor which determines number of linear layers in FF layer
        :param n_heads: num of hads in multihead attention
        """
        super(Transformer, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers,
                                          expansion_factor=expansion_factor, n_heads=n_heads)
        self.docoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers,
                                          expansion_factor=expansion_factor, n_heads=n_heads)

    def make_trg_mask(self, trg):
        """
        :param trg: target sequence
        :return: target mask
        """
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def decode(self, src, trg):
        """
        :param src: input to encoder
        :param trg: input to decoder
        :return: final prediction of sequence out_labels
        """

        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size, seq_len = src.shape[0], src.shape[1]
        # outputs = torch.zeros(seq_len, batch_size, self.target_vocab_size)
        out = trg
        for i in range(seq_len):  # 10
            out = self.decoder(out, enc_out, trg_mask)  # bs * seq_len * vocab_dim
            # taking the last token
            out = out[:, -1, :]
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, axies=0)

        return out_labels

    def forward(self, src, trg):
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)

        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs

src_vocab_size = 11
target_vocab_size = 11
num_layers = 6
seq_length= 12


# let 0 be sos token and 1 be eos token
src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1],
                    [0, 2, 8, 7, 3, 4, 5, 6, 7, 2, 10, 1]])
target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1],
                       [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])

print(src.shape,target.shape)
model = Transformer(embed_dim=512, src_vocab_size=src_vocab_size,
                    target_vocab_size=target_vocab_size, seq_length=seq_length,
                    num_layers=num_layers, expansion_factor=4, n_heads=8)
print(model)


# out = model(src, target)
# out.shape
#
# # inference
# model = Transformer(embed_dim=512, src_vocab_size=src_vocab_size,
#                     target_vocab_size=target_vocab_size, seq_length=seq_length,
#                     num_layers=num_layers, expansion_factor=4, n_heads=8)
#
# src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1]])
# trg = torch.tensor([[0]])
# print(src.shape, trg.shape)
# out = model.decode(src, trg)
# out
