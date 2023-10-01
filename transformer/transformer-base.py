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


# Encoder
class TransformerBlock(nn.Module):
    pass
