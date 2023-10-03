import torch
import torch.nn as nn
from torch.nn import functional


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGram, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embed_size)
        self.out_embeddings = nn.Embedding(vocab_size, embed_size)

    def forward(self, target):
        in_vec = self.in_embeddings(target)
        out_vecs = self.out_embeddings.weight
        scores = torch.matmul(in_vec, out_vecs.t())
        return functional.log_softmax(scores, dim=1)

# in_embedding is the final embedding will be used for downstream tasks
