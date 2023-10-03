import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

vocab_size = 10000
embedding_dim = 128
learning_rate = 0.001
batch_size = 64
epochs = 10

# 创建数据加载器
dataset = TensorDataset() # context_ids, target_labels
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # input vocab * emb_dim
        self.linear = nn.Linear(embedding_dim, vocab_size)  # emb_dim -> vocab_size

    def forward(self, context_ids):
        embeddings = self.embeddings(context_ids)
        embeddings = sum(embeddings).view(1, -1)
        output = self.linear(embeddings)
        return output


model = CBOW(vocab_size=vocab_size, embedding_dim=embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    total_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

print("Training complete")
