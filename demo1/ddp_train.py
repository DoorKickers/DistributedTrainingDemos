import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]

        q = (
            self.q_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(x)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e9"))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )
        return self.out_linear(output)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        max_len=5000,
    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.out_linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.out_linear(x)


class NLPDataset(Dataset):
    def __init__(self, size, length):
        self.data = []
        for i in range(size):
            self.data.append(torch.full((length,), i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


def train(rank, world_size):
    VOCAB_SIZE = 100
    D_MODEL = 12
    NUM_HEADS = 4
    D_FF = 24
    NUM_LAYERS = 2
    MAX_LEN = 100

    DATASET_SIZE = 12
    DATASET_LENGTH = 10
    BATCH_SIZE = 2

    dataset = NLPDataset(size=DATASET_SIZE, length=DATASET_LENGTH)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
    ).to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(200):
        for batch, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), data.view(-1))
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"Epoch {epoch}, Batch {batch}, Loss: {loss.item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    train(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
