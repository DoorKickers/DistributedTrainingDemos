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


from torch.distributed.pipelining import pipeline, SplitPoint
from torch.distributed.pipelining import ScheduleGPipe


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


def train(rank, world_size):
    # Transformer模型的基本参数
    VOCAB_SIZE = 100        # 词表大小
    D_MODEL = 12            # 模型的嵌入维度（hidden size）
    NUM_HEADS = 4           # 多头注意力的头数
    D_FF = 24               # 前馈网络维度
    NUM_LAYERS = 2          # Transformer 层数
    MAX_LEN = 100           # 序列最大长度

    # 数据集相关参数
    DATASET_SIZE = 12       # 数据集中样本数量
    DATASET_LENGTH = 10     # 每个样本的序列长度
    BATCH_SIZE = 4          # 每个 batch 的样本数

    NUM_MICROBATCHES = 2    # 用于 GPipe 分割的 microbatch 数

    # 定义损失函数的封装（方便被 GPipe 调用）
    def compute_loss(output, target):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.view(-1, VOCAB_SIZE), target.view(-1))
        return loss

    # 构造 NLP 数据集和 DataLoader
    dataset = NLPDataset(size=DATASET_SIZE, length=DATASET_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # 构造一个输入模板，用于 GPipe pipeline 初始化
    x = torch.zeros((BATCH_SIZE // NUM_MICROBATCHES, DATASET_LENGTH - 1), dtype=torch.long)

    # 构建带有 split_spec 的 GPipe pipeline，将模型划分为多个 stage
    pipe = pipeline(
        module=Transformer(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            d_ff=D_FF,
            num_layers=NUM_LAYERS,
            max_len=MAX_LEN,
        ),
        mb_args=(x,),  # microbatch 的输入形状，用于模型切分
        split_spec={
            "layers.1": SplitPoint.BEGINNING,  # 从 Transformer 第2层开始划分模型
        },
    )

    # 通过环境变量获取当前进程的rank和world size
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 初始化 PyTorch 分布式通信
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 指定当前 rank 使用的 GPU 设备
    device = f"cuda:{rank}"

    # 获取当前 rank 的模型 stage，用于定义 optimizer
    stage_mod = pipe.get_stage_module(rank)
    optimizer = optim.SGD(stage_mod.parameters(), lr=0.01)

    # 构建模型 stage，并设置 device 和通信信息
    stage = pipe.build_stage(rank, device, None)

    # 创建 GPipe 的训练调度器（ScheduleGPipe）
    schedule = ScheduleGPipe(stage, NUM_MICROBATCHES, compute_loss)

    # 正式开始训练过程
    for epoch in range(200):
        for batch, data in enumerate(dataloader):
            # 将输入数据切分为 input 和 target
            label = data[:, 1:].to(device)  # 目标序列（标签）
            x = data[:, :-1].to(device)     # 输入序列（模型输入）

            optimizer.zero_grad()           # 梯度清零

            # 对于 rank 0：只负责前向传输，不计算 loss
            if rank == 0:
                schedule.step(x)
            else:
                # 其他 stage（最后一段）负责 loss 计算和反向传播
                losses = []
                output = schedule.step(target=label, losses=losses)
                print(
                    f"Epoch {epoch}, Batch {batch}, Loss: {torch.stack(losses).mean()}"
                )

            # 优化参数
            optimizer.step()

    # 销毁进程组，结束训练
    dist.destroy_process_group()


if __name__ == "__main__":
    train(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
