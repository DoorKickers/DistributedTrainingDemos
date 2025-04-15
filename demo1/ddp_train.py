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
    # 设置模型和训练的超参数
    VOCAB_SIZE = 100        # 词汇表大小
    D_MODEL = 12            # 嵌入维度 / 模型维度
    NUM_HEADS = 4           # 注意力头数量
    D_FF = 24               # 前馈层的中间维度
    NUM_LAYERS = 2          # Transformer 层数
    MAX_LEN = 100           # 输入序列的最大长度

    DATASET_SIZE = 12       # 样本数量（总数据条数）
    DATASET_LENGTH = 10     # 每条样本的长度
    BATCH_SIZE = 2          # 每个进程的 batch 大小

    # 创建自定义数据集
    dataset = NLPDataset(size=DATASET_SIZE, length=DATASET_LENGTH)

    # 为分布式训练设置采样器，每个 rank（进程）处理自己的一份数据
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    # 使用带分布式采样器的数据加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # 设置后端：GPU 可用就使用 nccl，否则使用 gloo（CPU）
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # 初始化当前进程的分布式环境
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # 实例化 Transformer 模型并移动到对应设备
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
    ).to(device)

    # 将模型封装为分布式模型（DDP）
    # DDP 会自动将模型 warp 成一个数据并行的形式，这样不同的进程在执行 forward 的时候，DDP 会自动进行数据并行的相关处理，例如反向传播的时候会自动同步梯度
    # DDP 的初始化需要传入整个进程通信组的相关信息，例如有多少个进程参与此次训练，每个进程的通信地址是什么，从而建立卡间通信网络
    # device_ids 对应当前进程使用的 GPU 编号，从而保证不同进程的模型放到不用的 GPU 上。
    # DistributedSampler 确保 DDP 模型在不同进程上获得的数据不同，反向传播以后，每个进程保存的模型梯度是不同的，这时根据之前建立的进程组信息，DDP 会自动将不同进程的模型梯度同步，同步以后结果等于执行一次 batch_size 为 DP 组数的训练。因此 global_batch_size = num_dp_group * batch_size_per_group
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    # 使用交叉熵作为损失函数
    criterion = nn.CrossEntropyLoss()

    # 使用 SGD 作为优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 开始训练循环
    for epoch in range(200):
        for batch, data in enumerate(dataloader):
            # 目标是当前句子的下一位
            label = data[:, 1:].to(device)       # 标签是输入右移一位
            data = data[:, :-1].to(device)       # 输入去掉最后一个 token

            optimizer.zero_grad()                # 清空梯度
            outputs = model(data)                # 前向传播

            # 计算损失：输出和标签都 reshape 成二维，适配交叉熵
            loss = criterion(outputs.view(-1, VOCAB_SIZE), label.view(-1))

            loss.backward()                      # 反向传播
            optimizer.step()                     # 更新参数

            # 只在 rank==0 的主进程上打印日志，避免重复
            if rank == 0:
                print(f"Epoch {epoch}, Batch {batch}, Loss: {loss.item()}")

    # 训练完成后销毁分布式进程组
    dist.destroy_process_group()



if __name__ == "__main__":
    train(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))
