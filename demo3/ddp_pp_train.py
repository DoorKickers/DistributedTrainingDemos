import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler
from torch.distributed import ReduceOp
import math
import yaml
import argparse

from torch.distributed.pipelining import pipeline, SplitPoint, build_stage
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


def parse_yaml_config(file_path):
    # 解析 YAML 配置文件，读取模型、数据集、训练、并行配置参数
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 模型结构参数
    model_args = {
        "vocab_size": config.get("vocab_size"),
        "max_seq_length": config.get("max_seq_length"),
        "hidden_size": config.get("hidden_size"),
        "feedforward_size": config.get("feedforward_size"),
        "num_heads": config.get("num_heads"),
        "num_layers": config.get("num_layers"),
    }

    # 数据集参数
    dataset_args = {
        "dataset_size": config.get("dataset_size"),
        "data_length": config.get("data_length"),
    }

    # 训练参数
    training_args = {
        "train_epochs": config.get("train_epochs"),
        "micro_batch_size": config.get("micro_batch_size"),
        "micro_num": config.get("micro_num"),
        "learning_rate": config.get("learning_rate"),
        "device_type": config.get("device_type"),
    }

    # 由微批数和微批大小计算总 batch size
    training_args["batch_size"] = (
        training_args["micro_num"] * training_args["micro_batch_size"]
    )

    # 并行参数
    parallel_args = {
        "pipeline_parallel_size": config.get("pipeline_parallel_size"),
        "data_parallel_size": config.get("data_parallel_size"),
    }

    return model_args, dataset_args, training_args, parallel_args


def train(rank, world_size, config_file):
    model_args, dataset_args, training_args, parallel_args = parse_yaml_config(config_file)

    # 定义 loss 函数
    def compute_loss(output, target):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.view(-1, model_args["vocab_size"]), target.view(-1))
        return loss

    # 创建一个微批的输入占位符（用于推理管道构建）
    x = torch.zeros(
        (training_args["micro_batch_size"], dataset_args["data_length"] - 1),
        dtype=torch.long,
    )

    # 构建流水线切割点，按层划分阶段
    split_spec = {}
    for i in range(parallel_args["pipeline_parallel_size"] - 1):
        # 按等间隔方式划分 Transformer 层的切割点
        layers_id = (
            (model_args["num_layers"] - 1)
            // parallel_args["pipeline_parallel_size"]
            * (i + 1)
        )
        split_spec[f"layers.{layers_id}"] = SplitPoint.END

    # 构建流水线模型，进行 pipeline 切分
    pipe = pipeline(
        module=Transformer(
            vocab_size=model_args["vocab_size"],
            d_model=model_args["hidden_size"],
            num_heads=model_args["num_heads"],
            d_ff=model_args["feedforward_size"],
            num_layers=model_args["num_layers"],
            max_len=model_args["max_seq_length"],
        ),
        mb_args=(x,),
        split_spec=split_spec,
    )

    # 初始化 2D 并行设备网格（data 并行 x pipeline 并行）
    # 例如，4 张 GPU 进行训练，编号为 0 1 2 3，一种 2D 并行方法为：GPU 0 1 为一个DP组，两张卡负责存储完整的模型权重，其中 0 号卡负责前半部分模型，1 号卡后半部分模型。 2 3 号卡同理。
    # 这样 4 张 GPU 便可以存储两个完整的模型权重，每张 GPU 需要模型权重一半的显存。在节省显存的同时，因为同时处理两个 batch 的数据，也提高了训练的速度。
    # device_mesh 即定义了这种二维设备关系，例如哪两个卡属于同一个 DP 组。
    mesh_2d = init_device_mesh(
        training_args["device_type"],
        mesh_shape=(
            parallel_args["data_parallel_size"],
            parallel_args["pipeline_parallel_size"],
        ),
        mesh_dim_names=("dp", "pp"),
    )

    # 获取当前进程在 pipeline 和 data 并行中的 rank
    pp_group = mesh_2d.get_group("pp")
    dp_group = mesh_2d.get_group("dp")

    # pp rank 指的是该进程负责保存模型前半部分还是后半部分
    # dp rank 指的是该进程属于哪个 dp 组，相同 dp 组内的卡共同保存模型的权重
    pp_rank = dist.get_rank(pp_group)
    dp_rank = dist.get_rank(dp_group)

    # 设置当前 rank 使用的设备
    device = f"cuda:{rank % 2}" if training_args["device_type"] == "gpu" else "cpu"

    # 获取该 stage 的模型模块，并移动到设备
    stage_mod = pipe.get_stage_module(pp_rank).to(device)
    print("rank, stage_mod", rank, stage_mod)

    # 使用 DDP 包装当前模块，支持数据并行
    dp_mod = DDP(
        stage_mod,
        device_ids=[rank] if not device == "cpu" else None,
        process_group=dp_group,
    )

    # 构造优化器
    optimizer = optim.SGD(dp_mod.parameters(), lr=training_args["learning_rate"])

    # 构建流水线执行阶段
    info = pipe.info()
    stage = build_stage(stage_mod, pp_rank, info, device, pp_group)

    # 构建流水线调度器（GPipe），支持前后向分段流水并行
    schedule = ScheduleGPipe(stage, training_args["micro_num"], compute_loss)

    # 构建数据集和分布式采样器
    dataset = NLPDataset(
        size=dataset_args["dataset_size"], length=dataset_args["data_length"]
    )
    sampler = DistributedSampler(
        dataset, num_replicas=parallel_args["data_parallel_size"], rank=dp_rank
    )
    dataloader = DataLoader(
        dataset, batch_size=training_args["batch_size"], sampler=sampler
    )

    # 训练主循环
    # 在训练的主循环里，不仅用到了 DP ，也用到了 PP 。
    # 对于 DP，我们需要维护一个 DistributedSampler，对于 PP 我们有Pipeline Schedular。
    # DDP 将模型 warp 以后，在反向传播以后会自动同步梯度，而反向传播的具体过程，又由 schedular 提供，因为 torch 的工程实现，我们组合两种并行方法变得较为简单。
    for epoch in range(training_args["train_epochs"]):
        for batch, data in enumerate(dataloader):
            label = data[:, 1:].to(device)
            x = data[:, :-1].to(device)

            optimizer.zero_grad()

            if pp_rank == 0:
                # 第一个阶段只进行前向传播
                schedule.step(x)
            else:
                # 最后一个阶段进行前向 + 反向传播，并收集 loss
                losses = []
                output = schedule.step(target=label, losses=losses)
                loss = torch.stack(losses).mean()

                # 对 loss 进行 data-parallel 级别的 all-reduce
                dist.all_reduce(loss, op=ReduceOp.SUM, group=dp_group)
                if dp_rank == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch}, Loss: {loss / parallel_args['data_parallel_size']}"
                    )

            optimizer.step()

    # 销毁进程组，释放资源
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YAML Configuration Parser")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    args = parser.parse_args()
    train(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args.config)
