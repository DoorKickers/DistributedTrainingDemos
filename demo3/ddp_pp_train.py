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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

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

        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
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
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=5000):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
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
            self.data.append(torch.full((length, ), i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def parse_yaml_config(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_args = {
        'vocab_size': config.get('vocab_size'),
        'max_seq_length': config.get('max_seq_length'),
        'hidden_size': config.get('hidden_size'),
        'feedforward_size': config.get('feedforward_size'),
        'num_heads': config.get('num_heads'),
        'num_layers': config.get('num_layers')
    }

    dataset_args = {
        'dataset_size': config.get('dataset_size'),
        'data_length': config.get('data_length')
    }

    training_args = {
        'train_epochs': config.get('train_epochs'),
        'micro_batch_size': config.get('micro_batch_size'),
        'micro_num': config.get('micro_num'),
        'learning_rate': config.get('learning_rate'),
        'device_type': config.get('device_type')
    }

    training_args['batch_size'] = training_args['micro_num'] * training_args['micro_batch_size']


    parallel_args = {
        'pipeline_parallel_size': config.get('pipeline_parallel_size'),
        'data_parallel_size': config.get('data_parallel_size')
    }

    return model_args, dataset_args, training_args, parallel_args

if __name__ == '__main__':
    config_file = "config.yaml"

def train(rank, world_size, config_file):

    model_args, dataset_args, training_args, parallel_args = parse_yaml_config(config_file)

    def compute_loss(output, target):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output.view(-1, model_args['vocab_size']), target.view(-1))
        return loss


    x = torch.zeros((training_args['micro_batch_size'], dataset_args['data_length']), dtype=torch.long)

    split_spec = {}
    for i in range(parallel_args['pipeline_parallel_size'] - 1):
        layers_id = (model_args['num_layers'] - 1) // parallel_args['pipeline_parallel_size'] * (i + 1)
        split_spec[f'layers.{layers_id}'] = SplitPoint.END

    pipe = pipeline(
        module=Transformer(vocab_size=model_args['vocab_size'], d_model=model_args['hidden_size'], num_heads=model_args['num_heads'], d_ff=model_args['feedforward_size'], num_layers=model_args['num_layers'], max_len=model_args['max_seq_length']),
        mb_args=(x,),
        split_spec=split_spec
    )


    mesh_2d = init_device_mesh(training_args['device_type'], mesh_shape=(parallel_args['data_parallel_size'], parallel_args['pipeline_parallel_size']), mesh_dim_names=("dp", "pp"))

    pp_group = mesh_2d.get_group("pp")
    dp_group = mesh_2d.get_group("dp")

    pp_rank = dist.get_rank(pp_group)
    dp_rank = dist.get_rank(dp_group)

    device = f"cuda:{rank}" if training_args['device_type'] == 'gpu' else "cpu"

    stage_mod = pipe.get_stage_module(pp_rank).to(device)
    print("rank, stage_mod", rank, stage_mod)
    dp_mod = DDP(stage_mod, device_ids=[rank] if not device == "cpu" else None, process_group=dp_group)
    optimizer = optim.SGD(dp_mod.parameters(), lr=training_args['learning_rate'])
    info = pipe.info()
    stage = build_stage(stage_mod, pp_rank, info, device, pp_group)

    schedule = ScheduleGPipe(stage, training_args['micro_num'], compute_loss)
    dataset = NLPDataset(size=dataset_args['dataset_size'], length=dataset_args['data_length'])
    sampler = DistributedSampler(dataset, num_replicas=parallel_args['data_parallel_size'], rank=dp_rank)
    dataloader = DataLoader(dataset, batch_size=training_args['batch_size'], sampler=sampler)
    for epoch in range(training_args['train_epochs']):
        for batch, data in enumerate(dataloader):
            x = data.to(device)
            optimizer.zero_grad()
            if pp_rank == 0:
                schedule.step(x)
            else:
                losses = []
                output = schedule.step(target=x, losses=losses)
                loss = torch.stack(losses).mean()
                dist.all_reduce(loss, op=ReduceOp.SUM, group=dp_group)
                if dp_rank == 0:
                    print(f"Epoch {epoch}, Batch {batch}, Loss: {loss / parallel_args['data_parallel_size']}")
            optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YAML Configuration Parser")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()
    train(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), args.config)
