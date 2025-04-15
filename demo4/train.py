import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path


import sys

sys.path.append("./Megatron-LM")

from megatron.core import parallel_state
from megatron.core import dist_checkpointing
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.datasets.utils import compile_helpers
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training.tokenizer.tokenizer import _NullTokenizer


_SEQUENCE_LENGTH = 64


def initialize_distributed(
    tensor_model_parallel_size=1, pipeline_model_parallel_size=1
):
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )


def model_provider():
    """Build the model."""

    transformer_config = TransformerConfig(
        num_layers=10,
        hidden_size=512,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )

    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=100,
        max_sequence_length=_SEQUENCE_LENGTH,
    )

    return gpt_model


def get_train_data_iterator():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=_SEQUENCE_LENGTH),
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=128, shuffle=True)

    train_iterator = iter(train_dataloader)

    return train_iterator


# 定义 forward step 函数，用于执行一次前向传播
def forward_step_func(data_iterator, model):

    # 定义损失函数
    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # 只对有 label 的位置计算损失
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # 返回 loss 和一个带有命名的字典（便于日志记录等）
        return loss, {'lm loss': loss}

    # 从数据迭代器中取出一批数据，并放到指定设备上
    data = next(data_iterator)
    tokens = data['tokens'].to(device)
    attention_mask = data['attention_mask'].to(device)
    position_ids = data['position_ids'].to(device)
    labels = data['labels'].to(device)
    loss_mask = data['loss_mask'].to(device)

    # 调用模型进行前向传播，返回输出张量
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    # 返回输出和部分绑定好的 loss 函数（只差 output_tensor）
    return output_tensor, partial(loss_func, loss_mask)


# 保存分布式模型 checkpoint
def save_distributed_checkpoint(checkpoint_path, gpt_model):
    # 获取模型的分片 state_dict
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    # 保存分布式分片 checkpoint
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)


# 加载分布式模型 checkpoint
def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)
    # 加载参数到模型中
    gpt_model.load_state_dict(checkpoint)
    return gpt_model


if __name__ == "__main__":
    # 初始化分布式训练，设置张量并行大小为 2，流水线并行为 1（即不使用流水线并行）
    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    # 设置模型并行随机种子，保证不同设备上初始化一致
    model_parallel_cuda_manual_seed(123)

    # 初始化模型
    gpt_model = model_provider()

    # 将模型放到 CUDA 设备上
    device = torch.device("cuda")
    gpt_model.to(device)

    # 使用 Adam 优化器
    optim = Adam(gpt_model.parameters())

    # 获取训练数据迭代器（每次返回一批数据）
    train_iterator = get_train_data_iterator()

    # 获取前向和反向传播封装函数
    forward_backward_func = get_forward_backward_func()

    # 训练 200 次迭代
    for _ in range(200):
        optim.zero_grad()

        # 前向 + 反向传播，支持多 microbatch 的训练
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,  # 前向函数
            data_iterator=train_iterator,        # 数据迭代器
            model=gpt_model,                     # 模型
            num_microbatches=1,                  # microbatch 数量
            seq_length=_SEQUENCE_LENGTH,         # 输入序列长度
            micro_batch_size=32,                 # 每个 microbatch 的 batch size
            decoder_seq_length=_SEQUENCE_LENGTH, # decoder 长度（通常等于输入）
            forward_only=False,                  # 训练模式（非推理）
        )

        optim.step()

        print(f"Losses reduced :  {losses_reduced}")

    # 保存模型 checkpoint 到 ckpt 目录
    ckpt_path = os.getcwd() + "/ckpt"
    Path(ckpt_path).mkdir(exist_ok=True)
    save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)

    # 加载保存的模型
    gpt_model = load_distributed_checkpoint(
        gpt_model=gpt_model, checkpoint_path=ckpt_path
    )
    gpt_model.to(device)
    print("Successfully loaded the model")
