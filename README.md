# DistributedTrainingDemos

本项目目前包含 5 个 demo ，具体内容如下：

1. 基于 PyTorch DistributedDataParallel 类实现 transformer 模型的数据并行训练

2. 基于 PyTorch pipeline 方法等，实现 transformer 模型的流水线并行训练

3. 基于 demo1 和 demo2 实现一个简单的 DP PP 混合并行训练框架

4. 使用 Megatron-LM 的教学代码，完成 GPT 模型在张量并行下的预训练

5. 使用 huggingface transformers 库和 deepspeed ，实现 Qwen2.5-7B-Instruct 模型在两张 V100-32G 显卡下的微调，微调使用 deepseek-r1 蒸馏数据集
