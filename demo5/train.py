from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

dataset_path = "/nvme/zhanglantian/project/DistributedTrainingDemos/demo5/Chinese-DeepSeek-R1-Distill-data-110k-SFT"
dataset = load_dataset(dataset_path, split="train")
dataset = dataset.shuffle(seed=42).select(range(10))


model_path = "/nvme/models/models/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
 
# 定义系统提示词，一般用于设定模型身份或行为风格
system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# 构造训练样本的函数，用于从 prompt 和 completion 构造 input_ids, attention_mask 和 labels
def generate_r1_prompt(prompt, completion):
    input_ids, attention_mask, labels = [], [], []

    # 构造指令部分：系统设定和用户输入
    instruction = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        },
    ]

    # 构造响应部分：模型的回答（目标输出）
    response = [
        {
            "role": "assistant",
            "content": completion    
        }
    ]

    # 将 instruction 和 response 拼接成完整的对话
    full = instruction + response

    # 对 instruction（系统+用户）进行分词，用于计算标签掩码（仅训练 assistant 的回复）
    tokenized_instruction = tokenizer.apply_chat_template(
        instruction, tokenize=True, return_dict=True
    )

    # 对完整对话进行分词，生成 input_ids 和 attention_mask
    tokenized_full = tokenizer.apply_chat_template(
        full, tokenize=True, return_dict=True
    )

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]

    # 初始化标签为 input_ids 的副本（后续根据需要掩盖一部分）
    labels = input_ids.copy()

    # 只训练 assistant 的回复，因此将 instruction 部分的标签设为 -100，表示不参与 loss 计算
    instruction_length = len(tokenized_instruction["input_ids"])
    labels[:instruction_length] = [-100] * instruction_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 使用 map 操作对整个数据集进行处理，调用 generate_r1_prompt 来转换格式
# 并删除原始字段 instruction 和 output，只保留训练所需的 input_ids, attention_mask, labels
dataset = dataset.map(
    lambda x: generate_r1_prompt(x["instruction"], x["output"]),
    remove_columns=["instruction", "output"]
)


print(tokenizer.decode(dataset[0]["input_ids"]))

print(tokenizer.decode(list(filter(lambda x: x != -100, dataset[0]["labels"]))))

model = AutoModelForCausalLM.from_pretrained(model_path)

training_args = TrainingArguments(
    output_dir="./fine_tuned_qwen",               # 模型输出目录
    per_device_train_batch_size=1,                # 每张GPU的训练batch大小
    num_train_epochs=10,                          # 总训练轮数
    save_strategy="no",                           # 不保存中间checkpoint
    logging_dir="./logs",                         # 日志保存路径
    logging_steps=1,                              # 每训练1步记录一次日志
    evaluation_strategy="no",                     # 不进行评估
    save_total_limit=1,                           # 最多保留1个checkpoint
    deepspeed="deepspeed_config.json",            # 指定deepspeed配置文件
    fp16=True,                                    # 使用混合精度训练（float16）
    gradient_checkpointing=True                   # 启用梯度检查点，节省显存
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()

test_model = trainer.model_wrapped
prompt = "1.11和1.8哪个大"
inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       ).to("cuda")

gen_kwargs = {"max_length": 200, "do_sample": True, "top_k": 1}

with torch.no_grad():
    outputs = test_model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    if os.environ["RANK"] == "0":
        print("微调后模型推理结果：\n", tokenizer.decode(outputs[0], skip_special_tokens=True))