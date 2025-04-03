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
 
system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
def generate_r1_prompt(prompt, completion):
    input_ids, attention_mask, labels = [], [], []
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
    response = [
        {
            "role": "assistant",
            "content": completion    
        }
    ]

    full = instruction + response

    tokenized_instruction = tokenizer.apply_chat_template(instruction, tokenize=True, return_dict=True)
    tokenized_full = tokenizer.apply_chat_template(full, tokenize=True, return_dict=True)

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]
    labels = input_ids.copy()
    instruction_length = len(tokenized_instruction["input_ids"])
    labels[:instruction_length] = [-100] * instruction_length
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
 
dataset = dataset.map(lambda x: generate_r1_prompt(x["instruction"], x["output"]), remove_columns=["instruction", "output"])

print(tokenizer.decode(dataset[0]["input_ids"]))

print(tokenizer.decode(list(filter(lambda x: x != -100, dataset[0]["labels"]))))

model = AutoModelForCausalLM.from_pretrained(model_path)

training_args = TrainingArguments(
    output_dir="./fine_tuned_qwen",
    per_device_train_batch_size=1,
    num_train_epochs=10,
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=1,
    evaluation_strategy="no",
    save_total_limit=1,
    deepspeed="deepspeed_config.json",
    fp16=True,
    gradient_checkpointing=True,
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