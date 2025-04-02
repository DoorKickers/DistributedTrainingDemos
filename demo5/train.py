from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

dataset_path = "/nvme/zhanglantian/project/DistributedTrainingDemos/demo5/Chinese-DeepSeek-R1-Distill-data-110k-SFT"
dataset = load_dataset(dataset_path, split="train")
dataset = dataset.shuffle(seed=42).select(range(10))

tokenizer = AutoTokenizer.from_pretrained("/nvme/zhanglantian/project/DistributedTrainingDemos/recycle/Qwen2.5-0.5B")
 
def generate_r1_prompt(prompt, completion):
    input_ids, attention_mask, labels = [], [], []
    instruction = [
        {
            "role": "system",
            "content": "TEST"
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

    tokenized_instruction = tokenizer.apply_chat_template(instruction, tokenize=True, return_dict=True)
    tokenized_response = tokenizer.apply_chat_template(response, tokenize=True, return_dict=True)

    input_ids = tokenized_instruction["input_ids"] + tokenized_response["input_ids"]
    attention_mask = tokenized_instruction["attention_mask"] + tokenized_response["attention_mask"]
    labels = [-100] * len(tokenized_instruction["input_ids"]) + tokenized_response["input_ids"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
 
dataset = dataset.map(lambda x: generate_r1_prompt(x["instruction"], x["output"]), remove_columns=["instruction", "output"])

model = AutoModelForCausalLM.from_pretrained("/nvme/zhanglantian/project/DistributedTrainingDemos/recycle/Qwen2.5-0.5B")

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