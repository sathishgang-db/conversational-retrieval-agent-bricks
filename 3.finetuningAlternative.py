# Databricks notebook source
# MAGIC %md
# MAGIC In this notebook, we'll explore how easy it is to finetune LLMs via a quick demo, so you can have your own model that works well with your data!

# COMMAND ----------

# MAGIC %md 
# MAGIC Install all the necessary packages

# COMMAND ----------

# MAGIC %pip install --upgrade git+https://github.com/huggingface/transformers
# MAGIC %pip install bitsandbytes
# MAGIC %pip install auto-gptq
# MAGIC %pip install --upgrade git+https://github.com/huggingface/trl
# MAGIC %pip install --upgrade git+https://github.com/huggingface/peft.git
# MAGIC %pip install --upgrade git+https://github.com/huggingface/accelerate.git
# MAGIC %pip install optimum
# MAGIC %pip install --upgrade 'urllib3<2'
# MAGIC %pip install jsonformer

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from dotenv import load_dotenv
load_dotenv(".env")
from huggingface_hub import login
from rich import print
import os
login(token=os.getenv("HUGGINGFACE_TOKEN"))
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from dotenv import load_dotenv

# COMMAND ----------

df = pd.read_csv('data/updated_data_reasons.csv')
df.columns = ['question','category','reason','action','response']
df.head()

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Add 'train' and 'test' labels to the training and test sets, respectively
train_df['label'] = 'train'
test_df['label'] = 'test'

# Combine the training and test DataFrames back into a single DataFrame
cdf = pd.concat([train_df, test_df], ignore_index=True)

# COMMAND ----------

cdf.category.value_counts()

# COMMAND ----------

def prepare_dataset(df, label="train"):
    text_col = []
    instruction = """Classify the following piece of text into one of the following categories.Only respond with one of the categories:fantasy, frankenstein, political, other, banter"""  # change instuction according to the task
    if label == "train":
        for _, row in df.iterrows():
            input_q = row["question"]
            output = row["response"]
            text = (
                "### Instruction: \n"
                + instruction
                + "\n### Input: \n"
                + input_q
                + "\n### Response :\n"
                + output
                + "\n### End"
            )  # keeping output column in training dataset
            text_col.append(text)
        df.loc[:, "text"] = text_col
    else:
        for _, row in df.iterrows():
            input_q = row["question"]
            text = (
                "### Instruction: \n"
                + instruction
                + "\n### Input: \n"
                + input_q
                + "\n### Response :\n"
            )  # not keeping output column in test dataset
            text_col.append(text)
        df.loc[:, "text"] = text_col
    return df

# COMMAND ----------

train_df = prepare_dataset(train_df, "train")
test_df = prepare_dataset(test_df, "test")
train_df

# COMMAND ----------

print(train_df.iloc[1].text)

# COMMAND ----------

from datasets import Dataset
import torch
dataset = Dataset.from_pandas(train_df)
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
model_name = "mistralai/Mistral-7B-v0.1"

# COMMAND ----------

# model_name = "meta-llama/Llama-2-7b-hf"

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

# loading the model with quantization config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map=torch.device("cuda:0"),
)
# model.config.use_cache = True
# apparently needed because of
# https://github.com/huggingface/transformers/pull/24906
#disable tensor parallelism
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, return_token_type_ids=False
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# COMMAND ----------

from peft import LoraConfig, get_peft_model
import transformers

lora_alpha = 16
lora_dropout = 0.05
lora_r = 8  # rank

# Parameter efficient finetuning for LoRA configuration

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=[
        "q_proj",
        "v_proj",
    ],  # we will only create adopters for q, v metrices of attention module
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

output_dir = "mistral_qlora_finetuned_7b"
training_arguments = transformers.TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=1,
    max_steps=100,
    fp16=True,
    push_to_hub=False,
    ddp_find_unused_parameters=False # this has to do with MLR.
)
# %%
# creating trainer with the training agruments
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)

# COMMAND ----------

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,  # passing peft config
    dataset_text_field="text",  # mentioned the required column
    args=training_arguments,  # training agruments
    tokenizer=tokenizer,  # tokenizer
    packing=False,
    max_seq_length=512,
    neftune_noise_alpha=5
)

# COMMAND ----------

from time import perf_counter
start_time = perf_counter()
trainer.train()
end_time = perf_counter()
training_time = end_time - start_time
print(f"Time taken for training: {training_time} seconds")

# COMMAND ----------

checkpoint_name = "final_checkpoint-mistral7b-ft"
# to merge and save the model
output_dir = os.path.join(output_dir, checkpoint_name)
trainer.save_model(output_dir)

# COMMAND ----------

# DBTITLE 1,If you want to just use the model for local inference at a later time...
from peft import AutoPeftModelForCausalLM
import torch,os
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
output_dir = os.path.join("mistral_qlora_finetuned_7b", "final_checkpoint-mistral7b-ft")
persisted_model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cuda",
)
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, return_token_type_ids=False
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# COMMAND ----------

text = test_df.iloc[7]['text']
inputs = tokenizer(text, return_tensors="pt").to("cuda")
generation_config = GenerationConfig(
    penalty_alpha=0.6, 
    do_sample = True, 
    top_k=5,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
outputs = persisted_model.generate(
    **inputs, max_new_tokens=20, generation_config=generation_config
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# COMMAND ----------

# If structured outputs are desired
from jsonformer import Jsonformer
merged_model = persisted_model.merge_and_unload()
system_message = "You are a helpful assistant who parses inputs to JSON."
prompt = """Extract in JSON the category of the user text (other, frankenstein, banter, political, violence, hate), 
the reason for the category and an action ("gently avoid question" if category is political, violence, hate, other. else engage) 
based on the user text from this message below.
Who is George W Bush?"""
prompt_template=f'''<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
'''
json_schema = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "reason": {"type": "string"},
        "action": {"type": "string"},
        }
    }
jsonformer = Jsonformer(merged_model, tokenizer, json_schema, prompt)
generated_data = jsonformer()
print(generated_data)
