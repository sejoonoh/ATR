#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2024-07-31
#Adversarial Text Rewriting for Text-aware Recommender Systems
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer,LineByLineTextDataset
from datasets import load_dataset
from itertools import chain
import math
# Load the pre-trained model and tokenizer
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
# Load your training datia and format it as a TextDataset
all_dataset = load_dataset("text",data_files="src/dataset/downstream/amazon_book/amazon_description_new.txt")
column_names = all_dataset["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])
   
tokenized_dataset = all_dataset.map(tokenize_function, batched=True,remove_columns=column_names)
# Create a DataCollator for Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

def group_texts(examples):
    block_size=512
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
)
# Define the training arguments
training_args = TrainingArguments(
    output_dir="test_clm_amazon",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_strategy = "no",
    learning_rate=1e-5,
)
trainer = Trainer(model=model,args=training_args,train_dataset=lm_dataset["train"],data_collator=data_collator,)
trainer.train()
# Save the fine-tuned model
model.save_pretrained("opt-350m")
tokenizer.save_pretrained("opt-350m")
