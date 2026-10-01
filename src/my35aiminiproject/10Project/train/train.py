"""
  THE 10TH PROJECT : Fine-Tune Llama 3.2 With LoRA
Learn:
 1.fine tuning is the proccess of traing a small part of the model and freeze the rest (adjecting the paramaters /wights of the model)
 2.the flow : dataset (train.jsonl+test.jsonl) -> tain.py (with LORA) -> LoRA adjecements -> mrege.py (LoRA adjecements model + the base model) -> llama.cpp -> gguf version -> ollama create my_tuned_model -f Modelfile (it configure the created model).
 3.Note:
 The LoRA fine-tuned model successfully learned the target response style from the small training dataset. Compared with the base model, it more consistently followed the structured format using sections such as Goal, Plan, Milestone, and Contingency. However, the fine-tuned model still showed some limitations in following numerical and time constraints exactly, and some responses were either too short or repetitive.

 4.commmands:
  1.c:/Users/surface/Desktop/my35AIminiProject/src/my35aiminiproject/10Project/train/train.py   
  2.c:/Users/surface/Desktop/my35AIminiProject/src/my35aiminiproject/10Project/train/merge.py
  3.git clone https://github.com/ggml-org/llama.cpp.git  (and then checked if there a dir convert_hf_to_gguf.py )
  4.python convert_hf_to_gguf.py ..src my35aiminiproject 10Project merged --outfile .. src my35aiminiproject 10Project gguf llama-3.2-1b-ft-f16.gguf --outtype f16   
  5. @"                                                     
   >> FROM ./gguf/llama-3.2-1b-ft-f16.gguf                                                                                   
   >>                                
   >> PARAMETER temperature 0.7
   >> PARAMETER top_p 0.9
   >> "@ | Set-Content Modelfile

  6.ollama create my_model -f Modelfile 
  7.ollama run my_model  

"""

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "train.jsonl"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# ============================================================
# Basic checks
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Training dataset not found:\n{DATASET_PATH}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Llama 3.2 1B + LoRA Training")
print("=" * 60)
print(f"Model:   {MODEL_NAME}")
print(f"Dataset: {DATASET_PATH}")
print(f"Output:  {OUTPUT_DIR}")
print(f"Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("=" * 60)

# ============================================================
# Load dataset
# ============================================================

dataset = load_dataset(
    "json",
    data_files=str(DATASET_PATH),
    split="train",
)

print(f"Number of training examples: {len(dataset)}")

# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# ============================================================
# Prepare examples
# ============================================================

# 512 gives the response enough room to keep the full structure.
MAX_LENGTH = 512


def prepare_example(example):
    instruction = str(example["instruction"]).strip()
    response = str(example["response"]).strip()

    # --------------------------------------------------------
    # Full conversation
    # --------------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": instruction,
        },
        {
            "role": "assistant",
            "content": response,
        },
    ]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # --------------------------------------------------------
    # User prompt only
    # --------------------------------------------------------

    prompt_messages = [
        {
            "role": "user",
            "content": instruction,
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # --------------------------------------------------------
    # Tokenize full conversation
    # --------------------------------------------------------

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    # Tokenize only the prompt so we know where
    # the assistant response starts.
    prompt_tokens = tokenizer(
        prompt_text,
        truncation=False,
        padding=False,
    )

    input_ids = list(full_tokens["input_ids"])
    attention_mask = list(full_tokens["attention_mask"])

    prompt_length = len(prompt_tokens["input_ids"])

    # --------------------------------------------------------
    # Assistant-only loss
    # --------------------------------------------------------

    labels = [-100] * len(input_ids)

    # Only assistant tokens contribute to the loss.
    for i in range(
        min(prompt_length, len(input_ids)),
        len(input_ids),
    ):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


tokenized_dataset = dataset.map(
    prepare_example,
    remove_columns=dataset.column_names,
    writer_batch_size=10,
)

print("\nExample tokenized successfully.")
print(
    "First example length:",
    len(tokenized_dataset[0]["input_ids"])
)

# Check that the first example actually contains
# assistant tokens to train on.
first_labels = tokenized_dataset[0]["labels"]
trainable_tokens = sum(
    1 for x in first_labels if x != -100
)

print(
    "First example trainable assistant tokens:",
    trainable_tokens
)

if trainable_tokens == 0:
    raise RuntimeError(
        "No assistant tokens found for training. "
        "The prompt/response boundary is incorrect."
    )

# ============================================================
# Data collator
# ============================================================

def data_collator(features):
    max_len = max(
        len(feature["input_ids"])
        for feature in features
    )

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for feature in features:
        padding_length = (
            max_len - len(feature["input_ids"])
        )

        batch_input_ids.append(
            feature["input_ids"]
            + [tokenizer.pad_token_id] * padding_length
        )

        batch_attention_mask.append(
            feature["attention_mask"]
            + [0] * padding_length
        )

        batch_labels.append(
            feature["labels"]
            + [-100] * padding_length
        )

    return {
        "input_ids": torch.tensor(
            batch_input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            batch_attention_mask,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            batch_labels,
            dtype=torch.long,
        ),
    }


# ============================================================
# Base model
# ============================================================

print("\nLoading base model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# ============================================================
# LoRA configuration
# ============================================================

# Keep this CPU-friendly.
# We train the main attention projections that strongly
# affect how the model responds to prompts.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

model = get_peft_model(
    model,
    lora_config,
)

print("\nLoRA model:")
model.print_trainable_parameters()

# ============================================================
# Training arguments
# ============================================================

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    # Slightly more training to strengthen the style/format.
    num_train_epochs=5,

    # CPU-friendly.
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=2e-4,
    weight_decay=0.01,

    logging_steps=1,
    save_strategy="epoch",

    report_to="none",

    fp16=False,
    bf16=False,

    dataloader_num_workers=0,

    optim="adamw_torch",

    gradient_checkpointing=True,

)

# ============================================================
# Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ============================================================
# Train
# ============================================================

print("\nStarting LoRA training...\n")

trainer.train()

# ============================================================
# Save adapter
# ============================================================

print("\nSaving LoRA adapter...")

model.save_pretrained(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)
print("LoRA adapter saved to:")
print(OUTPUT_DIR)
print("=" * 60)