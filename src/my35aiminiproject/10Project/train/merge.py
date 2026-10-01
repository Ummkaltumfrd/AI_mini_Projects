from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_DIR = BASE_DIR / "output"
MERGED_DIR = BASE_DIR / "merged"

MERGED_DIR.mkdir(parents=True, exist_ok=True)


print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_DIR),
)

print("Merging LoRA adapter into base model...")

model = model.merge_and_unload()

print("Saving merged model...")

model.save_pretrained(
    str(MERGED_DIR),
    safe_serialization=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(str(MERGED_DIR))

print("\n" + "=" * 60)
print("MERGE FINISHED")
print("=" * 60)
print(f"Merged model saved to: {MERGED_DIR}")
print("=" * 60)