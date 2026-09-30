import os
import torch

# Base models
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Paths
DEFAULT_DATASET_PATH = "data/sample_exam_questions.jsonl"
DEFAULT_KNOWLEDGE_DIR = "data/academic_curricula"
OUTPUT_DIR = "output/checkpoints"
FINAL_ADAPTER_PATH = "output/Llama-8b-lora-ultra"
MERGED_MODEL_PATH = "output/Llama-8b-merged-full"

# LoRA Hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training Hyperparameters
MAX_SEQ_LENGTH = 1024
TRAIN_BATCH_SIZE = 2
GRAD_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

# Compute precision
def get_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    elif torch.cuda.is_available():
        return torch.float16
    return torch.float32
