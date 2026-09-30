import os
from datasets import load_dataset

def load_and_prepare_dataset(file_path: str, test_size: float = 0.15, seed: int = 42):
    """Loads a JSONL dataset from disk and splits into train and validation sets."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    raw_dataset = load_dataset("json", data_files=file_path, split="train")
    split_dataset = raw_dataset.train_test_split(test_size=test_size, seed=seed)
    return split_dataset["train"], split_dataset["test"]

def format_exam_prompt(example: dict, eos_token: str = "") -> str:
    """Formats an instruction-input-output example for fine-tuning."""
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()

    if input_text:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Output:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Output:\n{output}"

    if eos_token:
        text += eos_token
    return text
