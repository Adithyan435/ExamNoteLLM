import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .config import get_compute_dtype

def merge_and_export_lora(base_model_name: str, adapter_path: str, export_path: str):
    """Merges LoRA adapter into 16-bit base model weights for standalone export."""
    compute_dtype = get_compute_dtype()
    
    print(f"Loading base model ({base_model_name}) in 16-bit precision...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    print(f"Applying LoRA adapter from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged weights to {export_path}...")
    merged_model.save_pretrained(export_path, safe_serialization=True, max_shard_size="4GB")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(export_path)

    del base_model
    del peft_model
    del merged_model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model export complete.")
