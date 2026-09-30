import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
from .config import (
    BASE_MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LORA_TARGET_MODULES, MAX_SEQ_LENGTH, TRAIN_BATCH_SIZE,
    GRAD_ACCUMULATION_STEPS, NUM_TRAIN_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, WARMUP_RATIO, OUTPUT_DIR, get_compute_dtype
)

def build_qlora_model(model_name: str = BASE_MODEL_NAME):
    """Loads base model in 4-bit NF4 and attaches LoRA adapters."""
    compute_dtype = get_compute_dtype()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none"
    )

    model = get_peft_model(model, peft_config)
    return model, tokenizer

def train_qlora(model, tokenizer, train_dataset, val_dataset, formatting_func, output_dir: str = OUTPUT_DIR):
    """Fine-tunes the model with SFTTrainer using paged_adamw_8bit."""
    compute_dtype = get_compute_dtype()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        learning_rate=LEARNING_RATE,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        seed=3407,
        gradient_checkpointing=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        ),
        args=training_args
    )

    trainer.train()
    return trainer
