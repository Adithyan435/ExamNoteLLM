import os
import argparse
from src.dataset import load_and_prepare_dataset, format_exam_prompt
from src.qlora_trainer import build_qlora_model, train_qlora
from src.config import DEFAULT_DATASET_PATH, OUTPUT_DIR

def main():
    parser = argparse.ArgumentParser(description="Train ExamNoteLLM with QLoRA")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="Path to JSONL dataset")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Checkpoint output directory")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset}")
    train_ds, val_ds = load_and_prepare_dataset(args.dataset)

    print("Building QLoRA 4-bit model...")
    model, tokenizer = build_qlora_model()

    print("Starting training...")
    trainer = train_qlora(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        formatting_func=lambda ex: format_exam_prompt(ex, eos_token=tokenizer.eos_token),
        output_dir=args.output_dir
    )
    print("Training finished successfully.")

if __name__ == "__main__":
    main()
