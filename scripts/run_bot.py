import os
import argparse
from src.rag_engine import AcademicRAGIndex
from src.academic_assistant import AcademicAssistant
from src.telegram_bot import launch_telegram_bot
from src.config import DEFAULT_KNOWLEDGE_DIR

def main():
    parser = argparse.ArgumentParser(description="Run ExamNoteLLM Telegram Bot")
    parser.add_argument("--knowledge_dir", type=str, default=DEFAULT_KNOWLEDGE_DIR, help="Curriculum directory")
    args = parser.parse_args()

    rag = AcademicRAGIndex()
    rag.ingest_directory(args.knowledge_dir)

    assistant = AcademicAssistant(rag_index=rag)
    launch_telegram_bot(assistant)

if __name__ == "__main__":
    main()
