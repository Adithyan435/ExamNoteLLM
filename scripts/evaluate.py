import argparse
from src.rag_engine import AcademicRAGIndex
from src.academic_assistant import AcademicAssistant
from src.config import DEFAULT_KNOWLEDGE_DIR

def main():
    parser = argparse.ArgumentParser(description="Evaluate Academic Assistant on Exam Questions")
    parser.add_argument("--knowledge_dir", type=str, default=DEFAULT_KNOWLEDGE_DIR, help="Path to syllabus files")
    parser.add_argument("--question", type=str, default="Explain ACID properties in 10 marks", help="Question to test")
    parser.add_argument("--marks", type=int, default=10, help="Marks weightage")
    parser.add_argument("--subject", type=str, default="DBMS", help="Subject")
    args = parser.parse_args()

    print(f"Indexing knowledge base from {args.knowledge_dir}...")
    rag = AcademicRAGIndex()
    rag.ingest_directory(args.knowledge_dir)

    print("Initializing Academic Assistant...")
    assistant = AcademicAssistant(rag_index=rag)

    print(f"\n--- Generating {args.marks}-mark answer for {args.subject} ---")
    answer = assistant.generate_syllabus_notes(args.question, marks=args.marks, subject=args.subject)
    print(answer)

if __name__ == "__main__":
    main()
