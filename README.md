# ExamNoteLLM: AI Universal Academic Assistant with QLoRA & RAG

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/Model-Llama--3--8B--Instruct-yellow.svg)](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
[![PEFT](https://img.shields.io/badge/PEFT-QLoRA%204--bit-orange.svg)](https://github.com/huggingface/peft)
[![RAG](https://img.shields.io/badge/RAG-SentenceTransformers-green.svg)](https://www.sbert.net/)

An open-source academic intelligence platform combining **QLoRA parameter-efficient fine-tuning (4-bit NF4)** with a **multi-tenant Retrieval-Augmented Generation (RAG)** engine. Built to provide syllabus-grounded explanations, mark-specific exam notes, automated question generation, rubric-based student answer evaluation, and interactive revision across diverse subjects, institutions, and education levels.

---

## Architecture Overview

```
                      +-----------------------------------+
                      |   Curriculum Knowledge Base       |
                      |   (PDFs, TXT, MD Syllabi/Notes)   |
                      +-----------------+-----------------+
                                        |
                                        v
+-----------------------+     +-------------------+     +-----------------------+
|  Student Query        | --> | RAG Vector Store  | --> | Prompt Construction   |
|  (Question, Marks,    |     | (MiniLM + Multi-  |     | (Syllabus Context +   |
|  Subject, University) |     | Tenant Metadata)  |     | Mark Rubric Guidance) |
+-----------------------+     +-------------------+     +-----------+-----------+
                                                                    |
                                                                    v
+-----------------------+     +-------------------+     +-----------+-----------+
| Multi-Turn Telegram   | <-- | Response Engine   | <-- | Llama-3-8B-Instruct   |
| Bot / Python CLI API  |     | (Streaming Token  |     | (4-bit QLoRA Adapter) |
|                       |     | Generation)       |     |                       |
+-----------------------+     +-------------------+     +-----------------------+
```

---

## Core Capabilities

1. **Syllabus-Grounded Explanations**: Retrieves university syllabus modules and textbook references to keep model responses strictly within the course scope.
2. **Mark-Constrained Answer Synthesis**:
   - **2 Marks**: Core definition and 2 bullet points (~180 tokens).
   - **5 Marks**: Definition, 4-5 structured points, and a short example (~380 tokens).
   - **10 Marks**: Full university structure: Definition, Mechanism, Key Properties, Real-World Application (~750 tokens).
   - **15 Marks**: Comprehensive essay: Theory, Architecture, Comparative Table, Case Study, and Summary (~1100 tokens).
3. **Automated Exam Question Generation**: Generates balanced question papers (Part A definitions, Part B problem solving, Part C system design) complete with model answer keys.
4. **Rubric-Based Student Answer Evaluation**: Compares student submissions against authoritative textbook content to provide objective scorecards, key concepts covered, and missing points.
5. **Study Materials & Adaptive Learning**: Generates 5-minute pre-exam cheat-sheets, active-recall flashcards, and diagnostic 3-day recovery study plans.
6. **Telegram Bot Interface**: Multi-turn streaming bot with commands (`/ask`, `/eval`, `/quiz`, `/notes`, `/profile`).

---

## Repository Structure

```
ExamNoteLLM/
├── notebooks/
│   └── ExamNoteLLM_Universal_Assistant.ipynb   # Complete end-to-end Colab notebook
├── src/
│   ├── config.py                               # Central hyperparameters and paths
│   ├── dataset.py                              # Dataset loaders and prompt formatting
│   ├── qlora_trainer.py                        # 4-bit quantization & SFTTrainer
│   ├── model_exporter.py                       # LoRA adapter merging utility
│   ├── rag_engine.py                           # SentenceTransformers vector index
│   ├── academic_assistant.py                   # Academic workflows (notes, eval, quiz)
│   └── telegram_bot.py                         # Multi-turn streaming Telegram assistant
├── data/
│   ├── sample_exam_questions.jsonl             # Benchmark evaluation dataset
│   └── academic_curricula/                     # Sample curriculum notes (DBMS, OS, ML)
├── scripts/
│   ├── train.py                                # CLI fine-tuning script
│   ├── evaluate.py                             # CLI evaluation script
│   └── run_bot.py                              # CLI bot launcher
├── requirements.txt                            # Dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/Adithyan435/ExamNoteLLM.git
cd ExamNoteLLM
pip install -r requirements.txt
```

### 2. Fine-Tuning with QLoRA
```bash
python scripts/train.py --dataset data/sample_exam_questions.jsonl --output_dir output/checkpoints
```

### 3. Running RAG Evaluation
```bash
python scripts/evaluate.py --question "Explain ACID properties in DBMS" --marks 10 --subject DBMS
```

### 4. Deploying Telegram Assistant
```bash
export TELEGRAM_API_KEY="your_bot_token_here"
python scripts/run_bot.py
```

### 5. Google Colab Training
Open `notebooks/ExamNoteLLM_Universal_Assistant.ipynb` directly in Google Colab with a free T4 GPU.

---

## Training Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| Base Model | Meta-Llama-3-8B-Instruct | 8-billion parameter instruction-tuned model |
| Quantization | 4-bit NormalFloat (NF4) | Memory reduction via BitsAndBytes |
| LoRA Rank (r) | 16 | Rank dimension for low-rank updates |
| LoRA Alpha | 32 | Scaling factor (2x rank) |
| Target Modules | q, k, v, o, gate, up, down | All attention and MLP projection layers |
| Optimizer | paged_adamw_8bit | Page memory optimizer for 16GB VRAM |
| Batch Size | 2 per device (grad accum 8) | Effective batch size of 16 |
| Learning Rate | 2e-4 (Cosine decay) | Optimized for stable QLoRA convergence |
| Context Window | 1024 tokens | Academic multi-mark answer capacity |

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
