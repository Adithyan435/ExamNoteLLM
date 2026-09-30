import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from .config import BASE_MODEL_NAME, FINAL_ADAPTER_PATH, get_compute_dtype

class AcademicAssistant:
    """Executes syllabus notes, question papers, answer grading, and study materials."""
    def __init__(self, model_path: str = BASE_MODEL_NAME, adapter_path: str = FINAL_ADAPTER_PATH, rag_index=None):
        self.rag_index = rag_index
        compute_dtype = get_compute_dtype()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype
        )

        if adapter_path:
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            except Exception:
                pass

    def generate(self, instruction: str, context: str = "", max_tokens: int = 700, temperature: float = 0.4) -> str:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Output:\n" if context else f"### Instruction:\n{instruction}\n\n### Output:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=(temperature > 0.1),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full.split("### Output:\n")[-1].strip() if "### Output:\n" in full else full.strip()

    def generate_syllabus_notes(self, question: str, marks: int = 5, subject: str = "General", institution: str = "Universal", level: str = "Undergraduate") -> str:
        context_str = ""
        if self.rag_index:
            chunks = self.rag_index.search(query=question, top_k=2, subject=subject if subject != 'General' else None, institution=institution, education_level=level)
            context_str = "\n\n".join([f"[{c['metadata']['title']}]\n{c['content']}" for c in chunks])

        budgets = {2: 180, 5: 380, 10: 750, 15: 1100}
        instruction = (
            f"You are a university instructor in {subject} at {institution} ({level} level).\n"
            f"Answer the exam question: '{question}'.\n"
            f"Mark Allocated: {marks} Marks.\n"
            f"Format response with appropriate academic structure for {marks} marks."
        )
        return self.generate(instruction=instruction, context=context_str, max_tokens=budgets.get(marks, 600), temperature=0.4)

    def evaluate_student_answer(self, question: str, student_answer: str, max_marks: int = 10, subject: str = "General") -> str:
        context_str = ""
        if self.rag_index:
            chunks = self.rag_index.search(query=question, top_k=2, subject=subject if subject != 'General' else None)
            context_str = "\n\n".join([c["content"] for c in chunks])

        instruction = (
            f"Evaluate this student answer for {max_marks} marks in {subject}.\n"
            f"Question: '{question}'\n"
            f"Student Answer: \"{student_answer}\"\n\n"
            f"Provide: Score Awarded [X/{max_marks}], Concepts Correctly Covered, Missing Points, and Improvement Feedback."
        )
        return self.generate(instruction=instruction, context=context_str, max_tokens=650, temperature=0.3)
