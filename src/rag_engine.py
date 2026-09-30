import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pypdf
from .config import EMBEDDING_MODEL_NAME

class AcademicRAGIndex:
    """Multi-tenant semantic index for syllabi, textbooks, and notes."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.metadata = []
        self.embeddings = None

    def add_document(self, text: str, meta: dict, chunk_size: int = 400, overlap: int = 50):
        words = text.split()
        if not words:
            return

        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += (chunk_size - overlap)

        vectors = self.encoder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

        for c, m in zip(chunks, [meta] * len(chunks)):
            self.documents.append(c)
            self.metadata.append(m)

        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])

    def ingest_directory(self, directory_path: str) -> int:
        if not os.path.exists(directory_path):
            return 0

        indexed_count = 0
        supported = ('.txt', '.md', '.pdf', '.json')
        files = [f for f in os.listdir(directory_path) if f.lower().endswith(supported)]

        for filename in files:
            path = os.path.join(directory_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            base = os.path.splitext(filename)[0]

            meta = {
                "title": base,
                "filename": filename,
                "subject": "General",
                "institution": "Universal",
                "education_level": "Undergraduate",
                "unit": "General",
                "doc_type": ext.replace('.', '').upper()
            }
            if "_" in base:
                parts = base.split("_")
                meta["subject"] = parts[0]
                if len(parts) > 1:
                    meta["unit"] = parts[1]

            try:
                if ext in ('.txt', '.md'):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        self.add_document(f.read(), meta)
                    indexed_count += 1
                elif ext == '.pdf':
                    reader = pypdf.PdfReader(path)
                    text = ""
                    for p in reader.pages:
                        extracted = p.extract_text()
                        if extracted:
                            text += "\n" + extracted
                    if text.strip():
                        self.add_document(text, meta)
                        indexed_count += 1
                elif ext == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            t = item.get("content") or item.get("text") or str(item)
                            self.add_document(t, {**meta, **{k: v for k, v in item.items() if k not in ("content", "text")}})
                    elif isinstance(data, dict):
                        t = data.get("content") or json.dumps(data, indent=2)
                        self.add_document(t, {**meta, **data.get("metadata", {})})
                    indexed_count += 1
            except Exception as e:
                print(f"Error indexing {filename}: {e}")

        return indexed_count

    def search(self, query: str, top_k: int = 3, subject: str = None, institution: str = None, education_level: str = None, unit: str = None):
        if self.embeddings is None or len(self.documents) == 0:
            return []

        candidates = []
        for idx, meta in enumerate(self.metadata):
            match = True
            if subject and meta.get("subject", "").lower() != subject.lower():
                match = False
            if institution and institution.lower() != "universal" and meta.get("institution", "").lower() not in [institution.lower(), "universal"]:
                match = False
            if education_level and meta.get("education_level", "").lower() != education_level.lower():
                match = False
            if unit and meta.get("unit", "").lower() != unit.lower():
                match = False
            if match:
                candidates.append(idx)

        if not candidates:
            candidates = list(range(len(self.documents)))

        query_vec = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        candidate_vecs = self.embeddings[candidates]
        scores = np.dot(candidate_vecs, query_vec)

        ranked = np.argsort(scores)[::-1][:top_k]
        return [{
            "content": self.documents[candidates[r]],
            "metadata": self.metadata[candidates[r]],
            "score": float(scores[r])
        } for r in ranked]
