from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from extractor import load_documents


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class SearchResult:
    filename: str
    chunk_id: int
    score: float
    text: str


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _chunk_text(text: str, chunk_size_words: int = 120, overlap_words: int = 30) -> List[str]:
    words = _normalize_text(text).split()
    if not words:
        return []

    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)

    return chunks


def build_search_system(
    data_folder: str = "../data",
    model_name: str = DEFAULT_MODEL_NAME,
) -> Dict[str, Any]:
    documents = load_documents(data_folder)
    model = SentenceTransformer(model_name)

    records: List[Dict[str, Any]] = []
    chunk_texts: List[str] = []

    for filename, text in documents.items():
        chunks = _chunk_text(text)
        if not chunks:
            continue

        for chunk_id, chunk_text in enumerate(chunks):
            records.append(
                {
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )
            chunk_texts.append(chunk_text)

    if not chunk_texts:
        embeddings = np.empty((0, 0), dtype=np.float32)
    else:
        embeddings = model.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    return {
        "model": model,
        "records": records,
        "embeddings": embeddings,
        "data_folder": str(Path(data_folder)),
    }


def search_documents(
    query: str,
    search_system: Dict[str, Any],
    top_k: int = 3,
) -> List[SearchResult]:
    query = _normalize_text(query)
    if not query:
        return []

    records = search_system["records"]
    embeddings = search_system["embeddings"]
    model: SentenceTransformer = search_system["model"]

    if len(records) == 0 or embeddings.size == 0:
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)[0]

    scores = embeddings @ query_embedding
    top_indices = np.argsort(-scores)[: max(1, top_k)]

    results: List[SearchResult] = []
    for index in top_indices:
        record = records[int(index)]
        results.append(
            SearchResult(
                filename=record["filename"],
                chunk_id=record["chunk_id"],
                score=float(scores[int(index)]),
                text=record["text"],
            )
        )

    return results


def format_search_results(results: List[SearchResult]) -> str:
    if not results:
        return "No matching documents found."

    lines = []
    for result in results:
        lines.append(
            f"- {result.filename} [chunk {result.chunk_id}] score={result.score:.3f}\n  {result.text[:250]}"
        )
    return "\n".join(lines)


def answer_question(
    question: str,
    search_system: Dict[str, Any],
    top_k: int = 3,
    llm_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    results = search_documents(question, search_system, top_k=top_k)
    context = "\n\n".join(
        f"Source: {result.filename}\n{result.text}" for result in results
    )

    if not llm_model_name:
        return {
            "question": question,
            "answer": None,
            "context": context,
            "retrieved_results": [result.__dict__ for result in results],
        }

    try:
        from transformers import pipeline

        generator = pipeline(
            "text-generation",
            model=llm_model_name,
            tokenizer=llm_model_name,
        )

        prompt = (
            "Answer the question using only the provided context. "
            "If the context does not contain the answer, say you cannot determine it.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        output = generator(
            prompt,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
        )

        generated_text = output[0]["generated_text"]
        answer = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()

    except Exception as exc:
        answer = f"Local QA model could not be loaded: {exc}"

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "retrieved_results": [result.__dict__ for result in results],
    }


if __name__ == "__main__":
    system = build_search_system()
    query = input("Enter search query: ")
    print(format_search_results(search_documents(query, system, top_k=3)))