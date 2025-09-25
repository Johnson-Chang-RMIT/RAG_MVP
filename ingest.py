import os
import json
import math
import argparse
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INPUT = "Romeo_Juliet.txt"
DEFAULT_OUTDIR = "index"
DEFAULT_CHUNK_SIZE = 800  # characters
DEFAULT_CHUNK_OVERLAP = 150  # characters


def read_text_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input text file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append({
            "id": f"chunk-{len(chunks)}",
            "text": chunk,
            "start": start,
            "end": end
        })
        if end == text_len:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def build_embeddings(model_name: str, texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # using cosine similarity with normalized vectors
    index.add(embeddings)
    return index


def save_index(outdir: str, index: faiss.Index, metadata: List[Dict[str, str]]):
    os.makedirs(outdir, exist_ok=True)
    faiss.write_index(index, os.path.join(outdir, "faiss.index"))
    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Ingest a text file into a FAISS vector index.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to input text file")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory for index and metadata")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="SentenceTransformers model name")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in characters")
    args = parser.parse_args()

    print(f"Reading text from {args.input} ...")
    text = read_text_file(args.input)

    print(f"Chunking text (size={args.chunk_size}, overlap={args.chunk_overlap}) ...")
    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
    texts = [c["text"] for c in chunks]

    print(f"Building embeddings with {args.model} ...")
    embeddings = build_embeddings(args.model, texts)

    print("Building FAISS index ...")
    index = build_faiss_index(embeddings)

    print(f"Saving index and metadata to {args.outdir} ...")
    save_index(args.outdir, index, chunks)

    print(f"Done. Indexed {len(chunks)} chunks. Output dir: {args.outdir}")


if __name__ == "__main__":
    main()
