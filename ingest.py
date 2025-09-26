import os
import json
import math
import argparse
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
# Preprocesses and indexes the text


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INPUT = "Romeo_Juliet.txt"
DEFAULT_OUTDIR = "index"
DEFAULT_CHUNK_SIZE = 800  # characters
DEFAULT_CHUNK_OVERLAP = 150  # characters

# specify input type as str(path: str) 
# specify return type as str(-> str)
def read_text_file(path: str) -> str:
    if not os.path.exists(path):
        # check if file exists with os.path.exists() return boolean
        raise FileNotFoundError(f"Input text file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        # encoding="utf-8" to handle special characters
        return f.read()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    # chunks = []  With type hint
    # Python knows this will be a list of dictionaries
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append({
            "id": f"chunk-{len(chunks)}",
            # chunk-0, chunk-1, chunk-2...
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
    # -> np.ndarray means "this function returns a NumPy array"( N-dimensional array)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    # embeddings: shape = (num_chunks, embedding_dimension)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Creates structure but it's empty, using cosine similarity with normalized vectors
    index.add(embeddings)  # Fill the container with data
    return index
# # Input embeddings (3 chunks, 3 dimensions each):
# embeddings = [
#     [0.1, 0.2, 0.3],  # chunk-0 embedding
#     [0.5, 0.6, 0.7],  # chunk-1 embedding
#     [0.9, 0.1, 0.2]   # chunk-2 embedding
# ]
# dim = embeddings.shape[1]              # dim = 3
# index = faiss.IndexFlatIP(dim)         # Create index for 3D vectors
# index.add(embeddings)                  # Store all vectors
# # Now you can search: index.search(query_vector, top_k=2)
# # Later, in ask.py:
# query_embedding = [0.2, 0.3, 0.4]  # User's question as vector
# scores, ids = index.search(query_embedding, k=3)  # Find 3 most similar
# # Returns: scores=[0.95, 0.87, 0.72], ids=[1, 0, 2]  
# # Meaning: chunk-1 is most similar, then chunk-0, then chunk-2


def save_index(outdir: str, index: faiss.Index, metadata: List[Dict[str, str]]):
    os.makedirs(outdir, exist_ok=True)
    # exist_ok : "don't error if directory already exists"
    faiss.write_index(index, os.path.join(outdir, "faiss.index"))
    # faiss.write_index(index, path)
    # Write the searchable index to: index/faiss.index
    # os.path.join(...) is better than [outdir + "/" + "faiss.index"] because it can join multiple parts
    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        # Converts the metadata(chunks list of dicts) to JSON format and writes it into metadata.json
        # ensure_ascii : False to preserve special characters like curly quotes(so that when it's print out it shows actual curly quotes instead of \u201c and \u201d)
        # indent : makes it human-readable with proper indentation.
# Why save both files?
# FAISS index For fast similarity search
# metadata.json To get the actual text content of matching chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest a text file into a FAISS vector index.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to input text file")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory for index and metadata")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="SentenceTransformers model name")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in characters")
    # We add Command-Line Arguments:
    # Basic run:
    # python ingest.py
    # # With arguments:
    # python ingest.py --input mybook.txt --chunk_size 999 --outdir my_index
    #HELP let you see what arguments are available
    # now try run 
    # source .venv/bin/activate 
    # python ingest.py --help
    
    args = parser.parse_args()
    # Converts arguments into an object you can use

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
