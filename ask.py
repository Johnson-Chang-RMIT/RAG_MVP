import os
import json
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# OpenAI SDK (>=1.0)
from openai import OpenAI
import requests

DEFAULT_MODEL_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "index"
DEFAULT_TOP_K = 4
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# Ollama defaults
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"  # can be pulled with `ollama pull llama3.1:8b-instruct-q4_K_M`


def load_index(index_dir: str):
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "metadata.json")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Index not found. Run ingest.py first. Expected files in {index_dir}")
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def embed_query(model_name: str, text: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype(np.float32)


def search(index, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores, idxs = index.search(query_vec, top_k)
    return scores[0], idxs[0]


def build_prompt(question: str, contexts: List[str]) -> str:
    header = (
        "You are a helpful assistant answering questions about Shakespeare's Romeo and Juliet.\n"
        "Use ONLY the provided context passages to answer. If you are unsure or the answer\n"
        "is not contained in the context, say you don't know. Be concise and accurate.\n\n"
    )
    joined_context = "\n\n---\n\n".join(contexts)
    prompt = (
        f"{header}"
        f"Question: {question}\n\n"
        f"Context Passages:\n{joined_context}\n\n"
        f"Answer:"
    )
    return prompt


def call_openai(prompt: str, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set. Please export it.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise, factual assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def call_ollama(prompt: str, base_url: str, model: str, timeout: int = 120) -> str:
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


def ensure_ollama_available(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def interactive_loop(index_dir: str, embed_model: str, openai_model: str, top_k: int, llm: str, ollama_url: str, ollama_model: str):
    print(f"Loading index from {index_dir} ...")
    index, metadata = load_index(index_dir)
    print("Loading embedding model ...")
    embedder = SentenceTransformer(embed_model)

    if llm == "ollama":
        if not ensure_ollama_available(ollama_url):
            print(
                "Ollama is not reachable at {}.\n"\
                "- Install/start Ollama app (https://ollama.com).\n"\
                "- Ensure it is running (default port 11434).\n"\
                "- Pull a model, e.g.: `ollama pull {}`\n"\
                "- Or set --ollama_url to your host.".format(ollama_url, ollama_model)
            )
            return

    print("Type your question about Romeo and Juliet. Type 'quit' to exit.")
    while True:
        try:
            query = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if query.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if not query:
            continue

        query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, row_idxs = index.search(query_vec, top_k)
        row_idxs = row_idxs[0]

        contexts: List[str] = []
        for idx in row_idxs:
            idx_int = int(idx)
            if idx_int < 0 or idx_int >= len(metadata):
                continue
            contexts.append(metadata[idx_int]["text"])

        prompt = build_prompt(query, contexts)

        answer: Optional[str] = None
        if llm == "openai":
            try:
                answer = call_openai(prompt, openai_model)
            except Exception as e:
                msg = str(e)
                if "429" in msg or "insufficient_quota" in msg or "quota" in msg.lower():
                    print("OpenAI quota error detected. Falling back to Ollama...")
                else:
                    print(f"Error calling OpenAI: {e}")
                if answer is None:
                    try:
                        if not ensure_ollama_available(ollama_url):
                            print("Ollama not reachable for fallback. Please start Ollama and try again.")
                            continue
                        answer = call_ollama(prompt, ollama_url, ollama_model)
                    except Exception as e2:
                        print(f"Error calling Ollama: {e2}")
                        continue
        else:
            try:
                answer = call_ollama(prompt, ollama_url, ollama_model)
            except Exception as e:
                print(f"Error calling Ollama: {e}")
                continue

        print("A:", answer)
        print()


def main():
    parser = argparse.ArgumentParser(description="Ask questions against the Romeo and Juliet RAG index.")
    parser.add_argument("--index_dir", type=str, default=DEFAULT_INDEX_DIR, help="Directory containing FAISS index and metadata.json")
    parser.add_argument("--embed_model", type=str, default=DEFAULT_MODEL_EMBED, help="SentenceTransformers model for queries")
    parser.add_argument("--openai_model", type=str, default=DEFAULT_OPENAI_MODEL, help="OpenAI chat model to use")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of passages to retrieve")
    parser.add_argument("--llm", type=str, choices=["openai", "ollama"], default="openai", help="Primary LLM provider")
    parser.add_argument("--ollama_url", type=str, default=DEFAULT_OLLAMA_BASE_URL, help="Ollama base URL")
    parser.add_argument("--ollama_model", type=str, default=DEFAULT_OLLAMA_MODEL, help="Ollama model name/tag")
    args = parser.parse_args()

    interactive_loop(
        args.index_dir,
        args.embed_model,
        args.openai_model,
        args.top_k,
        args.llm,
        args.ollama_url,
        args.ollama_model,
    )


if __name__ == "__main__":
    main()
