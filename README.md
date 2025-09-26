# Romeo & Juliet RAG MVP (Mac-friendly)

This directory contains a minimal Retrieval-Augmented Generation (RAG) chatbot for Shakespeare's "Romeo and Juliet" that runs on a Mac (M2 ok). It uses SentenceTransformers + FAISS for retrieval and OpenAI for generation, with optional fallback to a local Ollama model if OpenAI quota errors occur.

## 1) Setup

```bash
# From workspace root
# cd to the workspace
cd <enter workspace path>

# (Recommended) Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt
```

optional: Set your OpenAI API key (optional, we mainly use ollama here):

```bash
export OPENAI_API_KEY=sk-...  # put your key here
```

Install Ollama for local fallback

<!-- - Install Ollama: see `https://ollama.com/download` -->

```bash
brew install ollama

```

Open another terminal and start the Ollama server

```bash
# Foreground (good for first run / debugging):
ollama serve
# Background via Homebrew Services (auto-start on login):
# brew services start ollama
# to stop: brew services stop ollama
```

- Come back and Pull a small instruct model (example):

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

## 2) Ingest the text

Ensure `Romeo_Juliet.txt` is present in this folder (it is already). Then build the FAISS index:

```bash
python ingest.py --input Romeo_Juliet.txt --outdir index --model sentence-transformers/all-MiniLM-L6-v2 --chunk_size 800 --chunk_overlap 150
```

This creates `index/faiss.index` and `index/metadata.json`.

## 3) Ask questions (chat loop)

Primary Use Ollama only (no OpenAI):

```bash
python ask.py --index_dir index --llm ollama --ollama_model llama3.1:8b-instruct-q4_K_M
```

Optional (OpenAI) with automatic fallback to Ollama on 429/quota error:

```bash
python ask.py --index_dir index --llm openai --openai_model gpt-4o-mini --ollama_model llama3.1:8b-instruct-q4_K_M
```

- Type your question at the `Q:` prompt.
- Type `quit` to exit.

## Notes

- If FAISS install fails on Mac, ensure you are using `faiss-cpu` (already in requirements) and Python 3.10/3.11.
- You can switch OpenAI models via `--openai_model`.
- To adjust retrieval granularity, tune `--chunk_size` and `--chunk_overlap` during ingest.
- To suppress the HuggingFace tokenizers parallelism warning, set:

```bash
export TOKENIZERS_PARALLELISM=false
```
