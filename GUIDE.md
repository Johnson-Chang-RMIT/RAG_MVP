# Romeo & Juliet RAG MVP — Teammate Guide (macOS)

This guide shows how to set up and run the RAG chatbot locally on macOS. It supports:

- Local, free LLM via Ollama (no API key)
- OpenAI API as primary with automatic fallback to Ollama on quota errors

If your local path is different, replace `/Users/johnsonchang/Desktop/WIL_reborn/RAG_fully_courser` with your path.

## 0) Install prerequisites

- Install Homebrew (if missing):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

- Install Python 3.11 (recommended):

```bash
brew install python@3.11
```

## 1) Enter the project

```bash
cd /Users/johnsonchang/Desktop/WIL_reborn/RAG_fully_courser
```

## 2) Create and activate a Python 3.11 virtual environment

```bash
/opt/homebrew/bin/python3.11 -m venv .venv   # On Intel Macs, path may be /usr/local/bin/python3.11
source .venv/bin/activate
python --version  # should show 3.11.x
```

## 3) Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Build the retrieval index (one-time per machine)

Ensure `Romeo_Juliet.txt` is present in this folder (provided).

```bash
python ingest.py --input Romeo_Juliet.txt --outdir index --model sentence-transformers/all-MiniLM-L6-v2 --chunk_size 800 --chunk_overlap 150
```

This creates `index/faiss.index` and `index/metadata.json`.

---

## Option A: Run with local free LLM (Ollama)

### A1) Install and start Ollama

```bash
brew install ollama
ollama serve
```

- Keep this terminal open. You should see: `Listening on 127.0.0.1:11434`.

### A2) In a new terminal, pull a small instruct model

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

- First download may take 30–90 minutes.

### A3) Activate venv again and run the chat

```bash
cd /Users/johnsonchang/Desktop/WIL_reborn/RAG_fully_courser
source .venv/bin/activate
python ask.py --index_dir index --llm ollama --ollama_model llama3.1:8b-instruct-q4_K_M
```

- Type your question at the `Q:` prompt. Type `quit` to exit.

---

## Option B: OpenAI primary (with automatic fallback to Ollama on quota 429)

### B1) Set your OpenAI API key

```bash
cd /Users/johnsonchang/Desktop/WIL_reborn/RAG_fully_courser
source .venv/bin/activate
export OPENAI_API_KEY=sk-...your_key...
```

### B2) (Optional but recommended) Start Ollama for fallback

- Follow Option A1 and A2 to start Ollama and pull a model.

### B3) Run the chat

```bash
python ask.py --index_dir index --llm openai --openai_model gpt-4o-mini --ollama_model llama3.1:8b-instruct-q4_K_M
```

- If OpenAI returns a quota error (429), the script falls back to Ollama automatically.

---

## Troubleshooting

- Python/NumPy/SciPy errors:

  - Ensure you are in the venv and using Python 3.11, not 3.13 or base Conda.
  - If you accidentally used another Python, remove `.venv` and recreate with Python 3.11.

- Ollama connection refused:

  - Make sure `ollama serve` is running (leave that terminal open).
  - Verify with: `curl http://127.0.0.1:11434/api/tags`
  - If not installed, run `brew install ollama`.

- Tokenizers warning:

```bash
export TOKENIZERS_PARALLELISM=false
```

- Slow first response on Ollama:

  - The first request loads the model; subsequent calls are faster.

- Rebuilding index:

  - If you change the text or chunking settings, rerun the ingest command.

- Exit:
  - Type `quit` at the `Q:` prompt to stop the loop.
