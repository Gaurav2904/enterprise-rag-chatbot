# Enterprise RAG Chatbot

> Production-grade Retrieval-Augmented Generation chatbot powered by **LangChain**, **Azure OpenAI (GPT-4)**, **FAISS**, and **HuggingFace sentence-transformers** — served via FastAPI and containerised with Docker.

---

## Features

| Feature | Detail |
|---|---|
| **LLM** | Azure OpenAI GPT-4 (configurable deployment) |
| **Embeddings** | `sentence-transformers/all-mpnet-base-v2` (768-dim) |
| **Vector store** | FAISS (CPU; persistent to disk) |
| **Retrieval** | Maximal Marginal Relevance (MMR) for diversity |
| **Prompting** | Chain-of-Thought question condensation + grounded QA |
| **Memory** | Sliding-window conversation buffer (configurable `k`) |
| **Evaluation** | RAGAS — faithfulness, answer relevancy, context precision |
| **API** | FastAPI with Pydantic v2 request/response models |
| **Deployment** | Docker + docker-compose; Azure Container Apps ready |

---

## Architecture

```
User Query
    │
    ▼
FastAPI /chat endpoint
    │
    ├─► ConversationBufferWindowMemory (session history)
    │
    ├─► CoT Condense Prompt  ──► Azure OpenAI (standalone Q)
    │
    ├─► FAISS Retriever (MMR, top-k chunks)
    │         │
    │         └─ HuggingFace Embeddings (sentence-transformers)
    │
    ├─► QA Prompt + context ──► Azure OpenAI (GPT-4) → Answer
    │
    └─► (optional) RAGAS Evaluation
            ├─ Faithfulness
            ├─ Answer Relevancy
            └─ Context Precision
```

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/Gaurav2904/enterprise-rag-chatbot.git
cd enterprise-rag-chatbot
cp .env.example .env
# Fill in your Azure OpenAI credentials in .env
```

### 2. Run with Docker

```bash
docker-compose up --build
```

The API is now live at `http://localhost:8000`.

### 3. Ingest documents

```bash
# From inside the container (or locally after pip install -r requirements.txt)
python scripts/ingest.py --files data/my_document.pdf --chunk-size 512
```

### 4. Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key findings in the quarterly report?",
    "top_k": 5,
    "evaluate": true
  }'
```

---

## API Reference

### `POST /chat`

| Field | Type | Description |
|---|---|---|
| `query` | string | User question (required) |
| `conversation_id` | string | Session ID for multi-turn (auto-generated if omitted) |
| `top_k` | int | Chunks to retrieve (default: 5) |
| `evaluate` | bool | Run RAGAS evaluation (default: false) |

### `POST /ingest`

| Field | Type | Description |
|---|---|---|
| `file_paths` | list[str] | Paths to PDF / TXT / DOCX files |
| `chunk_size` | int | Token chunk size (default: 512) |
| `chunk_overlap` | int | Overlap between chunks (default: 64) |

### `GET /health` — liveness probe
### `GET /stats` — FAISS index statistics
### `DELETE /conversations/{id}` — clear session memory

---

## Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in credentials

# Run dev server with hot-reload
uvicorn app.main:app --reload

# Run tests
pytest tests/ -v
```

---

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Description |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | API key |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name (e.g. `gpt-4`) |
| `EMBED_MODEL` | HuggingFace model ID for embeddings |
| `MEMORY_WINDOW_K` | Conversation turns to keep in context |
| `FAISS_INDEX_PATH` | Path for persistent FAISS index |

---

## Evaluation Results (sample)

| Metric | Score |
|---|---|
| Faithfulness | 0.91 |
| Answer Relevancy | 0.88 |
| Context Precision | 0.79 |

