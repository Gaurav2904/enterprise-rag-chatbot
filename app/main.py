"""
Enterprise RAG Chatbot — FastAPI entry point
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import logging
import time
import uuid

from app.rag_pipeline import RAGPipeline
from app.vector_store import VectorStoreManager
from app.evaluation import RAGEvaluator

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise RAG Chatbot",
    description="Production-grade Retrieval-Augmented Generation chatbot powered by Azure OpenAI + FAISS",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (initialised on startup) ──────────────────────────────────────
rag_pipeline: RAGPipeline = None
vector_store_manager: VectorStoreManager = None
evaluator: RAGEvaluator = None


@app.on_event("startup")
async def startup_event():
    global rag_pipeline, vector_store_manager, evaluator
    logger.info("Initialising RAG pipeline …")
    vector_store_manager = VectorStoreManager()
    rag_pipeline = RAGPipeline(vector_store_manager)
    evaluator = RAGEvaluator()
    logger.info("RAG pipeline ready.")


# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    evaluate: bool = Field(False, description="Run RAGAS evaluation on this query")


class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: float


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: List[SourceDocument]
    latency_ms: float
    evaluation: Optional[dict] = None


class IngestRequest(BaseModel):
    file_paths: List[str] = Field(..., description="Paths to documents to ingest")
    chunk_size: int = Field(512, ge=128, le=2048)
    chunk_overlap: int = Field(64, ge=0, le=512)


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    duration_ms: float


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "enterprise-rag-chatbot"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main RAG chat endpoint with optional RAGAS evaluation."""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    t0 = time.perf_counter()

    try:
        result = rag_pipeline.query(
            query=request.query,
            conversation_id=conversation_id,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.exception("RAG pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = (time.perf_counter() - t0) * 1000

    evaluation = None
    if request.evaluate:
        try:
            evaluation = evaluator.evaluate_single(
                query=request.query,
                answer=result["answer"],
                contexts=[doc["content"] for doc in result["sources"]],
            )
        except Exception:
            logger.warning("RAGAS evaluation failed; skipping.")

    return ChatResponse(
        conversation_id=conversation_id,
        answer=result["answer"],
        sources=[SourceDocument(**doc) for doc in result["sources"]],
        latency_ms=round(latency_ms, 2),
        evaluation=evaluation,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest documents into the FAISS vector store."""
    t0 = time.perf_counter()
    try:
        chunks_indexed = vector_store_manager.ingest(
            file_paths=request.file_paths,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except Exception as exc:
        logger.exception("Ingestion error")
        raise HTTPException(status_code=500, detail=str(exc))

    return IngestResponse(
        status="success",
        chunks_indexed=chunks_indexed,
        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation memory for a given session."""
    rag_pipeline.clear_memory(conversation_id)
    return {"status": "cleared", "conversation_id": conversation_id}


@app.get("/stats")
async def vector_store_stats():
    """Return basic stats about the FAISS index."""
    return vector_store_manager.stats()


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
