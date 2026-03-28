"""
Unit & integration tests for the RAG Chatbot.
Run with: pytest tests/ -v
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Inject dummy Azure creds so imports don't fail in CI."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    monkeypatch.setenv("FAISS_INDEX_PATH", "/tmp/test_faiss_index")


# ── VectorStoreManager tests ──────────────────────────────────────────────────
class TestVectorStoreManager:

    @patch("app.vector_store.HuggingFaceEmbeddings")
    def test_init_empty(self, mock_embed):
        """Manager initialises with no index when path doesn't exist."""
        from app.vector_store import VectorStoreManager
        mock_embed.return_value = MagicMock()
        vsm = VectorStoreManager()
        assert vsm.vectorstore is None

    @patch("app.vector_store.HuggingFaceEmbeddings")
    @patch("app.vector_store.FAISS")
    def test_ingest_creates_index(self, mock_faiss, mock_embed, tmp_path):
        """Ingesting a text file creates a FAISS index."""
        # Create a dummy text file
        doc = tmp_path / "sample.txt"
        doc.write_text("Paris is the capital of France." * 50)

        from app.vector_store import VectorStoreManager
        mock_embed.return_value = MagicMock()
        mock_faiss.from_documents.return_value = MagicMock(index=MagicMock(ntotal=5))

        vsm = VectorStoreManager()
        n = vsm.ingest(file_paths=[str(doc)])
        assert n > 0
        mock_faiss.from_documents.assert_called_once()

    @patch("app.vector_store.HuggingFaceEmbeddings")
    def test_stats_empty(self, mock_embed):
        from app.vector_store import VectorStoreManager
        mock_embed.return_value = MagicMock()
        vsm = VectorStoreManager()
        stats = vsm.stats()
        assert stats["status"] == "empty"
        assert stats["total_vectors"] == 0

    @patch("app.vector_store.HuggingFaceEmbeddings")
    def test_as_retriever_raises_without_index(self, mock_embed):
        from app.vector_store import VectorStoreManager
        mock_embed.return_value = MagicMock()
        vsm = VectorStoreManager()
        with pytest.raises(RuntimeError, match="empty"):
            vsm.as_retriever()


# ── RAGEvaluator tests ────────────────────────────────────────────────────────
class TestRAGEvaluator:

    def test_passes_quality_bar_positive(self):
        from app.evaluation import RAGEvaluator
        evaluator = RAGEvaluator()
        scores = {"faithfulness": 0.9, "answer_relevancy": 0.85}
        assert evaluator.passes_quality_bar(scores) is True

    def test_passes_quality_bar_negative(self):
        from app.evaluation import RAGEvaluator
        evaluator = RAGEvaluator()
        scores = {"faithfulness": 0.5, "answer_relevancy": 0.85}
        assert evaluator.passes_quality_bar(scores) is False

    def test_evaluate_single_no_ragas(self):
        """Gracefully returns error dict when RAGAS not installed."""
        from app.evaluation import RAGEvaluator
        evaluator = RAGEvaluator()
        evaluator._metrics = {}  # simulate RAGAS not installed
        result = evaluator.evaluate_single("q", "a", ["ctx"])
        assert "error" in result


# ── Prompt template tests ─────────────────────────────────────────────────────
class TestPrompts:

    def test_cot_prompt_contains_placeholders(self):
        from app.prompts import COT_CONDENSE_PROMPT
        assert "chat_history" in COT_CONDENSE_PROMPT.input_variables
        assert "question" in COT_CONDENSE_PROMPT.input_variables

    def test_qa_prompt_contains_placeholders(self):
        from app.prompts import QA_PROMPT
        assert "context" in QA_PROMPT.input_variables
        assert "question" in QA_PROMPT.input_variables

    def test_qa_prompt_format(self):
        from app.prompts import QA_PROMPT
        rendered = QA_PROMPT.format(context="Some context.", question="What is it?")
        assert "Some context." in rendered
        assert "What is it?" in rendered


# ── FastAPI route tests ───────────────────────────────────────────────────────
class TestHealthEndpoint:

    def test_health_returns_ok(self):
        from fastapi.testclient import TestClient
        with patch("app.main.RAGPipeline"), \
             patch("app.main.VectorStoreManager"), \
             patch("app.main.RAGEvaluator"):
            from app.main import app
            client = TestClient(app)
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
