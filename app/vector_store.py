"""
VectorStoreManager — wraps FAISS + HuggingFace sentence-transformers embeddings.
Handles ingestion, persistence, and retriever creation.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "data/faiss_index")
EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"
)
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")


class VectorStoreManager:
    """
    Manages a persistent FAISS vector store with HuggingFace embeddings.

    Features
    --------
    - Supports PDF, TXT, and DOCX ingestion
    - Saves / loads the FAISS index from disk automatically
    - Exposes an LangChain-compatible retriever
    """

    def __init__(self):
        self.index_path = Path(FAISS_INDEX_PATH)
        self.embeddings = self._load_embeddings()
        self.vectorstore: Optional[FAISS] = self._load_index()

    # ── Embeddings ────────────────────────────────────────────────────────────
    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        logger.info("Loading embedding model: %s (device=%s)", EMBED_MODEL, EMBED_DEVICE)
        return HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": EMBED_DEVICE},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load_index(self) -> Optional[FAISS]:
        if self.index_path.exists():
            logger.info("Loading existing FAISS index from %s", self.index_path)
            try:
                vs = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("FAISS index loaded (%d vectors).", vs.index.ntotal)
                return vs
            except Exception:
                logger.warning("Failed to load existing index; starting fresh.")
        return None

    def _save_index(self):
        if self.vectorstore is None:
            return
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_path))
        logger.info("FAISS index saved to %s.", self.index_path)

    # ── Document loading ──────────────────────────────────────────────────────
    @staticmethod
    def _load_documents(file_paths: List[str]) -> List[Document]:
        docs: List[Document] = []
        for path in file_paths:
            p = Path(path)
            if not p.exists():
                logger.warning("File not found: %s — skipping.", path)
                continue
            ext = p.suffix.lower()
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(str(p))
                elif ext in (".docx", ".doc"):
                    loader = UnstructuredWordDocumentLoader(str(p))
                else:
                    loader = TextLoader(str(p), encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata.setdefault("source", str(p))
                docs.extend(loaded)
                logger.info("Loaded %d pages from %s", len(loaded), p.name)
            except Exception as exc:
                logger.error("Error loading %s: %s", path, exc)
        return docs

    # ── Ingestion ─────────────────────────────────────────────────────────────
    def ingest(
        self,
        file_paths: List[str],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> int:
        """
        Load, chunk, embed, and index the supplied files.

        Returns
        -------
        int  — number of chunks indexed
        """
        docs = self._load_documents(file_paths)
        if not docs:
            raise ValueError("No documents could be loaded from the given paths.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info("Split into %d chunks (size=%d, overlap=%d).", len(chunks), chunk_size, chunk_overlap)

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        self._save_index()
        return len(chunks)

    # ── Retriever ─────────────────────────────────────────────────────────────
    def as_retriever(self, search_type: str = "mmr", search_kwargs: Optional[dict] = None):
        if self.vectorstore is None:
            raise RuntimeError("Vector store is empty. Please ingest documents first.")
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs or {"k": 5},
        )

    # ── Stats ─────────────────────────────────────────────────────────────────
    def stats(self) -> Dict:
        if self.vectorstore is None:
            return {"status": "empty", "total_vectors": 0}
        return {
            "status": "ready",
            "total_vectors": self.vectorstore.index.ntotal,
            "embedding_model": EMBED_MODEL,
            "index_path": str(self.index_path),
        }
