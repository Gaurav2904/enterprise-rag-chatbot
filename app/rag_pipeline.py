"""
RAG Pipeline — orchestrates retrieval + Azure OpenAI generation with
Chain-of-Thought prompting and context-window management.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import AzureChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from app.vector_store import VectorStoreManager
from app.prompts import SYSTEM_PROMPT, COT_CONDENSE_PROMPT, QA_PROMPT

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline:
      1. Retrieve top-k chunks from FAISS via MMR or similarity search
      2. Condense follow-up questions using the conversation history (CoT prompt)
      3. Generate a grounded answer with GPT-4 via Azure OpenAI
      4. Manage per-session context window (sliding-window memory)
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vsm = vector_store_manager
        self.llm = self._build_llm()
        self._memories: Dict[str, ConversationBufferWindowMemory] = {}
        logger.info("RAGPipeline initialised.")

    # ── LLM ──────────────────────────────────────────────────────────────────
    def _build_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "1024")),
            streaming=False,
        )

    # ── Memory ────────────────────────────────────────────────────────────────
    def _get_memory(self, conversation_id: str) -> ConversationBufferWindowMemory:
        """Return (or create) a sliding-window memory for the session."""
        if conversation_id not in self._memories:
            self._memories[conversation_id] = ConversationBufferWindowMemory(
                k=int(os.environ.get("MEMORY_WINDOW_K", "6")),
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
        return self._memories[conversation_id]

    def clear_memory(self, conversation_id: str) -> None:
        self._memories.pop(conversation_id, None)

    # ── Chain ─────────────────────────────────────────────────────────────────
    def _build_chain(
        self,
        conversation_id: str,
        top_k: int,
    ) -> ConversationalRetrievalChain:
        retriever = self.vsm.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": top_k * 3, "lambda_mult": 0.7},
        )
        memory = self._get_memory(conversation_id)

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            condense_question_prompt=COT_CONDENSE_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
            verbose=bool(int(os.environ.get("CHAIN_VERBOSE", "0"))),
        )
        return chain

    # ── Public API ────────────────────────────────────────────────────────────
    def query(
        self,
        query: str,
        conversation_id: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Run a RAG query and return answer + source metadata.

        Returns
        -------
        {
            "answer": str,
            "sources": [{"content": str, "source": str, "page": int, "score": float}]
        }
        """
        logger.info("Query [%s]: %s", conversation_id, query[:120])
        chain = self._build_chain(conversation_id, top_k)

        result = chain.invoke({"question": query})

        answer: str = result["answer"]
        raw_docs = result.get("source_documents", [])

        sources = self._format_sources(raw_docs)
        logger.info("Answer generated (%d tokens, %d sources)", len(answer.split()), len(sources))
        return {"answer": answer, "sources": sources}

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _format_sources(docs) -> List[Dict[str, Any]]:
        seen = set()
        sources = []
        for doc in docs:
            meta = doc.metadata or {}
            key = (meta.get("source", ""), meta.get("page", 0))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                {
                    "content": doc.page_content[:500],
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page"),
                    "score": float(meta.get("score", 0.0)),
                }
            )
        return sources
