"""
RAG Evaluation using the RAGAS framework.

Metrics evaluated:
  - Faithfulness      : is the answer grounded in the retrieved context?
  - Answer Relevancy  : how relevant is the answer to the question?
  - Context Precision : are the retrieved chunks relevant?
  - Context Recall    : does the context cover the ground truth? (when GT available)
"""

import logging
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Thin wrapper around the RAGAS evaluation pipeline.

    Usage
    -----
    evaluator = RAGEvaluator()

    # Single-query (no ground truth)
    scores = evaluator.evaluate_single(
        query="What is the capital of France?",
        answer="Paris is the capital of France.",
        contexts=["France is a country in Western Europe. Its capital is Paris."],
    )

    # Batch evaluation (with optional ground truth)
    scores = evaluator.evaluate_batch(
        queries=[...], answers=[...], contexts_list=[[...], ...], ground_truths=[...]
    )
    """

    def __init__(self):
        self._metrics = None  # lazy-load RAGAS to keep startup fast

    def _get_metrics(self):
        if self._metrics is None:
            try:
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                )
                self._metrics = {
                    "faithfulness": faithfulness,
                    "answer_relevancy": answer_relevancy,
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                }
                logger.info("RAGAS metrics loaded.")
            except ImportError:
                logger.warning("RAGAS not installed. pip install ragas to enable evaluation.")
                self._metrics = {}
        return self._metrics

    # ── Single query evaluation (no ground truth required) ────────────────────
    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single query-answer-context triple.

        Returns a dict like:
        {
            "faithfulness": 0.92,
            "answer_relevancy": 0.87,
            "context_precision": 0.75,
        }
        """
        return self.evaluate_batch(
            queries=[query],
            answers=[answer],
            contexts_list=[contexts],
            ground_truths=[ground_truth] if ground_truth else None,
        )

    # ── Batch evaluation ──────────────────────────────────────────────────────
    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        metrics = self._get_metrics()
        if not metrics:
            return {"error": "RAGAS not installed"}

        try:
            from ragas import evaluate

            data = {
                "question": queries,
                "answer": answers,
                "contexts": contexts_list,
            }
            if ground_truths:
                data["ground_truth"] = ground_truths

            dataset = Dataset.from_dict(data)

            active_metrics = [
                metrics["faithfulness"],
                metrics["answer_relevancy"],
                metrics["context_precision"],
            ]
            if ground_truths:
                active_metrics.append(metrics["context_recall"])

            result = evaluate(dataset, metrics=active_metrics)
            scores = {k: round(float(v), 4) for k, v in result.items()}
            logger.info("RAGAS scores: %s", scores)
            return scores

        except Exception as exc:
            logger.error("RAGAS evaluation failed: %s", exc)
            return {"error": str(exc)}

    # ── Threshold-based pass/fail ─────────────────────────────────────────────
    @staticmethod
    def passes_quality_bar(
        scores: Dict[str, float],
        min_faithfulness: float = 0.75,
        min_answer_relevancy: float = 0.70,
    ) -> bool:
        """Return True if the answer meets minimum quality thresholds."""
        return (
            scores.get("faithfulness", 0) >= min_faithfulness
            and scores.get("answer_relevancy", 0) >= min_answer_relevancy
        )
