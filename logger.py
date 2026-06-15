"""
Módulo de observabilidad: logging de queries, respuestas y métricas a JSONL.
No requiere GCP — escribe localmente en logs/queries.jsonl.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from evaluator import EvaluationResult

LOGS_DIR = Path("logs")
QUERIES_LOG = LOGS_DIR / "queries.jsonl"


def _ensure_logs_dir() -> None:
    LOGS_DIR.mkdir(exist_ok=True)


def log_query(
    question: str,
    chunks: list[dict],
    answer: str,
    eval_result: EvaluationResult,
) -> None:
    """Registra una query completa con sus métricas en el archivo JSONL."""
    _ensure_logs_dir()

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": question,
        "answer": answer,
        "chunks_count": len(chunks),
        "avg_similarity": eval_result.avg_similarity,
        "diversity": eval_result.diversity,
        "faithfulness": eval_result.faithfulness,
        "chunks_distances": [round(c["distancia"], 4) for c in chunks],
    }

    with open(QUERIES_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_stats() -> dict:
    """
    Lee el log JSONL y devuelve estadísticas agregadas.

    Retorna:
        total_queries, avg_faithfulness, avg_similarity, recent (últimas 10 entradas)
    """
    _ensure_logs_dir()

    if not QUERIES_LOG.exists():
        return {
            "total_queries": 0,
            "avg_faithfulness": None,
            "avg_similarity": 0.0,
            "recent": [],
        }

    entries = []
    with open(QUERIES_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not entries:
        return {
            "total_queries": 0,
            "avg_faithfulness": None,
            "avg_similarity": 0.0,
            "recent": [],
        }

    faithfulness_scores = [e["faithfulness"] for e in entries if e.get("faithfulness") is not None]
    avg_faithfulness = (
        round(sum(faithfulness_scores) / len(faithfulness_scores), 2)
        if faithfulness_scores else None
    )

    similarities = [e["avg_similarity"] for e in entries if e.get("avg_similarity") is not None]
    avg_similarity = round(sum(similarities) / len(similarities), 4) if similarities else 0.0

    return {
        "total_queries": len(entries),
        "avg_faithfulness": avg_faithfulness,
        "avg_similarity": avg_similarity,
        "recent": entries[-10:],
    }
