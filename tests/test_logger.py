"""Tests para logger.py — I/O local sin dependencias de GCP."""

import json
from unittest.mock import patch

from evaluator import EvaluationResult
import logger


def make_eval(avg_sim=0.8, diversity=0.6, faithfulness=4.0):
    return EvaluationResult(avg_similarity=avg_sim, diversity=diversity, faithfulness=faithfulness)


def test_get_stats_sin_logs(tmp_path):
    with patch.object(logger, "LOGS_DIR", tmp_path), \
         patch.object(logger, "QUERIES_LOG", tmp_path / "queries.jsonl"):
        stats = logger.get_stats()

    assert stats["total_queries"] == 0
    assert stats["avg_faithfulness"] is None
    assert stats["avg_similarity"] == 0.0
    assert stats["recent"] == []


def test_log_y_get_stats(tmp_path):
    queries_log = tmp_path / "queries.jsonl"

    with patch.object(logger, "LOGS_DIR", tmp_path), \
         patch.object(logger, "QUERIES_LOG", queries_log):

        chunks = [{"texto": "contexto relevante", "distancia": 0.2}]
        logger.log_query("¿Qué es RAG?", chunks, "RAG es búsqueda + generación.", make_eval())
        logger.log_query("¿Qué es ChromaDB?", chunks, "ChromaDB es un vector store.", make_eval(faithfulness=None))

        stats = logger.get_stats()

    assert stats["total_queries"] == 2
    assert stats["avg_similarity"] == 0.8
    assert stats["avg_faithfulness"] == 4.0  # solo la que tiene faithfulness
    assert len(stats["recent"]) == 2


def test_log_escribe_jsonl_valido(tmp_path):
    queries_log = tmp_path / "queries.jsonl"

    with patch.object(logger, "LOGS_DIR", tmp_path), \
         patch.object(logger, "QUERIES_LOG", queries_log):
        chunks = [{"texto": "texto", "distancia": 0.1}]
        logger.log_query("pregunta", chunks, "respuesta", make_eval())

    lines = queries_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["question"] == "pregunta"
    assert entry["chunks_count"] == 1
    assert "timestamp" in entry
