"""Tests para evaluator.py — funciones puras que no requieren GCP."""

from evaluator import score_retrieval, EvaluationResult, evaluate


def make_chunks(distances):
    return [{"texto": f"chunk {i}", "distancia": d} for i, d in enumerate(distances)]


def test_score_retrieval_vacio():
    avg, div = score_retrieval([])
    assert avg == 0.0
    assert div == 0.0


def test_score_retrieval_un_chunk():
    avg, div = score_retrieval(make_chunks([0.2]))
    assert avg == 0.8
    assert div == 1.0  # un solo chunk siempre es "diverso"


def test_score_retrieval_multiples():
    chunks = make_chunks([0.1, 0.2, 0.3, 0.4, 0.5])
    avg, div = score_retrieval(chunks)
    assert 0.0 < avg < 1.0
    assert 0.0 <= div <= 1.0


def test_score_retrieval_similitud_maxima():
    # distancia 0 → similitud 1.0
    avg, div = score_retrieval(make_chunks([0.0, 0.0]))
    assert avg == 1.0


def test_score_retrieval_distancia_mayor_a_1():
    # similitud nunca es negativa
    avg, div = score_retrieval(make_chunks([1.5]))
    assert avg == 0.0


def test_evaluate_sin_cliente_no_llama_gcp():
    chunks = make_chunks([0.15, 0.25])
    result = evaluate("pregunta", "respuesta", chunks, client=None, model=None)
    assert isinstance(result, EvaluationResult)
    assert result.faithfulness is None
    assert result.avg_similarity > 0
