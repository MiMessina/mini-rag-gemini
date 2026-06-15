"""
Módulo de evaluación de calidad RAG.
Funciona sin conexión a GCP — solo necesita el cliente LLM para faithfulness.
"""

import json
from dataclasses import dataclass

import google.genai as genai


@dataclass
class EvaluationResult:
    avg_similarity: float
    diversity: float
    faithfulness: float | None  # None si no se ejecuta LLM-as-judge


def score_retrieval(chunks: list[dict]) -> tuple[float, float]:
    """
    Calcula métricas de retrieval a partir de los chunks recuperados.

    Retorna:
        avg_similarity: promedio de similitud (1 - distancia coseno)
        diversity: fracción de chunks con similitud < 0.95 entre sí (no redundantes)
    """
    if not chunks:
        return 0.0, 0.0

    similarities = [max(0.0, 1.0 - c["distancia"]) for c in chunks]
    avg_similarity = sum(similarities) / len(similarities)

    # Diversidad: cuántos chunks tienen similitud < 0.95 con el mejor chunk
    best = max(similarities)
    diverse = sum(1 for s in similarities if abs(s - best) > 0.05)
    diversity = diverse / len(similarities) if len(similarities) > 1 else 1.0

    return round(avg_similarity, 4), round(diversity, 4)


def score_faithfulness(
    question: str,
    answer: str,
    context: str,
    client: genai.Client,
    model: str,
) -> float | None:
    """
    LLM-as-judge: le pide al modelo que evalúe si la respuesta es fiel al contexto.
    Retorna un puntaje del 1 al 5, o None si falla.
    """
    prompt = f"""Sos un evaluador de sistemas RAG. Tu tarea es evaluar si una respuesta
es fiel al contexto provisto, es decir, si se basa exclusivamente en la información
del contexto y no inventa datos.

Contexto:
{context[:2000]}

Pregunta: {question}

Respuesta evaluada: {answer}

Respondé ÚNICAMENTE con un JSON válido con este formato exacto:
{{"score": <número del 1 al 5>, "razon": "<una línea explicando el puntaje>"}}

Escala:
5 - La respuesta se basa completamente en el contexto
4 - La respuesta usa el contexto con pequeñas inferencias razonables
3 - La respuesta mezcla contexto con información externa
2 - La respuesta usa poco el contexto
1 - La respuesta ignora el contexto o inventa información"""

    try:
        response = client.models.generate_content(model=model, contents=prompt)
        text = response.text.strip()
        # Limpiar posibles bloques de código markdown
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        score = float(data.get("score", 0))
        return round(min(max(score, 1.0), 5.0), 2)
    except Exception:
        return None


def evaluate(
    question: str,
    answer: str,
    chunks: list[dict],
    client: genai.Client | None = None,
    model: str | None = None,
) -> EvaluationResult:
    """Evalúa una respuesta RAG completa."""
    avg_similarity, diversity = score_retrieval(chunks)

    faithfulness = None
    if client is not None and model is not None:
        context = "\n\n---\n\n".join(c["texto"] for c in chunks)
        faithfulness = score_faithfulness(question, answer, context, client, model)

    return EvaluationResult(
        avg_similarity=avg_similarity,
        diversity=diversity,
        faithfulness=faithfulness,
    )
