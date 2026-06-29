"""
API REST del pipeline RAG.

Endpoints:
    GET  /health   — estado del servicio
    POST /ingest   — indexar un PDF
    POST /query    — hacer una pregunta al documento indexado

Uso:
    uvicorn api:app --reload
"""

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from evaluator import evaluate
from logger import log_query
from rag import (
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    buscar_y_responder,
    client,
    ingestar_documento,
)

app = FastAPI(title="Mini RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    chunks_retrieved: int
    avg_similarity: float
    diversity: float
    faithfulness: float | None


class IngestResponse(BaseModel):
    document: str
    chunks_indexed: int


@app.get("/health")
def health():
    return {"status": "ok", "llm_model": GEMINI_MODEL, "embedding_model": EMBEDDING_MODEL}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF.")

    content = await file.read()
    total = ingestar_documento(io.BytesIO(content), file.filename)
    return IngestResponse(document=file.filename, chunks_indexed=total)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        answer, chunks = buscar_y_responder(request.question, top_k=request.top_k)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="No hay documentos indexados. Primero subí un PDF con POST /ingest.",
        )

    eval_result = evaluate(request.question, answer, chunks, client=client, model=GEMINI_MODEL)
    log_query(request.question, chunks, answer, eval_result)

    return QueryResponse(
        answer=answer,
        chunks_retrieved=len(chunks),
        avg_similarity=eval_result.avg_similarity,
        diversity=eval_result.diversity,
        faithfulness=eval_result.faithfulness,
    )
