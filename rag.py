"""
Núcleo del pipeline RAG: ingestión, embeddings y consulta.
Compartido entre la UI de Streamlit (app.py) y la API REST (api.py).
"""

import os
from typing import Callable

import chromadb
import google.genai as genai
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")

client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
COLLECTION_NAME = "documentos"


def extraer_texto_pdf(archivo) -> str:
    reader = PdfReader(archivo)
    texto = ""
    for pagina in reader.pages:
        contenido = pagina.extract_text()
        if contenido:
            texto += contenido + "\n"
    return texto


def chunking(texto: str, tamano_chunk: int = 800, overlap: int = 150) -> list[str]:
    chunks = []
    inicio = 0
    while inicio < len(texto):
        chunks.append(texto[inicio:inicio + tamano_chunk])
        inicio += tamano_chunk - overlap
    return chunks


def generar_embedding(texto: str) -> list[float]:
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=[texto])
    return result.embeddings[0].values


def ingestar_documento(
    archivo,
    nombre: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    texto = extraer_texto_pdf(archivo)
    chunks = chunking(texto)

    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    coleccion = chroma_client.create_collection(name=COLLECTION_NAME)

    for i, chunk in enumerate(chunks):
        embedding = generar_embedding(chunk)
        coleccion.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"chunk_id": i, "source": nombre}],
        )
        if progress_callback:
            progress_callback(i + 1, len(chunks))
    return len(chunks)


def buscar_y_responder(pregunta: str, top_k: int = 5) -> tuple[str, list[dict]]:
    coleccion = chroma_client.get_collection(name=COLLECTION_NAME)

    embedding_pregunta = generar_embedding(pregunta)
    resultados = coleccion.query(query_embeddings=[embedding_pregunta], n_results=top_k)

    chunks = [
        {"texto": doc, "distancia": resultados["distances"][0][i]}
        for i, doc in enumerate(resultados["documents"][0])
    ]

    contexto = "\n\n---\n\n".join(c["texto"] for c in chunks)
    prompt = f"""Sos un asistente que responde preguntas usando exclusivamente el contexto provisto.
Si la respuesta no está en el contexto, decí honestamente que no tenés esa información.

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""

    respuesta = client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text
    return respuesta, chunks
