"""
Mini RAG con Vertex AI + ChromaDB.
Versión CLI — sin interfaz gráfica.

Uso:
    python cli.py ingest mi_documento.pdf
    python cli.py query "¿Cuál es la pregunta?"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from pypdf import PdfReader
import chromadb

from evaluator import evaluate
from logger import log_query

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")

if not GCP_PROJECT_ID:
    print("ERROR: falta GCP_PROJECT_ID en el archivo .env")
    print("Copiá .env.example a .env y completá los valores.")
    sys.exit(1)

vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
llm = GenerativeModel(GEMINI_MODEL)
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
COLLECTION_NAME = "documentos"


def extraer_texto_pdf(ruta_pdf: str) -> str:
    reader = PdfReader(ruta_pdf)
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
    embeddings = embedding_model.get_embeddings([texto])
    return embeddings[0].values


def ingestar(ruta_pdf: str) -> None:
    print(f"Leyendo PDF: {ruta_pdf}")
    texto = extraer_texto_pdf(ruta_pdf)
    print(f"  Texto extraído: {len(texto)} caracteres")

    chunks = chunking(texto)
    print(f"  Chunks generados: {len(chunks)}")

    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    coleccion = chroma_client.create_collection(name=COLLECTION_NAME)

    print("Generando embeddings y guardando en ChromaDB...")
    for i, chunk in enumerate(chunks):
        embedding = generar_embedding(chunk)
        coleccion.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"chunk_id": i, "source": ruta_pdf}],
        )
        print(f"  Chunk {i + 1}/{len(chunks)}")

    print(f"Listo. {len(chunks)} chunks indexados.")


def buscar_chunks_relevantes(pregunta: str, top_k: int = 5) -> list[dict]:
    try:
        coleccion = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("ERROR: no hay colección creada. Ejecutá primero: python cli.py ingest <pdf>")
        sys.exit(1)

    embedding_pregunta = generar_embedding(pregunta)
    resultados = coleccion.query(query_embeddings=[embedding_pregunta], n_results=top_k)

    return [
        {"texto": doc, "distancia": resultados["distances"][0][i]}
        for i, doc in enumerate(resultados["documents"][0])
    ]


def generar_respuesta(pregunta: str, chunks: list[dict]) -> str:
    contexto = "\n\n---\n\n".join(c["texto"] for c in chunks)
    prompt = f"""Sos un asistente que responde preguntas usando exclusivamente el contexto provisto.
Si la respuesta no está en el contexto, decí honestamente que no tenés esa información.

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""
    return llm.generate_content(prompt).text


def query(pregunta: str) -> None:
    print(f"\nPregunta: {pregunta}\n")
    print("Buscando chunks relevantes...")
    chunks = buscar_chunks_relevantes(pregunta)

    print("Generando respuesta con Vertex AI...\n")
    respuesta = generar_respuesta(pregunta, chunks)

    print("=" * 60)
    print("RESPUESTA:")
    print("=" * 60)
    print(respuesta)

    # Evaluación y logging
    eval_result = evaluate(pregunta, respuesta, chunks, model=llm)
    log_query(pregunta, chunks, respuesta, eval_result)

    print("=" * 60)
    print(f"Métricas — Similitud promedio: {eval_result.avg_similarity:.4f} | "
          f"Diversidad: {eval_result.diversity:.2f} | "
          f"Faithfulness: {eval_result.faithfulness or 'N/A'}/5")


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python cli.py ingest <archivo.pdf>")
        print("  python cli.py query \"tu pregunta acá\"")
        sys.exit(1)

    comando = sys.argv[1]

    if comando == "ingest":
        if len(sys.argv) < 3:
            print("Falta especificar el archivo PDF")
            sys.exit(1)
        ruta = sys.argv[2]
        if not Path(ruta).exists():
            print(f"No existe el archivo: {ruta}")
            sys.exit(1)
        ingestar(ruta)

    elif comando == "query":
        if len(sys.argv) < 3:
            print("Falta especificar la pregunta")
            sys.exit(1)
        query(" ".join(sys.argv[2:]))

    else:
        print(f"Comando desconocido: {comando}")
        sys.exit(1)


if __name__ == "__main__":
    main()
