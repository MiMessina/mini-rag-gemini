"""
Mini RAG con Gemini API + ChromaDB.
Version CLI simple - sin interfaz grafica.

Uso:
    python cli.py ingest mi_documento.pdf
    python cli.py query "Cual es la pregunta?"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import chromadb

# Carga variables de entorno desde .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

if not GEMINI_API_KEY:
    print("ERROR: falta GEMINI_API_KEY en el archivo .env")
    sys.exit(1)

# Cliente de Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# ChromaDB local persistente
chroma_client = chromadb.PersistentClient(path="./chroma_db")
COLLECTION_NAME = "documentos"


def extraer_texto_pdf(ruta_pdf: str) -> str:
    """Lee un PDF y devuelve todo el texto."""
    reader = PdfReader(ruta_pdf)
    texto = ""
    for pagina in reader.pages:
        contenido = pagina.extract_text()
        if contenido:
            texto += contenido + "\n"
    return texto


def chunking(texto: str, tamano_chunk: int = 800, overlap: int = 150) -> list[str]:
    """
    Parte el texto en chunks de tamano_chunk caracteres
    con overlap de overlap caracteres entre chunks.
    """
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamano_chunk
        chunk = texto[inicio:fin]
        chunks.append(chunk)
        inicio += tamano_chunk - overlap
    return chunks


def generar_embedding(texto: str) -> list[float]:
    """
    Genera un embedding usando el modelo text-embedding-004 de Google.
    Devuelve un vector de 768 dimensiones.
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texto,
    )
    return response.embeddings[0].values


def ingestar(ruta_pdf: str) -> None:
    """
    Lee un PDF, hace chunking, genera embeddings y guarda todo en ChromaDB.
    Esto se hace UNA SOLA VEZ por documento.
    """
    print(f"Leyendo PDF: {ruta_pdf}")
    texto = extraer_texto_pdf(ruta_pdf)
    print(f"  Texto extraido: {len(texto)} caracteres")

    chunks = chunking(texto)
    print(f"  Chunks generados: {len(chunks)}")

    # Borra coleccion previa si existe, para empezar limpio
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
        print(f"  Procesado chunk {i + 1}/{len(chunks)}")

    print(f"Listo. Total de chunks indexados: {len(chunks)}")


def buscar_chunks_relevantes(pregunta: str, top_k: int = 5) -> list[dict]:
    """
    Busca los chunks mas similares a la pregunta en ChromaDB.
    """
    try:
        coleccion = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("ERROR: no hay coleccion creada. Ejecuta primero: python cli.py ingest <pdf>")
        sys.exit(1)

    embedding_pregunta = generar_embedding(pregunta)
    resultados = coleccion.query(
        query_embeddings=[embedding_pregunta],
        n_results=top_k,
    )

    chunks_relevantes = []
    for i, documento in enumerate(resultados["documents"][0]):
        chunks_relevantes.append({
            "texto": documento,
            "distancia": resultados["distances"][0][i],
        })
    return chunks_relevantes


def generar_respuesta(pregunta: str, chunks: list[dict]) -> str:
    """
    Arma el prompt con los chunks recuperados y llama a Gemini.
    """
    contexto = "\n\n---\n\n".join([c["texto"] for c in chunks])

    prompt = f"""Sos un asistente que responde preguntas usando exclusivamente el contexto provisto.
Si la respuesta no esta en el contexto, deci honestamente que no tenes esa informacion.

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def query(pregunta: str) -> None:
    """
    Hace una pregunta al sistema RAG.
    """
    print(f"\nPregunta: {pregunta}\n")
    print("Buscando chunks relevantes...")
    chunks = buscar_chunks_relevantes(pregunta)

    print(f"Chunks recuperados: {len(chunks)}")
    print("Generando respuesta con Gemini...\n")
    respuesta = generar_respuesta(pregunta, chunks)

    print("=" * 60)
    print("RESPUESTA:")
    print("=" * 60)
    print(respuesta)
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python cli.py ingest <archivo.pdf>")
        print("  python cli.py query \"tu pregunta aca\"")
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
        pregunta = " ".join(sys.argv[2:])
        query(pregunta)

    else:
        print(f"Comando desconocido: {comando}")
        sys.exit(1)


if __name__ == "__main__":
    main()
