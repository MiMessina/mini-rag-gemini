"""
Mini RAG con Gemini API + ChromaDB + Streamlit.
Version con interfaz grafica.

Uso:
    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import chromadb

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

if not GEMINI_API_KEY:
    st.error("Falta GEMINI_API_KEY en el archivo .env")
    st.stop()

# Cliente Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
COLLECTION_NAME = "documentos"


def extraer_texto_pdf(archivo) -> str:
    """Lee un PDF desde un archivo subido y devuelve el texto."""
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
        fin = inicio + tamano_chunk
        chunks.append(texto[inicio:fin])
        inicio += tamano_chunk - overlap
    return chunks


def generar_embedding(texto: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texto,
    )
    return response.embeddings[0].values


def ingestar_documento(archivo, nombre: str, progress_bar) -> int:
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
        progress_bar.progress((i + 1) / len(chunks), text=f"Procesado {i + 1}/{len(chunks)}")
    return len(chunks)


def buscar_y_responder(pregunta: str, top_k: int = 5) -> tuple[str, list[dict]]:
    coleccion = chroma_client.get_collection(name=COLLECTION_NAME)

    embedding_pregunta = generar_embedding(pregunta)
    resultados = coleccion.query(
        query_embeddings=[embedding_pregunta],
        n_results=top_k,
    )

    chunks = []
    for i, doc in enumerate(resultados["documents"][0]):
        chunks.append({
            "texto": doc,
            "distancia": resultados["distances"][0][i],
        })

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
    return response.text, chunks


# ===== UI Streamlit =====

st.set_page_config(page_title="Mini RAG con Gemini", page_icon="📚", layout="wide")

st.title("📚 Mini RAG con Gemini")
st.markdown("**Demostración técnica de RAG end-to-end** | Stack: Gemini API + ChromaDB + Streamlit")

with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown(f"**Modelo LLM:** `{GEMINI_MODEL}`")
    st.markdown(f"**Modelo Embeddings:** `{EMBEDDING_MODEL}`")
    st.markdown(f"**Vector Store:** ChromaDB (local)")
    st.markdown("---")
    st.markdown("**Flujo RAG:**")
    st.markdown("1. PDF → Texto")
    st.markdown("2. Texto → Chunks (800 chars + 150 overlap)")
    st.markdown("3. Chunks → Embeddings")
    st.markdown("4. Embeddings → ChromaDB")
    st.markdown("5. Pregunta → Embedding → Top-5 chunks")
    st.markdown("6. Chunks + Pregunta → Gemini → Respuesta")

tab1, tab2 = st.tabs(["📄 Subir documento", "💬 Hacer preguntas"])

with tab1:
    st.header("Indexar un PDF")
    archivo = st.file_uploader("Subí un PDF", type=["pdf"])
    if archivo:
        if st.button("Indexar documento", type="primary"):
            progress_bar = st.progress(0, text="Iniciando...")
            with st.spinner("Procesando..."):
                total = ingestar_documento(archivo, archivo.name, progress_bar)
            st.success(f"✅ Listo. {total} chunks indexados.")

with tab2:
    st.header("Hacé tu pregunta")
    pregunta = st.text_input("¿Qué querés saber del documento?")
    if pregunta and st.button("Buscar respuesta", type="primary"):
        with st.spinner("Buscando y generando respuesta..."):
            respuesta, chunks = buscar_y_responder(pregunta)

        st.subheader("Respuesta")
        st.write(respuesta)

        with st.expander("Ver chunks recuperados (debug)"):
            for i, c in enumerate(chunks):
                st.markdown(f"**Chunk {i + 1}** | Distancia: `{c['distancia']:.4f}`")
                st.text(c["texto"][:300] + "...")
                st.markdown("---")
