"""
Mini RAG con Vertex AI + ChromaDB + Streamlit.
Versión con interfaz gráfica y métricas de observabilidad.

Uso:
    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from pypdf import PdfReader
import chromadb

from evaluator import evaluate
from logger import log_query, get_stats

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")

if not GCP_PROJECT_ID:
    st.error("Falta GCP_PROJECT_ID en el archivo .env — copiá .env.example a .env y completá los valores.")
    st.stop()

vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
llm = GenerativeModel(GEMINI_MODEL)
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

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
    embeddings = embedding_model.get_embeddings([texto])
    return embeddings[0].values


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

    respuesta = llm.generate_content(prompt).text
    return respuesta, chunks


# ===== UI Streamlit =====

st.set_page_config(page_title="Mini RAG con Vertex AI", page_icon="📚", layout="wide")

st.title("📚 Mini RAG con Vertex AI")
st.markdown("**Demostración técnica de RAG end-to-end** | Stack: Vertex AI + ChromaDB + Streamlit")

with st.sidebar:
    st.header("⚙️ Configuración")
    st.markdown(f"**Modelo LLM:** `{GEMINI_MODEL}`")
    st.markdown(f"**Modelo Embeddings:** `{EMBEDDING_MODEL}`")
    st.markdown(f"**Proyecto GCP:** `{GCP_PROJECT_ID}`")
    st.markdown(f"**Región:** `{GCP_LOCATION}`")
    st.markdown(f"**Vector Store:** ChromaDB (local)")
    st.markdown("---")
    st.markdown("**Flujo RAG:**")
    st.markdown("1. PDF → Texto")
    st.markdown("2. Texto → Chunks (800 chars + 150 overlap)")
    st.markdown("3. Chunks → Vertex AI Embeddings")
    st.markdown("4. Embeddings → ChromaDB")
    st.markdown("5. Pregunta → Embedding → Top-5 chunks")
    st.markdown("6. Chunks + Pregunta → Vertex AI Gemini → Respuesta")
    st.markdown("7. Respuesta → Evaluación (LLM-as-judge) → Log")

tab1, tab2, tab3 = st.tabs(["📄 Subir documento", "💬 Hacer preguntas", "📊 Métricas"])

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
            eval_result = evaluate(pregunta, respuesta, chunks, model=llm)
            log_query(pregunta, chunks, respuesta, eval_result)

        st.subheader("Respuesta")
        st.write(respuesta)

        col1, col2, col3 = st.columns(3)
        col1.metric("Similitud promedio", f"{eval_result.avg_similarity:.3f}")
        col2.metric("Diversidad de chunks", f"{eval_result.diversity:.2f}")
        col3.metric(
            "Faithfulness (LLM-judge)",
            f"{eval_result.faithfulness}/5" if eval_result.faithfulness else "N/A",
        )

        with st.expander("Ver chunks recuperados (debug)"):
            for i, c in enumerate(chunks):
                st.markdown(f"**Chunk {i + 1}** | Distancia: `{c['distancia']:.4f}`")
                st.text(c["texto"][:300] + "...")
                st.markdown("---")

with tab3:
    st.header("Métricas de observabilidad")

    if st.button("Actualizar métricas"):
        st.rerun()

    stats = get_stats()

    if stats["total_queries"] == 0:
        st.info("Todavía no hay queries registradas. Hacé una pregunta en la pestaña anterior.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de queries", stats["total_queries"])
        col2.metric(
            "Faithfulness promedio",
            f"{stats['avg_faithfulness']}/5" if stats["avg_faithfulness"] else "N/A",
        )
        col3.metric("Similitud promedio", f"{stats['avg_similarity']:.4f}")

        st.subheader("Últimas 10 consultas")
        for entry in reversed(stats["recent"]):
            with st.expander(f"🔍 {entry['question'][:80]}"):
                st.markdown(f"**Respuesta:** {entry['answer'][:300]}...")
                st.markdown(
                    f"Chunks: `{entry['chunks_count']}` | "
                    f"Similitud: `{entry['avg_similarity']}` | "
                    f"Faithfulness: `{entry.get('faithfulness') or 'N/A'}`"
                )
                st.caption(entry["timestamp"])
