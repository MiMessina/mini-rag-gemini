"""
Mini RAG con Vertex AI + ChromaDB + Streamlit.
Versión con interfaz gráfica y métricas de observabilidad.

Uso:
    streamlit run app.py
"""

import streamlit as st

from evaluator import evaluate
from logger import log_query, get_stats
from rag import (
    GCP_LOCATION,
    GCP_PROJECT_ID,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    buscar_y_responder,
    client,
    ingestar_documento,
)

if not GCP_PROJECT_ID:
    st.error("Falta GCP_PROJECT_ID en el archivo .env — copiá .env.example a .env y completá los valores.")
    st.stop()


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
    st.markdown("**Vector Store:** ChromaDB (local)")
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

            def on_progress(i, total):
                progress_bar.progress(i / total, text=f"Procesado {i}/{total}")

            with st.spinner("Procesando..."):
                total = ingestar_documento(archivo, archivo.name, progress_callback=on_progress)
            st.success(f"✅ Listo. {total} chunks indexados.")

with tab2:
    st.header("Hacé tu pregunta")
    pregunta = st.text_input("¿Qué querés saber del documento?")
    if pregunta and st.button("Buscar respuesta", type="primary"):
        with st.spinner("Buscando y generando respuesta..."):
            respuesta, chunks = buscar_y_responder(pregunta)
            eval_result = evaluate(pregunta, respuesta, chunks, client=client, model=GEMINI_MODEL)
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
