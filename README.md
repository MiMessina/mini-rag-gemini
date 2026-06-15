# Mini RAG con Gemini

Demostración técnica de un pipeline RAG (*Retrieval-Augmented Generation*) end-to-end usando la API de Google Gemini y ChromaDB como vector store local.

## Stack

| Componente | Tecnología |
|---|---|
| LLM | Gemini 2.0 Flash |
| Embeddings | text-embedding-004 (768 dim) |
| Vector store | ChromaDB (persistente local) |
| Interfaz web | Streamlit |
| Lectura de PDF | pypdf |

## Flujo RAG

```
PDF → Texto → Chunks (800 chars, overlap 150) → Embeddings → ChromaDB
                                                                   ↑
Pregunta → Embedding → Top-5 chunks más similares ────────────────┘
                             ↓
              Gemini (contexto + pregunta) → Respuesta
```

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/MiMessina/mini-rag-gemini.git
cd mini-rag-gemini

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

## Configuración

Crear un archivo `.env` en la raíz del proyecto:

```env
GEMINI_API_KEY=tu_api_key_aqui
GEMINI_MODEL=gemini-2.0-flash        # opcional
EMBEDDING_MODEL=text-embedding-004   # opcional
```

Obtené tu API key en [Google AI Studio](https://aistudio.google.com/apikey).

## Uso

### Interfaz web (Streamlit)

```bash
streamlit run app.py
```

Abre `http://localhost:8501` en el navegador. Desde ahí podés subir un PDF y hacer preguntas sobre su contenido.

### CLI

```bash
# Indexar un PDF
python cli.py ingest mi_documento.pdf

# Hacer una pregunta
python cli.py query "¿Cuál es el tema principal del documento?"
```

## Estructura del proyecto

```
mini-rag-gemini/
├── app.py            # Interfaz Streamlit
├── cli.py            # Versión línea de comandos
├── requirements.txt
├── .env              # No incluido en el repo (contiene la API key)
└── chroma_db/        # Generado automáticamente al indexar un PDF
```
