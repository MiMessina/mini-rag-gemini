# Mini RAG con Vertex AI

Demostración técnica de un pipeline RAG (*Retrieval-Augmented Generation*) end-to-end usando **Vertex AI** de Google Cloud y ChromaDB como vector store local. Incluye módulo de evaluación de calidad (LLM-as-judge) y observabilidad con logging persistente.

> Pipeline RAG end-to-end con Vertex AI. Autenticación via Application Default Credentials (ADC).

## Stack

| Componente | Tecnología |
|---|---|
| LLM | Vertex AI — Gemini 2.5 Flash |
| Embeddings | Vertex AI — text-embedding-005 (768 dim) |
| Vector store | ChromaDB (persistente local) |
| Interfaz web | Streamlit |
| Evaluación | LLM-as-judge + métricas de retrieval |
| Observabilidad | Logging JSONL local |

## Flujo RAG

```
PDF → Texto → Chunks (800 chars, overlap 150) → Vertex AI Embeddings → ChromaDB
                                                                            ↑
Pregunta → Vertex AI Embedding → Top-5 chunks más similares ───────────────┘
                                        ↓
                       Vertex AI Gemini (contexto + pregunta) → Respuesta
                                        ↓
                          Evaluación (LLM-as-judge 1-5) → Log JSONL
```

## Diferencias clave: Gemini API vs Vertex AI

| | Gemini API (rama `main`) | Vertex AI (esta rama) |
|---|---|---|
| **Auth** | API key en `.env` | Service account JSON / ADC |
| **SDK** | `google-genai` | `google-genai` (v2.8.0+) |
| **Embeddings** | text-embedding-004 | text-embedding-005 |
| **Scope** | Personal / prototipo | Enterprise (IAM, VPC, audit logs) |
| **Facturación** | Por API key | Por proyecto GCP |
| **Ideal para** | Desarrollo rápido | Producción en GCP |

## Setup GCP (paso a paso)

### 1. Crear cuenta y proyecto GCP

1. Ir a [console.cloud.google.com](https://console.cloud.google.com) y crear una cuenta (hay $300 de crédito free)
2. Crear un nuevo proyecto y anotar el **Project ID**

### 2. Habilitar la API de Vertex AI

```bash
gcloud services enable aiplatform.googleapis.com
```

O desde la consola: **APIs & Services → Habilitar APIs → Vertex AI API**

### 3. Crear una Service Account

```bash
gcloud iam service-accounts create rag-demo \
  --display-name="RAG Demo"

gcloud projects add-iam-policy-binding TU_PROJECT_ID \
  --member="serviceAccount:rag-demo@TU_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud iam service-accounts keys create ./service-account.json \
  --iam-account=rag-demo@TU_PROJECT_ID.iam.gserviceaccount.com
```

### 4. Alternativa: Application Default Credentials (ADC)

Si tenés `gcloud` instalado localmente, podés saltear la service account:

```bash
gcloud auth application-default login
```

En este caso, no hace falta `GOOGLE_APPLICATION_CREDENTIALS` en el `.env`.

## Instalación

```bash
git clone https://github.com/MiMessina/mini-rag-gemini.git
cd mini-rag-gemini
git checkout vertex-ai

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

## Configuración

```bash
cp .env.example .env
# Editar .env con los valores del proyecto GCP
```

Contenido del `.env`:

```env
GCP_PROJECT_ID=mi-proyecto-gcp
GCP_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=text-embedding-005
```

## Uso

### Interfaz web (Streamlit)

```bash
streamlit run app.py
```

Abre `http://localhost:8501`. Tiene tres pestañas:
- **Subir documento** — indexa un PDF en ChromaDB
- **Hacer preguntas** — RAG con métricas por respuesta
- **Métricas** — dashboard de observabilidad con historial

### CLI

```bash
# Indexar un PDF
python cli.py ingest mi_documento.pdf

# Hacer una pregunta (muestra respuesta + métricas)
python cli.py query "¿Cuál es el tema principal del documento?"
```

## Estructura del proyecto

```
mini-rag-gemini/
├── app.py            # Interfaz Streamlit (Vertex AI + métricas)
├── cli.py            # CLI (Vertex AI + métricas)
├── evaluator.py      # Módulo de evaluación RAG (LLM-as-judge)
├── logger.py         # Observabilidad: logging a JSONL
├── requirements.txt
├── .env.example      # Template de variables de entorno
├── .env              # No incluido en el repo
├── service-account.json  # No incluido en el repo
├── logs/             # Generado automáticamente
│   └── queries.jsonl # Historial de queries con métricas
└── chroma_db/        # Generado automáticamente al indexar
```

## Evaluación y observabilidad

### Métricas de retrieval (sin costo adicional)
- **Similitud promedio**: `1 - distancia_coseno` de los chunks recuperados
- **Diversidad**: fracción de chunks con distancia significativa al mejor resultado

### Faithfulness — LLM-as-judge
Después de cada respuesta, se le pide al propio modelo que evalúe si la respuesta
se basó en el contexto (escala 1-5). El prompt fuerza una respuesta JSON:

```json
{"score": 4, "razon": "La respuesta usa el contexto con una pequeña inferencia razonable"}
```

### Logging
Cada query se registra en `logs/queries.jsonl`:

```json
{
  "timestamp": "2026-06-15T14:30:00Z",
  "question": "¿Qué dice el documento sobre X?",
  "answer": "Según el documento...",
  "chunks_count": 5,
  "avg_similarity": 0.8234,
  "diversity": 0.6,
  "faithfulness": 4.0,
  "chunks_distances": [0.12, 0.15, 0.18, 0.21, 0.25]
}
```

## Arquitectura de producción en GCP

Para escalar este POC a producción, ChromaDB local se reemplaza por servicios managed de GCP:

```
                    ┌─────────────────────────────────────────┐
                    │              Google Cloud                │
                    │                                         │
  Usuario           │  Cloud Run          Vertex AI           │
    │               │  ┌──────────┐   ┌───────────────────┐  │
    └──── HTTPS ───►│  │  app.py  │──►│ Embeddings API    │  │
                    │  │(Streamlit│   │ text-embedding-005 │  │
                    │  │  / CLI)  │   └───────────────────┘  │
                    │  │          │   ┌───────────────────┐  │
                    │  │          │──►│ Vector Search      │  │
                    │  │          │   │ (reemplaza ChromaDB│  │
                    │  │          │   └───────────────────┘  │
                    │  │          │   ┌───────────────────┐  │
                    │  │          │──►│ Gemini (LLM)       │  │
                    │  └──────────┘   └───────────────────┘  │
                    │       │                                  │
                    │       ▼                                  │
                    │  Cloud Logging / BigQuery                │
                    │  (métricas y observabilidad)             │
                    │                                         │
                    │  Cloud Storage                           │
                    │  (PDFs fuente)                           │
                    └─────────────────────────────────────────┘
```

### Por qué Vertex AI Vector Search en vez de ChromaDB
- Escala a miles de millones de vectores
- SLA de Google, no infraestructura propia
- Integración nativa con IAM y VPC
- Actualizaciones de índice en streaming
