# 📻 AudiX Engine: Production-Grade Multimodal Ingestion & Semantic Discovery Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gemini AI](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-FF6F00?style=for-the-badge&logo=facebook&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An enterprise-ready, high-throughput asynchronous transcreation, multimodal asset rendering, and semantic content discovery engine designed for multi-lingual audio streaming networks.

---

## 🎯 Executive Summary & Business Context

Expanding audio streaming platforms into regional markets (e.g., localizing English scripts into Hindi, Tamil, or Telugu) traditionally requires weeks of manual work by translation teams, voice actors, and media editors. 

**AudiX Engine** automates this end-to-end workflow:
1. **Cultural Transcreation**: Adapts raw episode scripts natively (jokes, idioms, cultural references) using LLM structural schemas.
2. **Asynchronous Audio Rendering**: Synthesizes regional voice tracks offloaded to background thread pools with deterministic caching.
3. **Semantic Discovery**: Maps show synopses into a continuous 384-dimensional vector space for sub-millisecond conceptual search queries.

---

## 🏛️ System Architecture


```

```
                                +-----------------------+
                                | Streamlit Front-End   |
                                |  (app/dashboard.py)   |
                                +-----------+-----------+
                                            |
                                      HTTP POST / GET
                                  X-API-KEY Authenticated
                                            v
                                +-----------------------+
                                |  FastAPI Gateway      |
                                |     (api/main.py)     |
                                +-----------+-----------+
                                            |
    +---------------------------------------+---------------------------------------+
    |                                       |                                       |
    v                                       v                                       v

```

+---------------+-------+               +-------+---------------+               +-------+---------------+
|  Translation Core     |               | Voice Synthesis Engine|               | Vector Search Engine  |
| (translation_engine)  |               |  (voice_synthesis)    |               |   (search_indexing)   |
+---------------+-------+               +-------+---------------+               +-------+---------------+
|                                       |                                       |
Gemini 2.5 API                         ThreadPool Workers                      SentenceTransformer
(Pydantic Validated)                     gTTS + Atomic Swap                      all-MiniLM-L6-v2 +
|                                       |                                FAISS IndexFlatIP ($L_2$)
v                                       v                                       v
Structured Script                    SHA-256 .mp3 Asset                      Dense Spatial Vector

```

---

## 📂 Project Directory Structure

```text
audix-engine/
├── .env                       # Secure API Keys & Configurations
├── config.py                  # Fail-Fast Path Resolution & App Constants
├── requirements.txt           # Dependencies
├── data/                      # Local Storage Volumes
│   ├── audio_cache/           # SHA-256 Hash Named .mp3 Files
│   ├── pipeline_history.db    # SQLite Database Logs
│   ├── faiss_index.bin        # Serialized Vector Index Snapshot
│   └── vector_metadata.pkl    # Metadata Store
├── src/                       # Core Pipeline Modules
│   ├── translation_engine.py  # GenAI Transcreation Engine
│   ├── voice_synthesis.py     # Thread-Pooled TTS & Storage
│   └── search_indexing.py     # FAISS Vector Search & Normalization
├── api/
│   └── main.py                # FastAPI Gateway
└── app/
    └── dashboard.py           # Streamlit Web Console

```

---

## ⚡ Key Technical Features & Innovations

### 1. Asynchronous Event Loop & Concurrency Isolation

* **Problem**: Network latency from Gemini API calls and slow disk audio writes block the single-threaded ASGI event loop.
* **Solution**: Offloaded disk writes to a dedicated `ThreadPoolExecutor(max_workers=4)` background pool via `loop.run_in_executor`. Structural extraction runs inside `asyncio.to_thread` to maintain high concurrency.

### 2. Deterministic Caching & Atomic OS Swaps

* **Deterministic Hashing**: Ingested text is hashed using `hashlib.sha256()` into a unique 64-character signature key. Duplicate requests trigger sub-millisecond cache hits.
* **Atomic OS Swaps**: Audio binary streams write to an isolated temporary file (`tempfile.NamedTemporaryFile`) first. Once complete, `shutil.move()` performs an atomic swap to prevent serving corrupted assets.

### 3. Native LLM Schema Binding (Pydantic)

* **Problem**: Non-deterministic LLMs often return unstructured markdown or malformed JSON text.
* **Solution**: Bound a Pydantic `LocalizedScript` schema directly into `gemini-2.5-flash` via `response_schema`. Token generation is constrained at the model output weight phase, guaranteeing type-safe outputs.

### 4. Geometric Vector Discovery ($L_2$ Normalized FAISS)

* **Math Optimization**: Show summaries are projected into 384 dense coordinates using `all-MiniLM-L6-v2`.
* **Normalization**: Every vector is divided by its Euclidean length ($L_2$ norm):

$$\text{embedding} = \frac{\text{embedding}}{\|\text{embedding}\|_2}$$



This normalizes vector magnitude to 1.0, mathematically converting FAISS Inner Product (`IndexFlatIP`) dot-product operations into exact Cosine Similarity calculations.

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites

* Python 3.10 or higher installed.
* Google Gemini API Key.

### 2. Environment Initialization

```bash
# Clone repository
git clone [https://github.com/your-username/audix-engine.git](https://github.com/your-username/audix-engine.git)
cd audix-engine

# Create & activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Configure Credentials

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
API_BEARER_TOKEN=testtoken123
API_BACKEND_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)

```

---

## 🛠️ Running the Engine

Always run commands strictly from the **project root directory** to preserve Python package resolution hooks.

### Terminal 1: Launch FastAPI Gateway

```bash
uvicorn api.main:app --reload --port 8000

```

* Swagger API Docs will be live at: `http://127.0.0.1:8000/docs`

### Terminal 2: Launch Streamlit Dashboard

```bash
streamlit run app/dashboard.py

```

* Dashboard Control Console will open automatically at: `http://localhost:8501`

---

## 📡 API Endpoint Reference

All requests require the `X-API-KEY` security authentication header.

### 1. Transcreate & Render Asset (`POST /api/v1/localize`)

* **Headers**: `X-API-KEY: testtoken123`
* **Request Body**:

```json
{
  "title": "Corporate Betrayal",
  "raw_script": "Rahul discovered secret offshore accounts. Maya warned him to stay quiet.",
  "target_language": "Hindi"
}

```

* **Sample Response (HTTP 200)**:

```json
{
  "status": "Success",
  "script_id": 1,
  "localized_content": {
    "original_title": "Corporate Betrayal",
    "target_language": "Hindi",
    "story_hook_summary": "जब राहुल को बेनामी खातों का पता चला, तो माया ने चेतावनी दी...",
    "dialogue_flow": [
      {
        "character_name": "Rahul",
        "translated_dialogue": "ये सब फर्जी खाते हैं!",
        "emotional_tone": "Shocked"
      }
    ]
  },
  "audio_asset_url": "/static/audio/Hindi_5e8842a1b9fd...mp3"
}

```

### 2. Semantic Search Discovery (`POST /api/v1/search`)

* **Request Body**:

```json
{
  "query_string": "A dark story about business fraud"
}

```

---

## 🧪 Production Defensive Design & Error Handling

* **Fail-Fast Configuration**: `config.py` enforces absolute path checks (`Path(__file__).resolve().parent`) and aborts startup immediately if critical environmental variables or keys are missing.
* **Exponential Backoff**: Translation calls are decorated with `tenacity.retry` (4 attempts, exponential backoff) to handle transient cloud network issues gracefully.
* **Thread Safety**: Vector operations use a `threading.Lock()` to prevent race conditions during concurrent vector insertions and metadata updates.

---

## 👤 Author & Acknowledgments

* **Developer**: Data & Machine Learning Engineer
* **Target Industry Focus**: Consumer-Tech, Audio-Streaming & GenAI Engineering

```

```
