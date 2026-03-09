# Law Retrieval (case-match)

**Main project README** – documentation, structure, and run instructions (backend, frontend). Everything via the web UI; server: `python backend/main.py`.  
**Repository:** [github.com/Itamarabir1/case-match](https://github.com/Itamarabir1/case-match)

## 1. Overview

### Core idea

- **Goal:** A system that takes a **legal problem** (text) and returns **similar cases/opinions** and what was decided – **not** “win/lose” prediction.
- **Technique:** Semantic (vector) search over an English case corpus, with an option for hybrid search (vector + BM25) and then result display (and optionally LLM summary).

### Technical choices

| Topic | Decision |
|-------|----------|
| **Data source** | **CourtListener API** – English cases/opinions; indexing via the UI (POST /index/rebuild). |
| **What is stored locally** | **Vectors + chunk text** in Chroma; full text and vectors also in `exports/courtlistener_first_cases/`. |
| **Data loading** | **Streaming** from CourtListener API (with checkpoint and `--resume` to continue). |
| **512-token limit** | Long cases → **chunking**. Use **RecursiveCharacterTextSplitter** with separators: `["\n\n", "\n", ". ", " "]` to split at paragraph/sentence boundaries. |
| **Chunk size** | ~400–450 tokens (or ~1,200–1,600 characters), with **overlap** 50–100 tokens. |
| **Embedding model** | **sentence-transformers**, English + CPU: e.g. `all-MiniLM-L6-v2` (light) or `all-mpnet-base-v2` (higher quality). |
| **Vector store** | **Chroma** to start; consider **FAISS** if you have very many chunks. |
| **Development order** | First **simple retrieval** (chunk → embed → store → query → aggregate). Only then add **Small-to-Big** (if you need LLM with large context) and **Hybrid + RRF**. |
| **Aggregation** | When grouping by case (doc_id): use **mean similarity score** of that case’s chunks, not just “how many chunks in top”. |
| **Validation** | After retrieval: filter out empty or **too short** chunks (e.g. fewer than 5 words or 50 characters). |
| **Hybrid Search** | **BM25** (keywords: statute number, judge name) + **vector search**; merge with **RRF** (Reciprocal Rank Fusion). |
| **Parallel** | For index build – **multiprocessing** (or workers) to use CPU; watch RAM and write exclusivity to DB. |
| **Small-to-Big** | Optional: small chunks (e.g. 200 words) for index, **parent** chunk (e.g. 1000 words) for LLM – retrieval precision + full context. |

---

## 2. Architecture and design principles

The project follows an **Architecture Guide** (Clean Architecture, fullstack layout): root files (.env.example, .gitignore, .dockerignore, pyproject.toml, docker-compose with healthcheck), backend with src/api, services, repositories, config; multi-stage non-root Dockerfile; config only via pydantic-settings.

- **Pydantic:** Models for Document, Chunk, Query, SearchResult, etc. – input/output validation.
- **Clear pipeline:** Index and query steps defined in separate functions/modules.
- **System prompt:** If using an LLM – fixed prompt (role, format, limits) in a separate file/variable.
- **JSON Schema:** Description of the API (request/response) – for docs and validation.
- **Config:** File (YAML/JSON/env) for model, chunk size, RRF, DB paths – no magic numbers in code.
- **Logging:** Structured logs (e.g. JSON) with request_id, stage, timings.
- **Zero trust:** Validate and sanitize input; check returned chunks before display.

---

## 3. Project layout (top-level view)

### 3.1 Project structure

```
Law/
├── README.md              ← Project documentation (single file for the whole project)
├── STRUCTURE.md           ← Structure and folder summary
├── .env.example           ← Shared env vars only; copy to .env
├── .gitignore             ← One per project
├── .dockerignore
├── .python-version        ← 3.12
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
├── render.yaml
│
├── backend/                ← API service
│   ├── src/                ← api/, config/, domain/, infrastructure/, prompts/, repositories/, schemas/, services/, utils/, app.py
│   ├── main.py             ← Run server: python backend/main.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example        ← Backend-specific env; copy to backend/.env
│   └── .dockerignore       ← One for backend
│
├── frontend/
│   ├── index.html          ← Single UI: text input, Find Similar Cases, RAG analysis, results
│   ├── .env.example        ← Frontend-specific (for future use)
│   └── .dockerignore       ← One for frontend
│
├── examples/               ← new_case.txt, sample_case.txt
├── exports/                ← Texts and checkpoint (not in Git)
├── chroma_db/              ← Chroma DB (not in Git)
└── tests/                  ← unit/, integration/
```

### 3.2 Index build flow (CourtListener)

1. Load cases from **CourtListener API** (streaming, with checkpoint to resume).
2. Per case: extract text → **chunking** (RecursiveCharacterTextSplitter, separators, size/overlap from config).
3. **Embed** each chunk (batch).
4. Store in **Chroma**: id, embedding, metadata (doc_id, chunk_index, text, title, citation, court, date_filed, etc.).
5. Export full text and vectors to `exports/courtlistener_first_cases/`.

Index is stored in `chroma_db/`; full texts in `exports/`.

### 3.3 Search flow (per query)

1. **Input:** Legal problem text (Pydantic validation – non-empty, max length).
2. **Embed** the query.
3. **Retrieve:** Vector search: fetch `reranker_candidates` chunks from Chroma (default 50).
4. **Reranker:** Cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranks chunks by relevance to the query; scores normalized to [0,1]. Top candidates kept after reranking.
5. **Validation:** Drop too-short or empty chunks.
6. **Aggregation:** Group by `doc_id`; case score = **mean** of that case’s chunk scores.
7. **Output:** Sorted list of cases (up to `top_k`, default 5) – doc_id, score, snippets, metadata.
8. (Optional) Send to **LLM** with system prompt – for display/summary only, not outcome prediction.

### 3.4 Main code components

- **schemas (Pydantic):** `Document`, `Chunk`, `SearchQuery`, `SearchResult`, `RankedCase`.
- **chunking:** Function that takes text and returns a list of `Chunk` (doc_id, index, text).
- **embedding:** Load sentence-transformers once; function `embed(texts)`.
- **store:** Init Chroma, `add_chunks(chunks)`, `search(query_embedding, k)`.
- **retrieval:** `search(query_text)` – embed → search (Chroma) → **rerank** (cross-encoder) → validate → aggregate → `list[RankedCase]`. Config: `reranker_enabled`, `reranker_candidates`, `top_k`.
- **POST /index/rebuild:** Build index (background) – stream from CourtListener → chunking → embed → store + export.
- **GET /cases/{doc_id}/text:** Return full case text as **PDF** (from exports; opens inline in browser).
- **POST /analyze:** RAG – retrieve + analyze with Groq; **streams** response as SSE (cases first, then AI tokens in real time). Final result: structured `analysis_json` (Legal Pattern, Common Outcome, Key Considerations). See **Streaming (POST /analyze)** below.
- **api + services:** Routes (search, index, analyze, cases) → services → display/API. Prompts in `src/prompts/rag.py`.

---

## 4. Short summary

- **What we build:** Retrieval over English legal cases – “enter a legal problem, get similar cases and what was decided”.
- **Where data lives:** CourtListener API; locally Chroma (vectors + metadata) + export under `exports/`.
- **How:** Smart chunking (RecursiveCharacterTextSplitter), embedding (sentence-transformers, CPU), Chroma, **reranker** (cross-encoder) after vector search, aggregation by mean score, validation on chunks; optional: Hybrid + RRF, Small-to-Big, LLM with system prompt.
- **Project layout:** config, src (api, services, repositories, schemas, infrastructure, utils), backend/main.py (server only), chroma_db – search and index build via API and web UI.

---

## 5. Running the project

**Install uv (if needed):**  
[https://docs.astral.sh/uv/](https://docs.astral.sh/uv/) or `pip install uv`

```bash
# From the Law project root
cd Law

# Create venv and install dependencies (from project root)
uv venv
uv sync

# (Windows) Activate venv
.venv\Scripts\activate

# (Linux/macOS) Activate venv
# source .venv/bin/activate

# Copy .env: shared at root, backend-specific in backend/
cp .env.example .env                    # Root – shared vars
cp backend/.env.example backend/.env    # Backend – API keys etc.
# Edit backend/.env and add COURTLISTENER_API_TOKEN, GROQ_API_KEY; optionally LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY for observability.

# Index build: after starting the server – via UI or API (POST /index/rebuild, GET /index/stats, POST /index/reset).

# Start server (from root)
uv run python backend/main.py
# Or: cd backend && uv run uvicorn src.app:app --reload --port 8000
# Search: web UI (Find Similar Cases) or POST http://localhost:8000/search

# Or with Docker
docker compose up --build
# API: http://localhost:8000 – open frontend/index.html in browser
```

**Alternative with pip:**  
From root: `pip install -r requirements.txt`. Run server: `python backend/main.py`. Search and index build – via web UI.

---

### Troubleshooting (Windows) – Hugging Face cache error

If you see:
```text
ERROR: Invalid disk cache: your machine does not support long paths.
```
This happens on Windows when the Hugging Face cache path is too long. **Fix:** Set the `HF_HOME` environment variable to a **short** path, e.g.:
- `C:\hf_cache`
- or `c:\Users\user\Desktop\Law\.cache\hf`

In PowerShell before running:
```powershell
$env:HF_HOME = "C:\hf_cache"
uv run python backend/main.py
```
Then build the index via the UI (POST /index/rebuild).  
Or set `HF_HOME` in System → Environment variables, or in .env if your tool passes it to the process.

---

### CourtListener – indexing and resume (checkpoint)

Add the token from [Developer Tools](https://www.courtlistener.com/help/api/rest/) (Your API Token) to `.env`:

```bash
COURTLISTENER_API_TOKEN=your_token_here
COURTLISTENER_BASE_URL=https://www.courtlistener.com/api/rest/v4
```

- **Chunk overlap:** Set in config (`CHUNK_OVERLAP`, default 150 characters) – shared words between adjacent chunks in a case.
- **Stored metadata:** Required: `id`, full text, `title` (case name – from cluster if needed). Recommended: `citation`, `court`, `date_filed`, `docket_number`, `disposition` (for Decision Support). Full case text also in `exports/.../texts`.
- **Unlimited run:** Default – downloads all cases. You can stop (Ctrl+C) anytime; progress is saved in the checkpoint.
- **Resume:** Index build (POST /index/rebuild) resumes automatically from checkpoint if it exists.

Index build via UI or API: **POST /index/rebuild** (optional: `?max_docs=5` for a test run). Texts go to `exports/courtlistener_first_cases/texts`, vectors to `exports/courtlistener_first_cases/vectors`. Checkpoint file: `exports/courtlistener_first_cases/courtlistener_checkpoint.json`. The index currently has about **50,000** cases (per checkpoint).

**Index build and rollback from CLI (from project root):**
```bash
# Download/resume index (with automatic checkpoint)
python backend/main.py build-index
# Limit number of cases (e.g. for testing)
python backend/main.py build-index --max-docs 100
# Rollback: revert to a target doc count (removes “last” docs from Chroma, texts, vectors and updates checkpoint)
python backend/main.py rollback --to-docs 50000
```

### Top 5 similar cases

Via the UI: “Find Similar Cases” (or POST /search) – returns 5 similar cases (default `TOP_K=5`).

### RAG – analyze new case vs similar cases (LLM)

Build **context** from the top 5 similar cases (full text from `exports/.../texts` or snippets), add the **new case** text from the user, and send to **Groq API** (cloud LLM) for structured analysis: Legal Pattern, Common Outcome, Key Considerations.

**Example input (the “prompt”):** The text you type in the UI (or paste from a file) is the **new case** description that gets sent to the LLM. You can use the sample in `examples/new_case.txt` as a template: paste its contents into the text box, run “Find Similar Cases”, then “Analyze these results with AI”. The same string is used both as the search query and as the `{new_case}` placeholder in the RAG user prompt (see `backend/src/prompts/rag.py`).

**SSE (POST /analyze):** The `/analyze` endpoint returns **Server-Sent Events (SSE)**. Flow: (1) Backend runs retrieval and sends `type: "cases"` with the list of similar cases. (2) It calls Groq once (non-streaming, because Groq Structured Outputs do not support streaming) and sends the full analysis in one event `type: "token"` with `content`. (3) A final event `type: "done"` includes `duration_ms`, `model`, and optional `usage` (token counts). The frontend parses the JSON and renders Legal Pattern, Common Outcome, Key Considerations, and a small trace line (model · tokens · time).

**Output:** Accumulated stream is parsed as `analysis_json` (Pydantic/JSON Schema: `legal_pattern`, `common_outcome`, `key_considerations`, optional `summary`, `caveats`). Prompts (system + user template) are in `backend/src/prompts/rag.py`.

### Langfuse – LLM observability

[Langfuse](https://langfuse.com) records every **Groq** call used for RAG analysis: prompt, response, latency, and token usage. This helps with debugging and cost/latency analysis.

- **When it runs:** Only for the **Analyze** flow (POST /analyze). Search is not traced so it is never blocked by observability.
- **Optional:** If `LANGFUSE_PUBLIC_KEY` is empty or unset, tracing is skipped; the app works normally without Langfuse.
- **Non-blocking:** Flush runs in a background thread, so the HTTP response is never delayed by Langfuse.

**Setup (optional):** In `backend/.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Get keys from [Langfuse Cloud](https://cloud.langfuse.com). Do not commit real keys; use `.env` only. In the UI, after an analysis you'll see a small line with model name, token count, and duration (e.g. `llama-3.1-8b-instant · 4809 tokens · 1529ms`). The same data appears in the Langfuse dashboard with full prompt and response.

**Usage:** Via API – `POST /analyze` (frontend: “Analyze these results with AI” after search). Settings in `.env`: `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL`, `EXPORTS_TEXTS_DIR`.

**Single case view:** `GET /cases/{doc_id}/text` – returns the case text as **PDF** (application/pdf) for reading or download.
