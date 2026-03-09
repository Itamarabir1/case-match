# Project structure

## Rules: .gitignore, .dockerignore, .env / .env.example

- **.gitignore** – **one per project** (at root). Defines what is not committed across all folders.
- **.dockerignore** – **one per backend and one per frontend**: `backend/.dockerignore`, `frontend/.dockerignore`. Each service defines what is not sent to its Docker build.
- **.env.example** and **.env** (the real file):
  - **Shared** (backend + frontend or compose) → **at root**: `.env.example` / `.env`
  - **Backend-specific** → **backend/.env.example** / **backend/.env**
  - **Frontend-specific** → **frontend/.env.example** / **frontend/.env**  
  docker-compose loads: `env_file: [.env, backend/.env]` (root + backend).

---

## Root files

| File | Purpose |
|------|---------|
| **README.md** | Documentation for the whole project |
| **.gitignore** | One per project – what not to commit |
| **.dockerignore** | (For use when building from root, if present) |
| **.env.example** | **Shared** env vars only (copy to .env) |
| **requirements.txt**, **pyproject.toml**, **docker-compose.yml** | |

---

## Folders – each service with its own .dockerignore and .env.example

| Folder | Contents |
|--------|----------|
| **backend/** | **Dockerfile**, **.dockerignore**, **.env.example** (backend-specific env). Code: `src/`. |
| **frontend/** | **.dockerignore**, **.env.example** (frontend-specific). `index.html`. No Dockerfile currently (static). |
| **examples/** | Example files. |
| **exports/** | Exported texts (not in Git). |
| **chroma_db/** | Chroma DB (not in Git). |
| **tests/** | Tests. |

---

## Running

- **Local:** From root run `uv sync` (or `pip install -r requirements.txt`), then `python backend/main.py` (or `cd backend && uvicorn src.app:app --reload --port 8000`).
- **Docker:** From root run `docker compose up --build` → API at http://localhost:8000. Open `frontend/index.html` in the browser.
