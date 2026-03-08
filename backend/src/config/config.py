"""Load configuration from environment. Never access os.environ outside this module."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = "chroma_db"
    chroma_collection: str = "law_chunks"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 5
    min_chunk_words: int = 5
    min_chunk_chars: int = 50
    # Reranker (cross-encoder) – second stage after vector search
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_candidates: int = 50
    port: int = 8000
    env: str = "development"
    courtlistener_api_token: str | None = None
    courtlistener_base_url: str = "https://www.courtlistener.com/api/rest/v4"
    # RAG – Groq API (chat completions)
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Path to exported full case texts (for RAG context)
    exports_texts_dir: str = "exports/courtlistener_first_cases/texts"


def get_settings() -> Settings:
    """Return fresh Settings (no caching). Values from os.environ + .env; env vars override .env unless load_dotenv(override=True) was used earlier."""
    return Settings()
