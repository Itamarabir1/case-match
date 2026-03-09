"""Configuration: env-loaded Settings and fixed constants. Never access os.environ elsewhere."""
from pydantic_settings import BaseSettings, SettingsConfigDict

CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]
MAX_QUERY_LENGTH_CHARS = 10_000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_path: str = "chroma_db"
    chroma_collection: str = "law_chunks"
    chunk_size: int = 1200   # characters per chunk (RecursiveCharacterTextSplitter)
    chunk_overlap: int = 150  # character overlap between consecutive chunks
    top_k: int = 5
    min_chunk_words: int = 5
    min_chunk_chars: int = 50
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_candidates: int = 50
    port: int = 8000
    env: str = "development"
    courtlistener_api_token: str | None = None
    courtlistener_base_url: str = "https://www.courtlistener.com/api/rest/v4"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    exports_texts_dir: str = "exports/courtlistener_first_cases/texts"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True


def get_settings() -> Settings:
    """Fresh Settings from os.environ + .env."""
    return Settings()
