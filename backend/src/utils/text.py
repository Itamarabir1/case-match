"""Text helpers."""
import re


def safe_filename(value: str) -> str:
    """Safe file/dir name: alphanumeric, dots, underscores, hyphens only."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "doc"
