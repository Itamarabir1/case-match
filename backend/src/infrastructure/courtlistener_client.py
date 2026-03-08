"""CourtListener REST API client – fetch opinions for indexing. No business logic."""
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator

from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _fetch_cluster_data(cluster_url: str, token: str, timeout: int = 30) -> tuple[str, str]:
    """GET cluster; return (case_name_short or case_name_full, disposition). Empty strings on failure."""
    if not cluster_url or not str(cluster_url).startswith("http"):
        return ("", "")
    req = urllib.request.Request(
        cluster_url,
        headers={"Authorization": f"Token {token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        title = str(
            data.get("case_name_short")
            or data.get("case_name_full")
            or data.get("case_name")
            or ""
        ).strip()
        disposition = str(data.get("disposition") or "").strip()
        return (title, disposition)
    except Exception as e:
        logger.debug("Failed to fetch cluster %s: %s", cluster_url, e)
        return ("", "")


def _strip_html(html: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_opinion_text(opinion: dict) -> str:
    """Extract plain text from opinion; fallback to stripped html_with_citations or html."""
    raw = opinion.get("plain_text") or opinion.get("html_with_citations") or opinion.get("html")
    if not raw:
        return ""
    if isinstance(raw, str) and raw.strip().startswith("<"):
        return _strip_html(raw)
    return (raw or "").strip()


def stream_courtlistener_opinions(
    max_rows: int | None = None,
    resume_from_url: str | None = None,
    on_page_done: Callable[[str | None], None] | None = None,
    fetch_title_from_cluster: bool = True,
) -> Iterator[dict]:
    """
    Yield opinion documents from CourtListener API as dicts with keys:
    id, document (full text), title, citation, docket_number, court, date_filed.
    Requires COURTLISTENER_API_TOKEN in config. Raises if token missing.
    - resume_from_url: if set, first request uses this URL (for checkpoint resume).
    - on_page_done(next_url): called after each page with the next page URL (or None when done).
    - fetch_title_from_cluster: if True, fetch case name from cluster when cluster is a URL.
    """
    settings = get_settings()
    token = settings.courtlistener_api_token
    if not token or not token.strip():
        raise ValueError(
            "CourtListener API token is required. Set COURTLISTENER_API_TOKEN in .env (see .env.example)."
        )
    base = settings.courtlistener_base_url.rstrip("/")
    next_url: str | None = resume_from_url or (
        f"{base}/opinions/?page_size=20&fields=id,plain_text,html_with_citations,html,cluster,type,citation,docket_number,court,date_filed"
    )
    collected = 0

    while next_url:
        req = urllib.request.Request(
            next_url,
            headers={
                "Authorization": f"Token {token.strip()}",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(
                f"CourtListener API error: {e.code} {e.reason}. {body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"CourtListener request failed: {e.reason}") from e

        results = data.get("results") or []
        if not results:
            if on_page_done:
                on_page_done(None)
            break

        for op in results:
            text = _get_opinion_text(op)
            if not text or not text.strip():
                continue
            doc_id = str(op.get("id", ""))
            if not doc_id:
                continue
            raw_title = op.get("case_name")
            cluster = op.get("cluster")
            cluster_url: str | None = None
            if isinstance(cluster, dict):
                cluster_url = cluster.get("resource_uri") or cluster.get("url") or None
            elif isinstance(cluster, str) and cluster.startswith("http"):
                cluster_url = cluster
            disposition = ""
            if fetch_title_from_cluster and cluster_url:
                fetched_title, disposition = _fetch_cluster_data(cluster_url, token.strip())
                if not raw_title or (isinstance(raw_title, str) and raw_title.startswith("http")):
                    raw_title = fetched_title
            if not raw_title and cluster:
                raw_title = str(cluster) if not (isinstance(cluster, str) and cluster.startswith("http")) else doc_id
            title = str(raw_title).strip() if raw_title else doc_id

            yield {
                "id": doc_id,
                "document": text.strip(),
                "title": title,
                "citation": str(op.get("citation") or ""),
                "docket_number": str(op.get("docket_number") or ""),
                "court": str(op.get("court") or ""),
                "date_filed": str(op.get("date_filed") or ""),
                "disposition": disposition,
            }
            collected += 1
            if max_rows is not None and collected >= max_rows:
                if on_page_done:
                    on_page_done(data.get("next"))
                return

        next_url = data.get("next")
        if on_page_done:
            on_page_done(next_url)
        if max_rows is not None and collected >= max_rows:
            break
