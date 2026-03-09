"""RAG orchestration: retrieval + Groq + parsing + Langfuse."""
from src.config import get_settings
from src.infrastructure.langfuse_client import get_langfuse_client
from src.schemas.analyze import RAGAnalysisStructured
from src.schemas.search_result import RankedCase

from .context import get_cases_and_prompts
from .groq_client import GROQ_RAG_SCHEMA, GroqUnavailableError, call_groq
from .parser import format_analysis_text, parse_rag_response


def run_rag(
    query: str, top_k: int | None = None
) -> tuple[list[RankedCase], str, RAGAnalysisStructured | None, str]:
    """
    Run retrieval + Groq analysis.
    Returns (cases, analysis_text, analysis_json, model_name).
    Raises GroqUnavailableError if GROQ_API_KEY is missing.
    Raises RuntimeError on Groq API or retrieval errors.
    """
    settings = get_settings()
    cases, system_prompt, user_prompt = get_cases_and_prompts(query, top_k)
    if not cases:
        return [], "", None, settings.groq_model
    api_key = (settings.groq_api_key or "").strip()
    if not api_key:
        raise GroqUnavailableError(
            "Groq API key is missing. Set GROQ_API_KEY in .env to use Full RAG Analysis."
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    langfuse = get_langfuse_client()
    raw = ""
    usage = {}
    try:
        if langfuse:
            with langfuse.start_as_current_observation(
                name="analyze",
                as_type="span",
                input={"query": query, "cases_count": len(cases)},
            ):
                with langfuse.start_as_current_observation(
                    name="groq-completion",
                    as_type="generation",
                    model=settings.groq_model,
                    input=messages,
                ) as gen:
                    try:
                        raw, usage = call_groq(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            api_key=api_key,
                            base_url=settings.groq_base_url,
                            model=settings.groq_model,
                            response_format=GROQ_RAG_SCHEMA,
                        )
                    except RuntimeError as e:
                        if "400" in str(e):
                            raw, usage = call_groq(
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                api_key=api_key,
                                base_url=settings.groq_base_url,
                                model=settings.groq_model,
                            )
                        else:
                            raise
                    usage_details = None
                    if isinstance(usage.get("prompt_tokens"), int) or isinstance(
                        usage.get("completion_tokens"), int
                    ):
                        usage_details = {
                            "prompt_tokens": usage.get("prompt_tokens") or 0,
                            "completion_tokens": usage.get("completion_tokens") or 0,
                        }
                    gen.update(output=raw, usage_details=usage_details)
            langfuse.flush()
        else:
            try:
                raw, usage = call_groq(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    api_key=api_key,
                    base_url=settings.groq_base_url,
                    model=settings.groq_model,
                    response_format=GROQ_RAG_SCHEMA,
                )
            except RuntimeError as e:
                if "400" in str(e):
                    raw, usage = call_groq(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        api_key=api_key,
                        base_url=settings.groq_base_url,
                        model=settings.groq_model,
                    )
                else:
                    raise
    except Exception:
        if langfuse:
            try:
                langfuse.flush()
            except Exception:
                pass
        raise

    structured = parse_rag_response(raw)
    has_parsed_sections = (
        (structured.legal_pattern or "").strip()
        and (structured.common_outcome or "").strip()
        and (structured.key_considerations or [])
    )
    analysis_text = format_analysis_text(structured) if has_parsed_sections else raw
    return cases, analysis_text, structured, settings.groq_model
