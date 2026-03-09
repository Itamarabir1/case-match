# Backend service – multi-stage, non-root (per architecture guide).
FROM python:3.12-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# Install PyTorch CPU first so sentence-transformers does not pull CUDA (~250MB vs ~3GB).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS production
WORKDIR /app
RUN addgroup --system appgroup && adduser --system appuser --ingroup appgroup
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
# Hugging Face / sentence-transformers need a writable cache (appuser has no write to default paths).
RUN mkdir -p /app/.cache && chown -R appuser:appgroup /app/.cache
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache
USER appuser
ENV PORT=8000
ENV API_ONLY=1
EXPOSE ${PORT}
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
