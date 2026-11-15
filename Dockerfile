FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml ./
COPY src/ ./src/

RUN python -m pip install \
    --prefix=/install \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.8.0+cpu" \
    ".[retrieval,llm]"

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CORFU_ENVIRONMENT=production \
    HF_HOME=/model-cache \
    CORFU_DATASET_PATH=/app/corfu_dataset.json \
    CORFU_WEATHER_FIXTURE_PATH=/app/corfu_synthetic_rain.json \
    CORFU_EMBEDDING_CACHE_DIR=/model-cache/sentence-transformers

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 corfu \
    && useradd --system --uid 10001 --gid corfu --create-home corfu \
    && install -d --owner=corfu --group=corfu /app /model-cache

COPY --from=builder /install/ /usr/local/
COPY --chown=corfu:corfu corfu_dataset.json corfu_synthetic_rain.json /app/

WORKDIR /app
USER corfu

EXPOSE 8000

CMD ["uvicorn", "corfu_assistant.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
