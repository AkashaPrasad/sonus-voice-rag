# Index is baked into the image so boot is a load, not a build.
FROM python:3.12-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir uv
COPY pyproject.toml .
RUN uv pip install --system --no-cache \
      fastapi "uvicorn[standard]" uvloop orjson pydantic "httpx[http2]" \
      model2vec numpy bm25s datasets huggingface-hub python-multipart

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.hf INDEX_PATH=/app/index PORT=8000
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY ingest/ /app/ingest/
COPY backend/ /app/backend/
# Pre-download the static embedder so the container never fetches at boot.
RUN python -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-multilingual-128M')"
# Build the index at image build time.
# Bilingual index: both sides of each parallel passage pair are indexed, so
# English queries match English passages instead of retrieving Indic noise.
ARG CORPUS_LANGS=hi,ta
ARG CORPUS_ROWS=2500
RUN python ingest/build_bilingual.py --langs $CORPUS_LANGS \
      --rows $CORPUS_ROWS --strategy s5_hierarchical --out /app/index \
 && rm -rf /root/.cache/huggingface/datasets /root/.cache/huggingface/hub/datasets*
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",8000)}/health')"
CMD ["sh","-c","uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --loop uvloop --no-access-log"]
