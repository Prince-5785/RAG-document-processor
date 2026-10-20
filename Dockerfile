# Production Dockerfile for InsureQuery API
FROM python:3.11-slim

WORKDIR /app

# Install minimal build and curl dependencies for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and initialize persistent directories
RUN useradd -m app && \
    mkdir -p /app/data/documents /app/data/chroma_index /app/data/cache /home/app/.cache/huggingface && \
    chown -R app:app /app /home/app

# Install CPU-only PyTorch (drastically reduces image size)
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Copy and install production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding and reranker models to /home/app/.cache/huggingface during build
ENV HF_HOME=/home/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application source code
COPY . .
RUN chown -R app:app /app /home/app

# Run as non-root user
USER app

# Internal port exposed to Caddy reverse proxy
EXPOSE 8000

# Health check against FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single-worker uvicorn (metrics counters and vector store registry are in-process)
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
