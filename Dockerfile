FROM python:3.11-slim

WORKDIR /app

# System deps: build tools for sentence-transformers + libsql, curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full codebase
COPY . .

# Pre-download embedding model so the container starts fast (failure is non-fatal)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# Ensure runtime directories exist
RUN mkdir -p data embeddings

# Koyeb / Cloud Run expose port 8000 by default
EXPOSE 8000

# Start the FastAPI server
# - host 0.0.0.0 so Koyeb can reach it from outside the container
# - workers 2 keeps memory low on free tier
# - timeout-keep-alive 75 prevents Koyeb load-balancer from dropping idle connections
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "75"]
