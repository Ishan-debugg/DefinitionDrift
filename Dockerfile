FROM python:3.11-slim

WORKDIR /app

# system deps for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# pre-download the embedding model (so container starts fast)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

RUN mkdir -p data embeddings

EXPOSE 8000
