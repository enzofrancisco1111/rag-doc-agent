FROM python:3.11-slim

WORKDIR /srv
ENV FASTEMBED_CACHE_PATH=/opt/fastembed PYTHONUNBUFFERED=1 CHROMA_DIR=/data/chroma DOCS_DIR=/srv/docs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY docs ./docs

# Pré-télécharge le modèle d'embeddings pour éviter le téléchargement au premier appel
RUN python -c "from fastembed import TextEmbedding as T; list(T('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').embed(['warmup']))"

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
