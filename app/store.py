"""Accès à la base vectorielle ChromaDB (embeddings locaux MiniLM via ONNX, sans clé API)."""
import chromadb

from . import config


def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_or_create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"}
    )
