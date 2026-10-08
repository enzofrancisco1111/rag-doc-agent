"""Accès à la base vectorielle ChromaDB avec embeddings multilingues locaux (pas de clé API)."""
from functools import lru_cache

import chromadb

from . import config
from .embeddings import FastEmbedFunction


@lru_cache(maxsize=4)
def _embedder(model_name: str) -> FastEmbedFunction:
    return FastEmbedFunction(model_name)


def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return client.get_or_create_collection(
        config.COLLECTION,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_embedder(config.EMBEDDING_MODEL),
    )
