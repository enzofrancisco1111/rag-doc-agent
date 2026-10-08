"""Accès à la base vectorielle ChromaDB avec embeddings multilingues locaux (pas de clé API)."""
from functools import lru_cache

import chromadb

from . import config
from .embeddings import FastEmbedFunction


@lru_cache(maxsize=4)
def _embedder(model_name: str) -> FastEmbedFunction:
    return FastEmbedFunction(model_name)


@lru_cache(maxsize=4)
def _client(path: str) -> chromadb.ClientAPI:
    """Un seul client par dossier : en rouvrir un à chaque requête est lent et fragile."""
    return chromadb.PersistentClient(path=path)


@lru_cache(maxsize=4)
def _collection(path: str, name: str, model_name: str):
    return _client(path).get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_embedder(model_name),
    )


def get_collection():
    return _collection(config.CHROMA_DIR, config.COLLECTION, config.EMBEDDING_MODEL)
