"""Fonction d'embedding multilingue (ONNX via fastembed, sans torch)."""
from functools import lru_cache

from chromadb import Documents, EmbeddingFunction, Embeddings


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Une seule session ONNX par modèle et par processus.

    ChromaDB reconstruit la fonction d'embedding depuis sa configuration persistée ;
    sans ce cache, chaque reconstruction rechargerait ~200 Mo en mémoire.
    """
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


class FastEmbedFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = _load_model(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return [v.tolist() for v in self._model.embed(list(input))]

    @staticmethod
    def name() -> str:
        return "fastembed"

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    @staticmethod
    def build_from_config(config: dict) -> "FastEmbedFunction":
        return FastEmbedFunction(config["model_name"])
