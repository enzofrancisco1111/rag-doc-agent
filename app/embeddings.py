"""Fonction d'embedding multilingue (ONNX via fastembed, sans torch)."""
from chromadb import Documents, EmbeddingFunction, Embeddings


class FastEmbedFunction(EmbeddingFunction):
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

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
