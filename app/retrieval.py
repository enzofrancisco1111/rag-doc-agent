"""Recherche vectorielle partagée par le workflow et l'agent (même seuil de pertinence des deux côtés)."""
from . import config
from .store import get_collection


def search(query: str, top_k: int | None = None, max_distance: float | None = None) -> list[dict]:
    """Passages les plus proches de `query`, en écartant ceux au-delà du seuil de distance."""
    top_k = top_k or config.TOP_K
    max_distance = config.MAX_DISTANCE if max_distance is None else max_distance
    res = get_collection().query(query_texts=[query], n_results=top_k)
    passages = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= max_distance:
            passages.append(
                {"text": doc, "source": meta["source"], "section": meta["section"], "url": meta.get("url", ""), "distance": round(dist, 3)}
            )
    return passages
