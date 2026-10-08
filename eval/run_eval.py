"""Évaluation du retrieval : hit@k, MRR, et taux de rejet des questions hors-sujet.

Usage : python -m eval.run_eval            (seuil courant)
        python -m eval.run_eval --sweep    (balayage de MAX_DISTANCE)
"""
import json
import sys
from pathlib import Path

from app import config
from app.store import get_collection

DATA = json.loads((Path(__file__).parent / "dataset.json").read_text(encoding="utf-8"))


def stem(source: str) -> str:
    return Path(source).stem


def evaluate(max_distance: float) -> dict:
    col = get_collection()
    hits, rr = 0, 0.0
    for item in DATA["in_scope"]:
        res = col.query(query_texts=[item["q"]], n_results=config.TOP_K)
        ranked = [
            stem(m["source"]) for m, d in zip(res["metadatas"][0], res["distances"][0]) if d <= max_distance
        ]
        pos = next((i for i, s in enumerate(ranked, 1) if s in item["expected"]), None)
        if pos:
            hits += 1
            rr += 1 / pos
    rejected = 0
    for q in DATA["out_of_scope"]:
        res = col.query(query_texts=[q], n_results=1)
        rejected += res["distances"][0][0] > max_distance
    n_in, n_out = len(DATA["in_scope"]), len(DATA["out_of_scope"])
    return {
        "max_distance": max_distance,
        f"hit@{config.TOP_K}": round(hits / n_in, 3),
        "mrr": round(rr / n_in, 3),
        "oos_rejection": round(rejected / n_out, 3),
    }


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        for t in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            print(evaluate(t))
    else:
        print(evaluate(config.MAX_DISTANCE))
