"""Évaluation de bout en bout de l'agent : appelle réellement le modèle (coûte des tokens, voir --n).

Compare à la base du workflow (python -m eval.run_eval) sur les mêmes questions :
  retrieved_hit : une des pages attendues figure parmi les passages que l'agent a récupérés
  cited_hit     : une des pages attendues figure parmi les sources citées dans la réponse (plus strict)
  oos_refusal   : une question hors-sujet ne produit aucune source citée

Usage : python -m eval.run_agent_eval --n 6            (6 questions dans le sujet + 6 hors-sujet, en anglais)
        python -m eval.run_agent_eval --n 6 --lang fr
"""
import argparse
import statistics
import time
import uuid

from langgraph.checkpoint.memory import MemorySaver

from app import config
from app.agent import build_agent_graph
from eval.run_eval import DATA, stem


def ask(graph, question: str) -> tuple[dict, float]:
    start = time.perf_counter()
    out = graph.invoke(
        {"messages": [{"role": "user", "content": question}]}, {"configurable": {"thread_id": uuid.uuid4().hex}}
    )
    return out, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--lang", choices=("en", "fr"), default="en")
    args = parser.parse_args()

    graph = build_agent_graph(MemorySaver())
    qk = "q_fr" if args.lang == "fr" else "q"
    oos = DATA["out_of_scope_fr"] if args.lang == "fr" else DATA["out_of_scope"]
    runs, retrieved_hits, cited_hits, refusals = [], 0, 0, 0

    items = DATA["in_scope"][: args.n]
    for item in items:
        out, seconds = ask(graph, item[qk])
        runs.append((out, seconds))
        retrieved_hits += any(stem(p["source"]) in item["expected"] for p in out.get("passages", []))
        cited_hits += any(stem(s["source"]) in item["expected"] for s in out["sources"])
    for question in oos[: args.n]:
        out, seconds = ask(graph, question)
        runs.append((out, seconds))
        refusals += not out["sources"]

    n_oos = len(oos[: args.n])
    in_tokens = [o["usage"]["input_tokens"] for o, _ in runs]
    out_tokens = [o["usage"]["output_tokens"] for o, _ in runs]
    print(
        {
            "model": config.AGENT_MODEL,
            "lang": args.lang,
            "questions": len(items) + n_oos,
            "retrieved_hit": round(retrieved_hits / len(items), 3),
            "cited_hit": round(cited_hits / len(items), 3),
            "oos_refusal": round(refusals / n_oos, 3),
            "avg_searches": round(statistics.mean(o["searches"] for o, _ in runs), 2),
            "avg_input_tokens": round(statistics.mean(in_tokens)),
            "avg_output_tokens": round(statistics.mean(out_tokens)),
            "median_latency_s": round(statistics.median(s for _, s in runs), 1),
        }
    )


if __name__ == "__main__":
    main()
