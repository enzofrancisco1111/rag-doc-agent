"""Workflow LangGraph : retrieve -> (generate | no_answer)."""
from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import llm
from .retrieval import search


class State(TypedDict, total=False):
    question: str
    passages: list[dict]
    answer: str
    sources: list[dict]


def retrieve(state: State) -> State:
    return {"passages": search(state["question"])}


def generate(state: State) -> State:
    answer = llm.generate(state["question"], state["passages"])
    sources = [
        {"ref": i, "source": p["source"], "section": p["section"], "url": p["url"], "distance": p["distance"]}
        for i, p in enumerate(state["passages"], 1)
    ]
    return {"answer": answer, "sources": sources}


def no_answer(state: State) -> State:
    return {
        "answer": "Je n'ai trouvé aucun passage pertinent dans la documentation pour cette question.",
        "sources": [],
    }


def route(state: State) -> str:
    return "generate" if state["passages"] else "no_answer"


def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("no_answer", no_answer)
    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", route, {"generate": "generate", "no_answer": "no_answer"})
    g.add_edge("generate", END)
    g.add_edge("no_answer", END)
    return g.compile()
