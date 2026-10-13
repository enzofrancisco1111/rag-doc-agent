"""Agent RAG sous LangGraph : le modèle décide quand chercher, reformule si besoin, puis répond.

Boucle bornée (MAX_SEARCHES recherches, MAX_STEPS appels au modèle). Le seuil de pertinence reste
appliqué dans l'outil de recherche : l'agent ne reçoit jamais d'extrait hors-sujet.
"""
import operator
import re
from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from . import config
from .retrieval import search

SYSTEM = (
    "Tu es un assistant de support technique pour la documentation LangGraph. "
    "Utilise l'outil search_docs pour chercher les extraits pertinents ; la documentation est en anglais, "
    "formule donc tes requêtes en anglais. Si les extraits ne répondent pas à la question, reformule la requête "
    "et cherche à nouveau (deux reformulations au maximum). "
    "Réponds uniquement à partir des extraits reçus, en citant chaque affirmation avec son numéro entre crochets, "
    "ex. [1], [2]. Si, après recherche, la documentation ne contient pas la réponse, dis-le clairement sans "
    "répondre de mémoire. Si la question n'a aucun rapport avec LangGraph, refuse poliment sans chercher. "
    "Réponds dans la langue de la question, de façon concise, avec un seul exemple de code au maximum."
)

TOOLS = [
    {
        "name": "search_docs",
        "description": (
            "Recherche dans la documentation LangGraph et renvoie des extraits numérotés. "
            "Requête courte en anglais, centrée sur le concept recherché."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Requête de recherche en anglais"}},
            "required": ["query"],
        },
    }
]

NO_RESULT = "Aucun extrait pertinent trouvé pour cette requête."
LIMIT_REACHED = (
    "Limite de recherches atteinte : réponds avec les extraits déjà obtenus, "
    "ou indique que la documentation ne contient pas la réponse."
)
GAVE_UP = "Je n'ai pas pu conclure avec les extraits trouvés dans la documentation."


class AgentState(TypedDict, total=False):
    messages: Annotated[list, operator.add]  # historique Anthropic ; chaque nœud n'ajoute que le nouveau
    passages: Annotated[list, operator.add]  # tous les passages du fil, numérotés [ref]
    searches: int
    steps: int
    usage: dict
    answer: str
    sources: list[dict]


@lru_cache(maxsize=1)
def get_client():
    import anthropic

    return anthropic.Anthropic()


def _block_to_dict(block) -> dict:
    """Blocs du SDK -> dicts simples (sérialisables par le checkpointer et renvoyables à l'API)."""
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)}
    return {"type": "text", "text": block.text}


def _tool_calls(message: dict) -> list[dict]:
    content = message["content"]
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"] if isinstance(content, list) else []


def _text(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(b["text"] for b in content if b.get("type") == "text")


def start_turn(state: AgentState) -> AgentState:
    return {
        "searches": 0,
        "steps": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def call_model(state: AgentState) -> AgentState:
    resp = get_client().messages.create(
        model=config.AGENT_MODEL,
        max_tokens=config.AGENT_MAX_TOKENS,
        system=SYSTEM,
        tools=TOOLS,
        messages=state["messages"],
    )
    usage = state["usage"]
    return {
        "messages": [{"role": "assistant", "content": [_block_to_dict(b) for b in resp.content]}],
        "steps": state["steps"] + 1,
        "usage": {
            "input_tokens": usage["input_tokens"] + resp.usage.input_tokens,
            "output_tokens": usage["output_tokens"] + resp.usage.output_tokens,
        },
    }


def _tool_results(state: AgentState) -> tuple[list[dict], list[dict], int]:
    """Exécute les appels d'outils du dernier message ; renvoie (tool_results, nouveaux passages, recherches faites)."""
    results, new_passages = [], []
    searches = state["searches"]
    known = len(state.get("passages", []))
    for call in _tool_calls(state["messages"][-1]):
        if call["name"] != "search_docs" or searches >= config.MAX_SEARCHES:
            text = LIMIT_REACHED
        else:
            searches += 1
            found = search(str(call["input"].get("query", "")))
            blocks = []
            for p in found:
                p = {**p, "ref": known + len(new_passages) + 1}
                new_passages.append(p)
                blocks.append(f"[{p['ref']}] ({p['source']} › {p['section']})\n{p['text']}")
            text = "\n\n".join(blocks) if blocks else NO_RESULT
        results.append({"type": "tool_result", "tool_use_id": call["id"], "content": text})
    return results, new_passages, searches


def run_tools(state: AgentState) -> AgentState:
    results, new_passages, searches = _tool_results(state)
    return {"messages": [{"role": "user", "content": results}], "passages": new_passages, "searches": searches}


def route_after_model(state: AgentState) -> str:
    wants_tools = bool(_tool_calls(state["messages"][-1]))
    return "tools" if wants_tools and state["steps"] < config.MAX_SEARCHES + 2 else "finish"


def finish(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    update: AgentState = {}
    if _tool_calls(last):
        # Plafond atteint alors que le modèle voulait encore chercher : on referme proprement l'historique
        # (tout tool_use doit être suivi d'un tool_result, sinon le tour suivant du fil serait refusé par l'API).
        closing = [
            {"type": "tool_result", "tool_use_id": c["id"], "content": LIMIT_REACHED} for c in _tool_calls(last)
        ]
        update["messages"] = [
            {"role": "user", "content": closing},
            {"role": "assistant", "content": [{"type": "text", "text": GAVE_UP}]},
        ]
        answer, sources = GAVE_UP, []
    else:
        answer = _text(last)
        # Sources = uniquement les passages réellement cités ; un refus sans citation n'affiche aucune source.
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        used = [p for p in state.get("passages", []) if p["ref"] in cited]
        sources = [
            {"ref": p["ref"], "source": p["source"], "section": p["section"], "url": p["url"], "distance": p["distance"]}
            for p in used
        ]
    return {**update, "answer": answer, "sources": sources}


def build_agent_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("start_turn", start_turn)
    g.add_node("model", call_model)
    g.add_node("tools", run_tools)
    g.add_node("finish", finish)
    g.set_entry_point("start_turn")
    g.add_edge("start_turn", "model")
    g.add_conditional_edges("model", route_after_model, {"tools": "tools", "finish": "finish"})
    g.add_edge("tools", "model")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)
