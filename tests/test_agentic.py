"""Tests de l'agent (boucle LangGraph) avec un faux client Anthropic : aucun appel réseau, aucun coût."""
from types import SimpleNamespace

import pytest

from app import agent as agent_module
from app import config


def text(value):
    return SimpleNamespace(type="text", text=value)


def search_call(index, query):
    return SimpleNamespace(type="tool_use", id=f"tu{index}", name="search_docs", input={"query": query})


class FakeClient:
    """Rejoue des réponses scriptées : respond(n, messages) -> liste de blocs pour le n-ième appel."""

    def __init__(self, respond):
        self.respond = respond
        self.seen = []  # messages reçus à chaque appel
        self.messages = self

    def create(self, **kwargs):
        self.seen.append(list(kwargs["messages"]))
        n = len(self.seen) - 1
        return SimpleNamespace(
            content=self.respond(n, kwargs["messages"]), usage=SimpleNamespace(input_tokens=100, output_tokens=20)
        )


@pytest.fixture
def run(client, monkeypatch):
    """Lance un tour de l'agent avec un faux modèle ; `client` garantit que l'index est construit."""
    from langgraph.checkpoint.memory import MemorySaver

    def _run(respond, questions=("Comment mettre un graphe en pause ?",), thread="t1"):
        fake = FakeClient(respond)
        monkeypatch.setattr(agent_module, "get_client", lambda: fake)
        graph = agent_module.build_agent_graph(MemorySaver())
        out = None
        for q in questions:
            out = graph.invoke({"messages": [{"role": "user", "content": q}]}, {"configurable": {"thread_id": thread}})
        return out, fake

    return _run


def assert_valid_history(messages):
    """Règle de l'API : chaque tool_use est immédiatement suivi d'un tool_result portant le même id."""
    for i, m in enumerate(messages):
        calls = [b for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"] if isinstance(m["content"], list) else []
        if calls:
            nxt = messages[i + 1]["content"]
            assert {b["tool_use_id"] for b in nxt} == {c["id"] for c in calls}


def test_agent_searches_then_cites_retrieved_passages(run):
    def respond(n, messages):
        return [search_call(n, "interrupt pause graph human approval")] if n == 0 else [text("Voir [1] [2] [3] [4].")]

    out, _ = run(respond)
    assert out["searches"] == 1
    assert out["sources"] and any("interrupts" in s["source"] for s in out["sources"])
    assert out["usage"] == {"input_tokens": 200, "output_tokens": 40}


def test_agent_reformulates_when_first_search_finds_nothing(run):
    queries = ["tiramisu recipe", "interrupt pause graph human approval"]

    def respond(n, messages):
        return [search_call(n, queries[n])] if n < 2 else [text("Voir [1] [2] [3] [4].")]

    out, fake = run(respond)
    assert out["searches"] == 2
    assert fake.seen[1][-1]["content"][0]["content"] == agent_module.NO_RESULT  # le modèle a été prévenu
    assert out["sources"]


def test_agent_refuses_off_topic_without_searching(run):
    out, _ = run(lambda n, m: [text("Je ne peux répondre qu'aux questions sur LangGraph.")], ["Recette de tiramisu ?"])
    assert out["searches"] == 0 and out["sources"] == []


def test_agent_stops_after_search_limit_and_keeps_history_valid(run):
    out, fake = run(lambda n, m: [search_call(n, "interrupt")])
    assert out["answer"] == agent_module.GAVE_UP and out["sources"] == []
    assert len(fake.seen) == config.MAX_SEARCHES + 2  # la boucle est bornée
    assert_valid_history(out["messages"])


def test_agent_remembers_within_a_thread_only(run):
    answer = lambda n, m: [text("Ok.")]  # noqa: E731
    _, fake_same = run(answer, ["Je m'appelle Sam.", "Quel est mon prénom ?"], thread="a")
    _, fake_new = run(answer, ["Quel est mon prénom ?"], thread="b")
    assert len(fake_same.seen[1]) == 3  # question 1, réponse 1, question 2
    assert len(fake_new.seen[0]) == 1


def test_api_agentic_mode_returns_thread_usage_and_latency(client, monkeypatch):
    from app import api

    fake = FakeClient(lambda n, m: [search_call(n, "interrupt pause graph")] if n == 0 else [text("Voir [1] [2].")])
    monkeypatch.setattr(agent_module, "get_client", lambda: fake)
    monkeypatch.setattr(config, "AGENT_MODE", "agentic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert api.agentic_enabled()
    r = client.post("/ask", json={"question": "Comment mettre un graphe en pause ?"}).json()
    assert r["thread_id"] and r["usage"]["searches"] == 1 and r["latency_ms"] >= 0
    assert client.get("/health").json()["mode"] == "agentic"


def test_agentic_mode_falls_back_to_workflow_without_api_key(client, monkeypatch):
    monkeypatch.setattr(config, "AGENT_MODE", "agentic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert client.get("/health").json()["mode"] == "workflow"
    assert "thread_id" not in client.post("/ask", json={"question": "How do I install LangGraph?"}).json()
