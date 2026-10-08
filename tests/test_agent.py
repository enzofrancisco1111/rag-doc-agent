import importlib

import pytest


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    os.environ["CHROMA_DIR"] = str(tmp_path_factory.mktemp("chroma"))
    os.environ.pop("ANTHROPIC_API_KEY", None)
    from app import config

    importlib.reload(config)
    from fastapi.testclient import TestClient

    from app.api import app

    with TestClient(app) as c:
        yield c


def test_split_sections_ignores_headings_in_code_blocks():
    from app.ingest import split_sections

    text = "# A\nfoo bar\n```python\n# not a heading\nx = 1\n```\n## B\nbaz"
    assert [t for t, _ in split_sections(text)] == ["A", "B"]


def test_clean_mdx_strips_frontmatter_and_jsx():
    from app.ingest import clean_mdx

    text, title = clean_mdx("---\ntitle: Interrupts\n---\n<Tip>\nHello\n</Tip>\n")
    assert title == "Interrupts"
    assert "<Tip>" not in text and "Hello" in text


def test_chunking_respects_size():
    from app.ingest import chunk_text

    chunks = chunk_text("Phrase. " * 400, size=500, overlap=50)
    assert len(chunks) > 1 and all(len(c) <= 520 for c in chunks)


def test_ask_cites_sources_with_url(client):
    r = client.post("/ask", json={"question": "How do I pause a graph and wait for human approval?"}).json()
    assert r["sources"], "au moins une source attendue"
    assert any("interrupts" in s["source"] for s in r["sources"])
    assert r["sources"][0]["url"].startswith("https://docs.langchain.com/")


def test_off_topic_returns_no_answer(client):
    r = client.post("/ask", json={"question": "What is the recipe for a tiramisu?"}).json()
    assert r["sources"] == []


def test_validation_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_retrieval_quality_gate(client):
    """Garde-fou : la qualité du retrieval ne doit pas régresser."""
    from eval.run_eval import evaluate

    for lang, floor in (("en", 0.75), ("fr", 0.70)):
        m = evaluate(0.50, lang)
        assert m["hit@4"] >= floor, m
        assert m["oos_rejection"] >= 0.9, m


def test_french_question_finds_english_docs(client):
    r = client.post("/ask", json={"question": "Comment mettre un graphe en pause pour attendre une validation humaine ?"}).json()
    assert any("interrupts" in s["source"] for s in r["sources"])
