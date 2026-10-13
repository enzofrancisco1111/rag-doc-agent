import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from . import config
from .agent import build_agent_graph
from .graph import build_graph
from .ingest import ingest
from .store import get_collection


STATIC = Path(__file__).parent / "static"


class Question(BaseModel):
    question: str = Field(min_length=3)
    thread_id: str | None = None  # mode agentic : même identifiant = même conversation


def agentic_enabled() -> bool:
    """Le mode agentic exige un modèle ; sans clé API on retombe sur le workflow (extractif)."""
    return config.AGENT_MODE == "agentic" and bool(os.getenv("ANTHROPIC_API_KEY"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_graph()
    app.state.agent = build_agent_graph(MemorySaver())  # mémoire en processus : voir ticket ENG-8 pour la persistance
    if get_collection().count() == 0:
        ingest()
    yield


app = FastAPI(title="Doc Support Agent", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": get_collection().count(),
        "llm": bool(os.getenv("ANTHROPIC_API_KEY")),
        "mode": "agentic" if agentic_enabled() else "workflow",
    }


@app.post("/ingest")
def reingest():
    return ingest()


@app.post("/ask")
def ask(q: Question):
    start = time.perf_counter()
    if agentic_enabled():
        thread_id = q.thread_id or uuid.uuid4().hex
        out = app.state.agent.invoke(
            {"messages": [{"role": "user", "content": q.question}]},
            {"configurable": {"thread_id": thread_id}},
        )
        return {
            "answer": out["answer"],
            "sources": out["sources"],
            "thread_id": thread_id,
            "usage": {**out["usage"], "searches": out["searches"]},
            "latency_ms": round((time.perf_counter() - start) * 1000),
        }
    out = app.state.graph.invoke({"question": q.question})
    return {"answer": out["answer"], "sources": out["sources"], "latency_ms": round((time.perf_counter() - start) * 1000)}
