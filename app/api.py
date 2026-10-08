import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .graph import build_graph
from .ingest import ingest
from .store import get_collection


STATIC = Path(__file__).parent / "static"


class Question(BaseModel):
    question: str = Field(min_length=3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_graph()
    if get_collection().count() == 0:
        ingest()
    yield


app = FastAPI(title="Doc Support Agent", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": get_collection().count(), "llm": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.post("/ingest")
def reingest():
    return ingest()


@app.post("/ask")
def ask(q: Question):
    out = app.state.graph.invoke({"question": q.question})
    return {"answer": out["answer"], "sources": out["sources"]}
