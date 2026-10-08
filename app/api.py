from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .graph import build_graph
from .ingest import ingest
from .store import get_collection


class Question(BaseModel):
    question: str = Field(min_length=3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_graph()
    if get_collection().count() == 0:
        ingest()
    yield


app = FastAPI(title="Doc Support Agent", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": get_collection().count()}


@app.post("/ingest")
def reingest():
    return ingest()


@app.post("/ask")
def ask(q: Question):
    out = app.state.graph.invoke({"question": q.question})
    return {"answer": out["answer"], "sources": out["sources"]}
