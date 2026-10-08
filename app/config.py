import os
from pathlib import Path

DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
COLLECTION = "tech_docs_" + "".join(c if c.isalnum() else "_" for c in EMBEDDING_MODEL.split("/")[-1])[:50]
TOP_K = int(os.getenv("TOP_K", "4"))
# Distance cosinus max (0 = identique) au-delà de laquelle un passage est jugé hors-sujet.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.50"))
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
CHUNK_SIZE = 900  # caractères
CHUNK_OVERLAP = 150
