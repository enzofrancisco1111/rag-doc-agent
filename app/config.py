import os
from pathlib import Path

DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION = "tech_docs"
TOP_K = int(os.getenv("TOP_K", "4"))
# Distance cosinus max (0 = identique) au-delà de laquelle un passage est jugé hors-sujet.
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.55"))
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
CHUNK_SIZE = 900  # caractères
CHUNK_OVERLAP = 150
