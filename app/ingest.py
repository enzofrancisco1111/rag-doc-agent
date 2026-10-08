"""Ingestion : lecture des .md/.txt, découpage par titres puis par taille, vectorisation dans ChromaDB."""
import hashlib
import re
from pathlib import Path

from . import config
from .store import get_collection


FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.S)
JSX_TAG = re.compile(r"</?[A-Z][A-Za-z]*[^>\n]*>")  # <Tip>, <Expandable ...>, </Note>...
DOC_BASE_URL = "https://docs.langchain.com/oss/python/langgraph/"


def clean_mdx(text: str) -> tuple[str, str | None]:
    """Retire le front-matter et les balises JSX ; renvoie (texte, titre de page)."""
    title = None
    m = FRONTMATTER.match(text)
    if m:
        t = re.search(r"^title:\s*(.+)$", m.group(1), re.M)
        title = t.group(1).strip().strip("\"'") if t else None
        text = text[m.end():]
    return JSX_TAG.sub("", text), title


def source_url(source: str) -> str | None:
    if not source.startswith("langgraph/"):
        return None
    name = Path(source).stem
    return DOC_BASE_URL + ("errors/" if name.isupper() else "") + name


def split_sections(text: str, default_title: str = "Introduction") -> list[tuple[str, str]]:
    """Découpe un markdown en (titre, contenu) selon les lignes '#' (hors blocs de code)."""
    sections, title, buf, in_code = [], default_title, [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                sections.append((title, "\n".join(buf).strip()))
            title, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((title, "\n".join(buf).strip()))
    return sections


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Fenêtres glissantes, coupées de préférence en fin de paragraphe ou de phrase."""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def build_chunks(path: Path, root: Path) -> list[dict]:
    text, page_title = clean_mdx(path.read_text(encoding="utf-8", errors="ignore"))
    source = path.relative_to(root).as_posix()
    url = source_url(source)
    out = []
    for title, body in split_sections(text, page_title or "Introduction"):
        if len(body) < 40:
            continue
        for i, chunk in enumerate(chunk_text(body)):
            cid = hashlib.sha1(f"{source}|{title}|{i}|{chunk}".encode()).hexdigest()[:16]
            out.append({"id": cid, "text": f"{page_title or Path(source).stem} › {title}\n{chunk}", "meta": {"source": source, "section": title, "url": url or ""}})
    return out


def ingest(docs_dir: Path | None = None, reset: bool = True) -> dict:
    root = Path(docs_dir or config.DOCS_DIR)
    files = sorted(p for p in root.rglob("*") if p.name.upper() != "README.MD" and p.suffix.lower() in {".md", ".mdx", ".txt"})
    chunks = [c for f in files for c in build_chunks(f, root)]
    col = get_collection()
    if reset and col.count():
        col.delete(ids=col.get()["ids"])
    if chunks:
        col.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["meta"] for c in chunks],
        )
    return {"files": len(files), "chunks": len(chunks)}


if __name__ == "__main__":
    print(ingest())
