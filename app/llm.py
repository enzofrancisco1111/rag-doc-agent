"""Génération de la réponse : Claude si ANTHROPIC_API_KEY est défini, sinon mode extractif."""
import os
import re

from . import config

SYSTEM = (
    "Tu es un assistant de support technique. Réponds uniquement à partir des extraits fournis. "
    "Cite chaque affirmation avec son numéro entre crochets, ex. [1], [2]. "
    "Si les extraits ne contiennent pas la réponse, dis-le clairement. Réponds dans la langue de la question."
)


CODE_BLOCK = re.compile(r"```.*?(```|\Z)", re.S)
MIN_PROSE = 80  # caractères de texte utiles pour qu'un passage soit affichable


def prose(text: str) -> str:
    """Texte lisible d'un passage : sans en-tête vectorisé, blocs de code ni balises ':::'."""
    body = text.split("\n", 1)[1] if "›" in text.split("\n", 1)[0] else text
    body = CODE_BLOCK.sub("", body)
    lines = [l.strip() for l in body.splitlines() if is_prose_line(l)]
    return " ".join(lines)


def is_prose_line(line: str) -> bool:
    """Heuristique : les blocs de code sont souvent coupés au milieu d'un passage, donc on filtre par ligne."""
    s = line.strip()
    if len(s) < 3 or s.startswith((":::", "```", "#", "|", "import ", "from ", "def ", "return ")):
        return False
    if "[!code" in s or s.endswith(("{", "}", ";", ",")):
        return False
    symbols = sum(ch in "{}()[];=<>_\\" for ch in s)
    return symbols / len(s) < 0.06 and " " in s


def extractive_answer(passages: list[dict], limit: int = 600) -> str:
    """Sans LLM : premier passage contenant assez de prose, tronqué en fin de phrase."""
    for i, p in enumerate(passages, 1):
        text = prose(p["text"])
        if text[:1].islower():  # passage coupé en plein milieu d'une phrase : repartir à la phrase suivante
            start = text.find(". ", 0, 250)
            text = text[start + 2:] if start != -1 else text
        if len(text) >= MIN_PROSE:
            if len(text) > limit:
                cut = text.rfind(". ", 0, limit)
                text = text[: cut + 1 if cut > limit // 2 else limit].rstrip() + ("" if cut > limit // 2 else "…")
            return f"[{i}] {p['section']} — {p['source']}\n\n{text}"
    p = passages[0]
    return f"[1] {p['section']} — {p['source']}\n\n(Ce passage est surtout du code : voir le lien dans les sources.)"


def generate(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({p['source']} › {p['section']})\n{p['text']}" for i, p in enumerate(passages, 1)
    )
    if not os.getenv("ANTHROPIC_API_KEY"):
        return extractive_answer(passages)
    import anthropic

    msg = anthropic.Anthropic().messages.create(
        model=config.LLM_MODEL,
        max_tokens=800,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Extraits :\n{context}\n\nQuestion : {question}"}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")
