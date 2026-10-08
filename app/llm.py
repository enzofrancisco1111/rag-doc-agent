"""Génération de la réponse : Claude si ANTHROPIC_API_KEY est défini, sinon mode extractif."""
import os

from . import config

SYSTEM = (
    "Tu es un assistant de support technique. Réponds uniquement à partir des extraits fournis. "
    "Cite chaque affirmation avec son numéro entre crochets, ex. [1], [2]. "
    "Si les extraits ne contiennent pas la réponse, dis-le clairement. Réponds dans la langue de la question."
)


def generate(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({p['source']} › {p['section']})\n{p['text']}" for i, p in enumerate(passages, 1)
    )
    if not os.getenv("ANTHROPIC_API_KEY"):
        best = passages[0]
        return (
            f"(Mode extractif, pas de clé LLM) Passage le plus pertinent [1] "
            f"- {best['source']} › {best['section']} :\n{best['text']}"
        )
    import anthropic

    msg = anthropic.Anthropic().messages.create(
        model=config.LLM_MODEL,
        max_tokens=800,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Extraits :\n{context}\n\nQuestion : {question}"}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")
