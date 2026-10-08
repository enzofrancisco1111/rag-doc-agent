# Doc Support Agent — agent RAG avec citations

Agent IA qui répond à des questions techniques sur la documentation **LangGraph** (33 pages, ~1 350 passages indexés) **en citant ses sources**, et qui **refuse de répondre** quand la documentation ne contient pas l'information.

**Stack** : LangGraph · ChromaDB · FastAPI · Docker · Claude (optionnel) · pytest · GitHub Actions

## Architecture

```
docs/**/*.mdx ─► ingest.py ─► ChromaDB (embeddings MiniLM locaux, distance cosinus)
                                   │
question ─► FastAPI POST /ask ─► LangGraph
                                   ├─ retrieve  : top-k passages + filtre de pertinence
                                   ├─ generate  : réponse avec citations [1], [2] (Claude)
                                   └─ no_answer : aucun passage pertinent → pas d'hallucination
```

- **Ingestion** : nettoyage MDX (front-matter, balises JSX), découpage par titres en ignorant les blocs de code, puis fenêtres de 900 caractères / 150 de recouvrement. Le titre de page est ajouté au texte vectorisé.
- **Garde-fou anti-hallucination** : un passage dont la distance cosinus dépasse `MAX_DISTANCE` (0,55) est écarté ; sans passage restant, le nœud `no_answer` répond « je ne sais pas » sans appeler le LLM.
- **Citations** : chaque source renvoyée contient le fichier, la section et le lien vers la page officielle.
- **Sans clé API**, l'agent fonctionne en mode extractif (renvoie le passage le plus pertinent). Avec `ANTHROPIC_API_KEY`, Claude rédige la réponse en citant les extraits [n].

## Évaluation

Jeu de test dans [`eval/dataset.json`](eval/dataset.json) : 22 questions avec la page attendue + 7 questions hors-sujet.

| Version du retrieval | hit@4 | MRR | Hors-sujet rejeté |
|---|---|---|---|
| Passages bruts | 86 % | 0,69 | 100 % |
| + titre de page dans le texte vectorisé (actuelle) | **91 %** | **0,82** | **100 %** |

Le seuil `MAX_DISTANCE` a été choisi par balayage (`python -m eval.run_eval --sweep`) : plateau de 0,50 à 0,65 ; 0,55 laisse de la marge des deux côtés. Un test pytest échoue si hit@4 passe sous 85 % (garde-fou de non-régression en CI).

Limites connues : le modèle d'embeddings par défaut est anglophone (questions en anglais recommandées pour la recherche ; Claude répond dans la langue de la question) ; pas de reranking ni de recherche hybride.

## Lancer en local

```bash
pip install -r requirements.txt
uvicorn app.api:app --reload        # http://localhost:8000/docs
```

L'index se construit automatiquement au premier démarrage (ou `python -m app.ingest`).

## Lancer avec Docker

```bash
docker compose up --build
```

Pour utiliser Claude : copier `.env.example` en `.env` et renseigner `ANTHROPIC_API_KEY`.

## API

| Route | Rôle |
|---|---|
| `GET /health` | état + nombre de passages indexés |
| `POST /ingest` | ré-indexe le dossier `docs/` |
| `POST /ask` | `{"question": "..."}` → `{"answer", "sources": [{ref, source, section, url, distance}]}` |

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How do I pause a graph and wait for human approval?"}'
```

## Utiliser sa propre documentation

Déposer des `.md` / `.mdx` / `.txt` dans `docs/` puis `POST /ingest`. Adapter `eval/dataset.json` et relancer l'évaluation pour recalibrer `MAX_DISTANCE`.

## Tests

```bash
pytest        # 7 tests : ingestion, API, citations, rejet hors-sujet, qualité du retrieval
```

## Crédits

La documentation indexée provient de [langchain-ai/docs](https://github.com/langchain-ai/docs) (licence MIT, © 2025 LangChain) — voir `docs/LICENSE-langchain-docs.txt`.

## Pistes d'amélioration

- Embeddings multilingues, recherche hybride (BM25 + vecteurs) et reranking
- Streaming de la réponse et mémoire de conversation (checkpointer LangGraph)
- Évaluation de la qualité de génération (fidélité aux sources) avec un LLM juge
