# Backlog

Fonctionnement : un ticket = une branche (`feat/...`, `fix/...`) = une Pull Request relue puis fusionnée dans `main` quand la CI est verte. On ne pousse pas directement sur `main`.

## Definition of Done (valable pour tout ticket)
- [ ] Critères d'acceptation du ticket remplis
- [ ] Tests automatiques ajoutés ou mis à jour, **hors-ligne** (aucun appel payant en CI)
- [ ] CI verte (tests + build Docker)
- [ ] Aucun secret dans le code ni dans l'historique git
- [ ] Une décision d'architecture notable = un fichier dans `adr/`
- [ ] README et ce backlog mis à jour

## Statuts : À faire · En cours · En revue · Fait

| ID | Ticket | Critères d'acceptation | Statut |
|---|---|---|---|
| ENG-1 | Boucle agentic (recherche pilotée par le modèle) | Le modèle décide de chercher, reformule si rien n'est trouvé, boucle bornée (recherches et appels plafonnés), refus hors-sujet conservé, mémoire par `thread_id`, repli sur le workflow sans clé API, tests hors-ligne | En revue |
| ENG-2 | Évaluer agentic vs workflow | Jeu de test étendu (≥ 60 questions), mesure qualité / tokens / latence des deux modes, résultats dans le README, décision documentée sur le mode par défaut | À faire |
| ENG-3 | Observabilité | Traces (LangSmith ou Langfuse), tokens et coût par requête, métriques Prometheus (`/metrics`), tableau Grafana dans Docker Compose | À faire |
| ENG-4 | Sécurité de l'API | Clé d'API, limitation de débit, journaux structurés (sans données sensibles), CORS restreint, entrée validée | À faire |
| ENG-5 | Qualité de génération | Mesure de fidélité aux sources (LLM juge ou vérification des citations), seuil dans la CI | À faire |
| ENG-6 | Qualité de la recherche | Recherche hybride (BM25 + vecteurs), reranking, passages alignés sur la limite d'entrée du modèle d'embeddings ; gain mesuré | À faire |
| ENG-7 | Qualité du code en CI | Lint (ruff), couverture, scan des dépendances, build de l'image | À faire |
| ENG-8 | Infrastructure AWS (Terraform) | ECR + ECS Fargate + équilibreur, clé dans Secrets Manager, droits IAM minimaux, alerte de budget, mémoire des conversations persistante, `terraform destroy` documenté | À faire |
| ENG-9 | Livraison continue | Déploiement automatique depuis `main` vers un environnement de test, promotion manuelle vers la démo, retour arrière documenté | À faire |

## Dette connue
- 22 questions d'évaluation seulement : écarts inférieurs à 5 points non significatifs.
- La mémoire des conversations est en processus (perdue au redémarrage).
- L'interface web n'envoie pas encore de `thread_id`.
