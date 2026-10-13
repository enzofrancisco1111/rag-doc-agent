# ADR 0001 : boucle agentic pour la recherche

Statut : proposé (ticket ENG-1) · Date : 2026-10-13

## Contexte
Le projet est un workflow fixe : `retrieve -> (generate | no_answer)`. Le code décide de tout. Limites : une seule recherche par question, aucune reformulation si la requête tombe à côté, pas de mémoire entre deux questions.

## Décision
Ajouter un mode **agentic** à côté du workflow :
- la recherche devient un outil (`search_docs`) que le modèle appelle quand il le juge utile, avec une requête qu'il formule lui-même (en anglais, la documentation l'étant) ;
- le modèle peut **reformuler** et chercher à nouveau si le résultat ne répond pas ;
- la boucle est **bornée** : `MAX_SEARCHES` recherches (3 par défaut) et `MAX_SEARCHES + 2` appels au modèle, pour plafonner le coût et la latence ;
- le **seuil de pertinence** reste appliqué dans l'outil : l'agent ne reçoit jamais d'extrait hors-sujet, et un refus ne produit aucune source ;
- les **sources** renvoyées sont uniquement les passages cités dans la réponse ;
- mémoire par `thread_id` via le checkpointer LangGraph ;
- activé par `AGENT_MODE=agentic`, **désactivé par défaut**. Sans clé API, repli sur le workflow.

## Alternatives écartées
- **Remplacer le workflow** : on ne sait pas encore si l'agentic fait mieux. Il faut le mesurer d'abord (ENG-2).
- **Framework d'agents prêt à l'emploi** : une boucle explicite de trois nœuds reste lisible, testable avec un faux modèle, et sans dépendance supplémentaire.

## Conséquences
- Plus de souplesse (reformulation, mémoire) au prix de plus d'appels au modèle : plus de tokens, plus de latence, moins de prévisibilité.
- Le comportement dépend du modèle : les tests de la boucle utilisent un faux client, la qualité réelle se mesure par l'évaluation de bout en bout (`eval/run_agent_eval.py`, payante).
- La mémoire est en processus : perdue au redémarrage et non partagée entre instances. À traiter avant un déploiement multi-instances (ENG-8).
