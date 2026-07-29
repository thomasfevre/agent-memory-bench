# Validation primaire : fusion hybride, LoCoMo et LongMemEval

Date : 2026-07-29  
Périmètre : trois vérifications, exclusivement à partir des publications et dépôts officiels.

## 1. Reciprocal Rank Fusion

Dans l’article original de Cormack, Clarke et Büttcher, un document \(d\) reçoit :

\[
\operatorname{RRFscore}(d)=\sum_{r\in R}\frac{1}{k+r(d)}
\]

où \(R\) est l’ensemble des classements et \(r(d)\) le rang de \(d\) dans le classement \(r\). Les auteurs fixent **\(k=60\)** pendant une étude pilote, puis ne le modifient plus pendant la validation. Leur premier pilote indiquait que 60 était proche de l’optimum, sans que le choix soit critique. La formule publiée additionne les classements avec le même poids : une « weighted RRF » est donc une extension, pas la formule originale.

Sources : [texte original hébergé par l’Université de Waterloo](https://cormack.uwaterloo.ca/cormacksigir09-rrf), [notice ACM, DOI 10.1145/1571941.1572114](https://dl.acm.org/doi/10.1145/1571941.1572114).

## 2. LoCoMo : catégories et évaluation publiée

Le papier définit cinq catégories QA :

1. **single-hop** : réponse issue d’une seule session ;
2. **multi-hop** : synthèse de plusieurs sessions ;
3. **temporal** : raisonnement sur les dates et indices temporels ;
4. **open-domain knowledge** : information du dialogue combinée à du sens commun ou des connaissances externes ;
5. **adversarial** : question non répondable que le système doit reconnaître comme telle.

Dans les labels numériques du fichier publié, la correspondance est : **4 = single-hop, 1 = multi-hop, 2 = temporal, 3 = open-domain, 5 = adversarial**. Les 1 986 questions des dix conversations se répartissent respectivement en 841, 282, 321, 96 et 446 exemples.

Le code officiel applique un F1 lexical aux catégories 2, 3 et 4, un F1 partiel par sous-réponse à la catégorie 1, et donne 1 à la catégorie 5 uniquement si la sortie contient « no information available » ou « not mentioned ». Pour le RAG, le rappel est la proportion des identifiants de dialogues de preuve récupérés. L’implémentation fournie peut indexer dialogues, observations ou résumés de session, avec Contriever comme retriever par défaut et un `top-k` configurable.

Sources : [papier officiel](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/static/paper/locomo.pdf), [README et format](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/README.MD), [données annotées](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json), [évaluation QA](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py), [runner RAG](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluate_qa.py).

## 3. LongMemEval : décomposition, retrieval et abstention

LongMemEval contient 500 questions couvrant extraction d’information, raisonnement multi-session, mises à jour, temporalité et abstention. Le format distingue six types de questions répondables ; un identifiant terminé par `_abs` marque l’abstention. Les variantes officielles sont **S** (environ 115 000 tokens et 40 sessions), **M** (500 sessions) et **oracle** (sessions de preuve uniquement).

Le papier décompose un système en trois étapes, **indexing, retrieval, reading**, avec quatre points de contrôle : valeur stockée, clé d’index, requête et stratégie de lecture. Le runner officiel compare BM25 et plusieurs retrievers denses, au niveau tour ou session. Il calcule `recall_any`, `recall_all` et `nDCG` aux rangs 1, 3, 5, 10, 30 et 50.

Limite essentielle : les **30 questions d’abstention sont exclues des métriques de retrieval**, car elles n’ont pas de localisation de preuve correcte. L’abstention est évaluée seulement en bout de chaîne, par un juge LLM qui vérifie si la réponse reconnaît explicitement que la question est non répondable. Un seuil de score de retrieval n’est donc pas, à lui seul, le protocole officiel d’abstention.

Sources : [papier ICLR 2025](https://openreview.net/forum?id=pZiyCaVuti), [README officiel](https://github.com/xiaowu0162/longmemeval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/README.md), [métriques de retrieval](https://github.com/xiaowu0162/longmemeval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/retrieval/eval_utils.py), [runner](https://github.com/xiaowu0162/longmemeval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/retrieval/run_retrieval.py), [juge QA et abstention](https://github.com/xiaowu0162/longmemeval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py).

## Pourquoi régler poids et budget sur des groupes tenus à l’écart

Les poids de fusion, \(k\), la profondeur des candidats et le budget de contexte sont des hyperparamètres : ils modifient à la fois le rappel, le bruit transmis au lecteur, la latence et le coût. Les choisir sur les questions finales transforme le test en jeu de réglage et surestime la généralisation. Les questions partageant une même conversation sont corrélées ; les séparer entre réglage et test laisse fuiter vocabulaire, personnes et événements. Pour LoCoMo, il faut donc tenir des **conversations entières** à l’écart, idéalement par validation croisée leave-one-conversation-out. Plus généralement, tous les exemples issus d’un même historique ou scénario doivent rester dans le même groupe. Le poids et le budget sont choisis uniquement sur les groupes d’entraînement/validation, puis gelés avant le score du groupe test.
