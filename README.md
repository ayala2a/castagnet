# CastagNet — tri automatique de châtaignes par vision (MESPR BIHAR)

> **Projet étudiant** — Mise en Situation Professionnelle (MESPR), MSc BIHAR / ESTIA.
> Réalisé par **Mario Caballero** (compte GitHub *ayala2a*). Classifieur de châtaignes
> sèches pour une ligne de tri industrielle, du travail sur la donnée jusqu'au modèle
> exportable en production.

Classification d'images de châtaignes sèches en **4 classes** — Conforme / NON
Conforme / PIETRA / Vide — à partir de **2 vues caméra** (dessus **T** + dessous
**B**) par fruit, pour une ligne de tri à 12 caméras.

Ce dépôt regroupe le travail des trois volets du sujet : qualité de la donnée
(§4.1), modélisation & comparaison d'architectures (§4.2), export ONNX & latence
production (§4.3).

## Livrables (pour l'évaluation)

- **Rapport PDF** : **`reports/RAPPORT_CastagNet.pdf`** (rapport de projet — court et
  accessible). Détails techniques : **`reports/ANNEXE_TECHNIQUE.md`**.
- **Modèle ONNX** : `model_dualbranch.onnx` + `model_dualbranch.onnx.data` (racine du
  dépôt), aussi téléchargeable dans la **[Release v1.0](../../releases/tag/v1.0)**.
- **Données & labels** : `data/labels_principal.csv` (mis à jour), `data/pairs_TB.csv`
  et `data/video_pairs.csv` (correspondances T/B dataset + vidéo).
- **Traces MLflow** : `src/training/mlflow.db` (`mlflow ui`) + figures dans `reports/figures/`.
- **Code** : `src/data/` (audit, appariement, vidéo) + `src/training/` (splits, modèles,
  entraînement, évaluation, export, inférence).

## Principe clé : une châtaigne = une paire (T, B)

Une châtaigne physique est photographiée par deux caméras. **La décision de classe
se prend sur la paire, pas sur une image isolée** : ~34 % des châtaignes présentes
apparaissent « Vide » sur une seule des deux vues (positionnement). L'analyse montre
que les vues T et B ne sont **jamais en conflit** sur une vraie classe — le seul écart
possible est « une face Vide ». D'où la règle d'agrégation *non-Vide gagne*
(détails dans `reports/ANNEXE_TECHNIQUE.md`).

## Structure

```
castagnet/
├── data/
│   ├── pairs_TB.csv          # 1 ligne / châtaigne : vues T & B, labels, tags
│   └── labels_chestnut.csv   # labels agrégés par châtaigne (cible du split)
├── reports/
│   ├── quality_report.md     # §4.1 — rapport qualité actualisé
│   ├── quality_stats.json    # stats brutes reproductibles
│   └── figures/              # graphiques du rapport
├── src/
│   ├── data/
│   │   ├── build_pairs.py    # appariement T/B + agrégation label châtaigne
│   │   └── quality_report.py # figures & stats §4.1
│   ├── training/             # §4.2 (à venir) : make_splits / models / train / evaluate / export_onnx
│   └── utils/
├── configs/                  # configs d'expériences (yaml)
└── notebooks/
```

## Données

Le dataset (images) appartient au projet CastagnIA / GRPTMC ; **les images** ne sont
pas dans ce dépôt (suivies par DVC dans le dépôt data d'origine). Les **CSV de labels
et de correspondances** sont fournis dans `data/` et sont aussi **régénérables** via :

```bash
python src/data/build_pairs.py      # -> data/pairs_TB.csv, data/labels_chestnut.csv
python src/training/make_splits.py  # -> data/splits_*.csv
python src/data/video_pairing.py    # -> data/video_pairs.csv (correspondance T/B vidéo)
```

## Modèle entraîné

Le modèle final est fourni au format ONNX, en **deux fichiers à garder ensemble** :
`model_dualbranch.onnx` (graphe) + `model_dualbranch.onnx.data` (poids). Voir aussi
la **Release** du dépôt pour le télécharger directement. Inférence :
`python src/training/predict.py --t vue_dessus.jpg --b vue_dessous.jpg`.

## Installation

```bash
# venv du projet (déjà créé à la racine marron/)
source ../.venv/bin/activate     # torch, onnx, opencv, mlflow, sklearn...
```

## Reproduire l'étape 1 (qualité §4.1)

```bash
python src/data/build_pairs.py      # -> data/pairs_TB.csv, data/labels_chestnut.csv
python src/data/quality_report.py   # -> reports/figures/*.png, reports/quality_stats.json
```

## Feuille de route

- [x] **§4.1** Appariement T/B + rapport qualité actualisé (par châtaigne)
- [x] **§4.1** Extraction vidéo (2×750 frames) + crop circulaire des fruits
- [x] **§4.1** Relectures terminées (100 %)
- [x] **§4.2** Splits groupés anti-leak + CNN maison + dual-branch T/B (fusion |T−B|) + MLflow
- [x] **§4.2** Cible cahier des charges atteinte (Conforme P=0,953 / R=0,884 sur test)
- [x] **§4.3** Export ONNX vérifié + latence + analyse 12 flux
- [x] **Rapport final** : `reports/RAPPORT_FINAL.md`

Modèle retenu : dual-branch MobileNetV3-Large, fusion `[T, B, |T−B|]`, + TTA.

## Contraintes cahier des charges (Annexe A)

- Classe **Conforme** : **rappel ≥ 85 %** et **précision ≥ 95 %** (priorité pureté du lot).
- Temps réel : 12 flux caméra, cadence 100 kg/h. Cible : GTX 1060 3 Go / i7-980 / Linux.
