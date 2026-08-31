# CastagNet — tri automatique de châtaignes par vision (MESPR BIHAR)

Classification d'images de châtaignes sèches en **4 classes** — Conforme / NON
Conforme / PIETRA / Vide — à partir de **2 vues caméra** (dessus **T** + dessous
**B**) par fruit, pour une ligne de tri à 12 caméras.

Ce dépôt regroupe le travail des trois volets du sujet : qualité de la donnée
(§4.1), modélisation & comparaison d'architectures (§4.2), export ONNX & latence
production (§4.3).

## Principe clé : une châtaigne = une paire (T, B)

Une châtaigne physique est photographiée par deux caméras. **La décision de classe
se prend sur la paire, pas sur une image isolée** : ~34 % des châtaignes présentes
apparaissent « Vide » sur une seule des deux vues (positionnement). L'analyse montre
que les vues T et B ne sont **jamais en conflit** sur une vraie classe — le seul écart
possible est « une face Vide ». D'où la règle d'agrégation *non-Vide gagne*
(voir `reports/quality_report.md`).

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

## Données sources

Le dataset (images + labels) vit dans le dépôt voisin `castagnia_data-main/`
(35 254 images suivies par DVC, `labels_principal.csv`, `labels_masked.csv`). Ce
dépôt-ci n'y touche pas ; il consomme les CSV en lecture.

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
- [ ] **§4.1** Extraction vidéo (2×750 frames) + crop autour des fruits
- [ ] **§4.1** Terminer les 270 relectures restantes
- [ ] **§4.2** Splits groupés anti-leak + CNN maison (baseline) + CNN dual-input T/B + MLflow
- [ ] **§4.3** Export ONNX + mesure de latence vs 12 flux (cible GTX 1060 3 Go)

## Contraintes cahier des charges (Annexe A)

- Classe **Conforme** : **rappel ≥ 85 %** et **précision ≥ 95 %** (priorité pureté du lot).
- Temps réel : 12 flux caméra, cadence 100 kg/h. Cible : GTX 1060 3 Go / i7-980 / Linux.
