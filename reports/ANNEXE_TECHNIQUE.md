# Annexe technique — CastagNet

> Complément au rapport `RAPPORT_CastagNet.pdf`. Regroupe les chiffres, les choix
> techniques justifiés, la correspondance avec la grille d'évaluation et les outils.
> Reproductible : voir `README.md` et `src/`.

## 1. Chiffres clés du dataset

- **35 254 images** (720×480), 2 années (2025 / 2026), 6 postes × 2 vues, 4 classes.
- **Classe « Vide » absente du rapport fourni** : en réalité **12 561 images (35,6 %)**
  reclassées Vide à la relecture. Aucun changement entre vraies classes (le seul
  mouvement est *classe → Vide*).
- **Appariement T/B** (clé = nom de fichier sans la position caméra) → **19 078 châtaignes**
  (16 176 paires complètes + 2 902 orphelins). **100 % des désaccords T/B impliquent
  « Vide » d'un côté** → règle d'agrégation *non-Vide gagne, sinon la plus sévère*.
- Répartition **par châtaigne** : Conforme 37 % · PIETRA 22 % · Vide 21 % · NON Conforme 19 %.
- **Vidéos** : 2 clips 30 s / 750 frames, décalage mesuré **A = B + 5 frames** (corrélation
  des séquences d'arrivée, cadence commune 27 frames) → **20 paires T/B** vidéo.
- **Split** : 70/15/15, stratifié (classe × année), **groupé par châtaigne** → 0 fuite T/B.

## 2. Choix techniques justifiés (why / why-not)

| Décision | Choix | Pourquoi |
|---|---|---|
| Unité d'apprentissage | la châtaigne (paire T/B) | ~34 % des fruits paraissent « Vide » sur une seule vue |
| Agrégation label | non-Vide gagne, sinon plus sévère | 0 conflit réel ; protège la pureté du lot (cahier des charges) |
| Split | `StratifiedGroupKFold` groupé par châtaigne | évite la fuite T/B (sinon scores faussés) |
| Baseline | CNN maison (1 image) | référence exigée ; montre la limite mono-vue (PIETRA rappel 0,17) |
| Modèle | dual-branch siamois (poids partagés) | 2 vues → 1 décision ; moitié moins de paramètres |
| Backbone | MobileNetV3-Large | léger/rapide → tient sur GTX 1060 3 Go, 12 flux |
| Fusion | `[T, B, |T−B|]` | un défaut vu d'un seul côté crée une asymétrie entre vues |
| Représentation | radiale (déroulé polaire) | cohérence radiale + invariance à la rotation → **meilleur modèle** |
| Prétraitement | center-crop + masque circulaire | ignore le fond, garde le disque utile |
| Augmentation | rotation 0–360°, TTA | disque invariant en rotation ; TTA = robustesse gratuite |
| Déséquilibre | loss pondérée | ne pas négliger NON Conforme / PIETRA |
| Décision | seuil calibré sur la précision Conforme | contrainte asymétrique ; l'argmax n'optimise pas ça |
| Suivi | MLflow | comparer les runs à égalité, sélection sur le rappel @P95 |
| Export | ONNX, batch dynamique, FP16 | portable, rapide, tient en 3 Go |
| Pas de YOLO | classification, pas détection | le fruit est déjà centré |
| Pas de Lightning/Hydra | boucle PyTorch nue | lisibilité, moins de dépendances |

## 3. Résultats (jeu de test) et robustesse

| Modèle | Accuracy | Conforme P / R |
|---|---|---|
| SimpleCNN (baseline, 1 vue) | 0,659 | 0,954 / 0,296 |
| DualBranch concat | 0,861 | 0,955 / 0,774 |
| DualBranch `|T−B|` + TTA | 0,913 | 0,953 / 0,884 |
| **DualBranch radial + TTA (retenu)** | **0,923** | **0,951 / 0,925** |

**Validation croisée 5 plis** (stabilité) : précision 0,949 ± 0,012 · rappel 0,868 ± 0,019
· accuracy 0,887 ± 0,012 → résultat stable, pas un coup de chance.

**Production** : 3,71 M paramètres · ONNX ~15,3 Mo (≈7,7 Mo en FP16) · mémoire ~220 Mo/cycle
· latence ~48 ms/cycle (CPU dev) → compatible 12 flux, large marge sur 3 Go.

## 4. Correspondance grille d'évaluation

| Compétence | Où c'est traité |
|---|---|
| C28 — problématique data | §2 rapport, classe Vide, hétérogénéité du diagnostic |
| C29 — modèles & métriques | §3, comparatif, seuil calibré, MLflow, validation croisée |
| C30 — exploitation/optim données | prétraitement, augmentation, export ONNX & latence |
| C31 — organisation données | unité châtaigne, agrégation, split anti-fuite, dépôt Git |
| C32 — conception programme IA | architecture dual-branch/radiale, pipeline `src/` |

## 5. Outils & références

**Outils** : Python 3.13 · PyTorch (MPS) + torchvision (MobileNetV3) · OpenCV · scikit-learn
(`StratifiedGroupKFold`) · MLflow · ONNX / onnxruntime · DVC · Streamlit · Git.

**Repos de référence** : lightning-hydra-template (structure), grading de pommes multi-vues +
SeedSortNet (analogues tri de fruits/graines), timm (backbones légers).

**Proposition — labellisation collaborative** : sharding par (année, caméra), format
append-only, `.gitattributes merge=union`, colonnes de traçabilité obligatoires — pour éviter
les conflits Git sur le CSV de labels.

## 6. Reproduire

```bash
source .venv/bin/activate
python src/data/build_pairs.py && python src/data/quality_report.py
python src/data/video_pairing.py
python src/training/make_splits.py
python src/training/train.py --model dualbranch --backbone mobilenetv3_large \
       --fusion concat_diff --polar --tag _radial --epochs 25
python src/training/evaluate.py --model dualbranch --backbone mobilenetv3_large \
       --fusion concat_diff --polar --tag _radial --tta 4
python src/training/export_onnx.py --model dualbranch --backbone mobilenetv3_large \
       --fusion concat_diff --tag _radial
python src/training/predict.py --t vue_dessus.jpg --b vue_dessous.jpg
```
