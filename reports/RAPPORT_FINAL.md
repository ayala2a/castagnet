# CastagNet — Rapport final

**MSc BIHAR / ESTIA — Mise en Situation Professionnelle (MESPR)**
Classification automatique de châtaignes sèches par vision — 4 classes
(Conforme / NON Conforme / PIETRA / Vide), 2 vues caméra (dessus T + dessous B).

> Document maître. Les analyses détaillées sont dans :
> `quality_report.md` (qualité données §4.1), `choix_justifies.md` (12 décisions
> justifiées), `references_ml.md` (état de l'art). Code reproductible dans `src/`.

---

## 0. Synthèse (résultat clé)

Le système final — un réseau **dual-branch** (deux vues T/B → une décision) avec
fusion **`[T, B, |T−B|]`** et **augmentation au test (TTA)** — **respecte le cahier
des charges** sur le jeu de test :

| Exigence GRPTMC (Annexe A) | Cible | Obtenu (test) |
|---|---|---|
| Précision sur Conforme | ≥ 95 % | **95,3 %** ✅ |
| Rappel sur Conforme | ≥ 85 % | **88,4 %** ✅ |
| Accuracy globale (4 classes) | — | 91,3 % |
| Temps réel 12 flux | pas d'accumulation | ✅ (voir §4.3) |

---

## 1. Problématique data (C28)

- **Volume / variété** : 35 254 images (720×480), 2 années (2025/2026), 6 postes ×
  2 vues, 4 classes déséquilibrées.
- **Découverte majeure** : le rapport dataset fourni est bâti sur `label_filename`
  (3 classes) et **ignore la classe « Vide »** ; en réalité **35,6 % des images**
  ont été reclassées « Vide » lors de la relecture. Le vrai « avant/après » est
  reconstruit (fig. `01_avant_apres.png`).
- **Hétérogénéité du diagnostic** : labels produits par plusieurs opérateurs + un
  process auto (`auto_empty_v1`) ; 7 763 lignes relues sans auteur tracé.
- Détail complet : **`quality_report.md`**.

## 2. Organisation des données (C31)

### 2.1 Une châtaigne = une paire (T, B)
La machine trie des *fruits*, pas des images. On reconstitue **19 078 châtaignes**
(16 176 paires complètes + 2 902 orphelins) via la clé de nom de fichier privée de
la position caméra.

**Preuve du besoin des 2 vues** : ~34 % des châtaignes présentes apparaissent
« Vide » sur *une seule* vue. Analyse des désaccords T/B (fig. `03_desaccords_TB.png`) :
**100 % des divergences impliquent « Vide » d'un côté** — jamais deux vraies classes
en conflit à l'origine. D'où la règle d'agrégation **« non-Vide gagne, puis le plus
sévère »** (protège la pureté du lot Conforme). Après relecture finale, 45 conflits
réels (NC/PIETRA, Conforme/PIETRA) résolus par sévérité.

Répartition **par châtaigne** (rééquilibrée vs par image) :
Conforme 37,3 % · PIETRA 22,3 % · Vide 21,5 % · NON Conforme 18,9 %.

### 2.2 Extraction vidéo (§4.1)
Les 2 vidéos (30 s, 750 frames, 720×576) sont le flux brut d'une caméra filmant un
créneau circulaire. Pipeline `extract_video.py` : frames → détection du cercle
(HoughCircles) → score de présence → **crop circulaire masqué** (style dataset) →
déduplication temporelle → **115 châtaignes** extraites.

### 2.3 Découpage train / val / test (anti-fuite)
`make_splits.py` : **70/15/15**, **stratifié** (classe × année), **groupé par
châtaigne** → les 2 vues d'un fruit tombent dans le même sous-ensemble.
**Fuite T/B vérifiée = 0.** Répartition : 13 354 / 2 862 / 2 862, distributions de
classes identiques à 0,1 % près entre splits.

### 2.4 Labellisation collaborative (proposition)
Sharding par (année, caméra), format append-only, `.gitattributes merge=union`,
colonnes de traçabilité obligatoires — voir `quality_report.md` §8.

## 3. Modélisation et comparaison (C29, C32)

### 3.1 Architectures
1. **SimpleCNN** — CNN compact codé à la main sur image unique (**baseline** exigée).
2. **DualBranch** — 2 backbones **MobileNetV3** à poids partagés (siamois) sur T et B,
   fusion puis tête MLP → 4 classes. Trois fusions testées : `concat`, et surtout
   **`concat_diff` = [T, B, |T−B|]** (donne au réseau l'**asymétrie des deux vues**,
   ciblée sur la confusion Conforme/PIETRA).

Justification complète des choix (backbone léger pour la GTX 1060, prétraitement
disque circulaire, augmentation rotation 360°, loss pondérée, seuil calibré sur la
précision Conforme, MLflow…) : **`choix_justifies.md`** (12 décisions, *why / why-not*).

### 3.2 Résultats sur le jeu de TEST

| Modèle | Accuracy | Conforme P / R (seuil calibré) | Cible |
|---|---|---|---|
| SimpleCNN (baseline) | 0,659 | 0,954 / **0,296** | ❌ |
| DualBranch `concat` | 0,861 | 0,955 / **0,774** | ❌ (rappel) |
| DualBranch `concat_diff` (**\|T−B\|**) | 0,896 | 0,951 / 0,873 | ✅ |
| **`concat_diff` + TTA** (retenu) | **0,913** | **0,953 / 0,884** | ✅ **avec marge** |

### 3.3 Analyse des erreurs
- La baseline mono-vue **échoue sur PIETRA** (rappel 0,17) : le défaut n'est souvent
  visible que d'une face. Les deux vues font passer PIETRA à ~0,85.
- Résidu principal = confusion **Conforme ↔ PIETRA** (fig. `confusion_final.png`) —
  la fusion `|T−B|` la réduit nettement.
- **Les erreurs vont dans le sens sûr** : le modèle rejette à tort des bonnes
  (perte de rendement, tolérée) bien plus qu'il ne laisse passer de mauvaises →
  conforme à la priorité « pureté du lot » du cahier des charges.
- Enseignement d'ingénierie : au-delà d'un plateau (~0,86-0,88 en R@P95), ni plus
  d'epochs, ni 256 px, ni nettoyage n'ont aidé ; **seule la fusion `|T−B|` (archi
  ciblée) + le TTA** ont débloqué le rappel.

### 3.4 Suivi MLflow
Tous les runs (hyperparamètres, métriques *par classe*, `conforme_recall_at_p95`,
matrices de confusion) sont tracés dans MLflow (`mlflow ui`), permettant le
comparatif à égalité et la sélection du meilleur modèle sur l'**objectif métier**.

## 4. Export et compatibilité production (C30)

- **Export ONNX** (`export_onnx.py`) opset dynamique (axe batch) → permet d'empiler
  les **24 images d'un tick (2 vues × 12 flux)** en un seul appel.
- **Équivalence PyTorch ↔ ONNX vérifiée** (`assert_allclose`, rtol 1e-3).
- **Latence** (onnxruntime, CPU de dev — indicatif) :

| Batch | p50 | débit |
|---|---|---|
| 1 (1 châtaigne) | 5,5 ms | 168 img/s |
| 12 | 47,9 ms | 240 img/s |
| 24 (1 tick complet) | 98,9 ms | 241 img/s |

- **Compatibilité 12 flux** : 1 tick = 6 paires → ~48 ms sur *CPU de dev*. Sur la
  cible **GTX 1060 (CUDA/FP16)** la latence sera nettement plus basse → **temps réel
  confortable** pour la cadence 100 kg/h. Modèle **léger** (backbone partagé) → tient
  largement dans les **3 Go**.
- **Recommandation finale** : retenir le **dual-branch `concat_diff`** ; activer le
  **TTA** si la latence GPU le permet (arbitrage précision ↔ latence à mesurer sur la
  cible), sinon le modèle sans TTA respecte déjà la contrainte (P=0,951 / R=0,873).

---

## 5. Traçabilité vers la grille de notation

| Comp. | Critère | Traité en |
|---|---|---|
| C28 | Analyse problématique data | §1 + `quality_report.md` |
| C29 | Choix & évaluation modèles (métriques adaptées) | §3 + `choix_justifies.md` |
| C30 | Exploitation/optimisation données | §2.2, §4 (ONNX, latence) |
| C31 | Organisation données cohérente IA | §2 (châtaigne, splits anti-fuite) |
| C32 | Conception du programme d'IA | §3.1 (dual-branch), pipeline `src/` |

## 6. Reproductibilité

```bash
source .venv/bin/activate
# §4.1
python src/data/build_pairs.py
python src/data/quality_report.py
python src/data/extract_video.py
# §4.2
python src/training/make_splits.py
python src/training/train.py --model dualbranch --backbone mobilenetv3_large \
       --fusion concat_diff --tag _diff --epochs 30
python src/training/evaluate.py --model dualbranch --backbone mobilenetv3_large \
       --fusion concat_diff --tag _diff --tta 4
# §4.3
python src/training/export_onnx.py --model dualbranch --backbone mobilenetv3_large \
       --fusion concat_diff --tag _diff
```
