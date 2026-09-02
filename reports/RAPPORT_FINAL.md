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
Les 2 vidéos (30 s, 750 frames, 720×576) filment le **même créneau sous deux angles**
(vue « haut » et vue « bas » d'une même châtaigne). Pipeline `extract_video.py` :
frames → détection du cercle (HoughCircles) → score de présence → **crop circulaire
masqué** (style dataset) → déduplication temporelle.

**Appariement T/B des vidéos** (`video_pairing.py`) — c'est le point « faire des choix
et les justifier » du sujet. Les deux vidéos ne sont pas synchronisées : il y a un
micro-décalage temporel. Méthode retenue :
1. on détecte la **séquence d'arrivée des châtaignes** dans chaque vidéo (signal de
   présence « beige/brun » par frame) ;
2. les deux séquences ont la **même cadence** (27 frames ≈ 1,08 s entre fruits) →
   c'est bien le même flux ;
3. l'alignement des deux séquences d'événements donne un **décalage de 5 frames
   (0,2 s), la vidéo B en avance sur A** (13 paires concordantes, diffs 4-5-6) ;
4. règle d'appariement : châtaigne à la frame *N* dans B ↔ frame *N+5* dans A ; pour
   chaque fruit on prend la **meilleure vue** dans une fenêtre ±3 frames de chaque
   vidéo.

Résultat : **20 paires T/B** issues des vidéos, validées visuellement (même fruit, même
repère vert, deux angles). Table de correspondance dans `data/video_pairs.csv`. C'est
la même logique T/B que pour le dataset (§2.1), mais reconstruite par **alignement
temporel** au lieu de la clé de nom de fichier.

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

### 3.1bis Comment fonctionne le modèle retenu (pas-à-pas)

1. **Entrée** : les 2 images d'une même châtaigne (vue dessus T + vue dessous B),
   recadrées au centre, masquées en disque, redimensionnées en 224×224.
2. **Extraction** : un même backbone MobileNetV3-Large (poids partagés) transforme
   chaque vue en un vecteur de features (576 dimensions).
3. **Fusion `|T−B|`** : on donne à la suite du réseau le vecteur
   `[features_T, features_B, |features_T − features_B|]`. Le terme `|T−B|` encode
   l'**asymétrie entre les deux faces** — un défaut visible d'un seul côté crée un
   grand écart, signal directement exploitable.
4. **Décision** : une petite tête (MLP) sort 4 scores (Conforme / NON Conforme /
   PIETRA / Vide).
5. **À l'inférence** : **TTA** (on moyenne la prédiction sur quelques rotations), puis
   on n'accepte « Conforme » que si la confiance dépasse un **seuil calibré** — c'est
   ce seuil qui garantit la précision ≥ 95 % exigée.
6. **En production** : 1 cycle = les 12 caméras = 6 paires T/B → 6 décisions calculées
   en un seul appel batché.

### 3.1ter Tests de robustesse (validation croisée)

Pour vérifier que le résultat n'est pas un coup de chance sur un seul découpage, on a
mené une **validation croisée 5 folds** (StratifiedGroupKFold groupé par châtaigne,
config allégée 12 epochs sans TTA). Résultats stables :

| Métrique | Moyenne ± écart-type |
|---|---|
| Conforme précision | 0,949 ± 0,012 |
| Conforme rappel | 0,868 ± 0,019 |
| Accuracy | 0,887 ± 0,012 |

L'écart-type de ~1-2 points confirme la **stabilité** du modèle. Ces modèles de
validation ne sont **pas** le modèle livré : le modèle retenu reste celui entraîné en
30 epochs + TTA (§3.2), déjà au-dessus de la cible ; la validation croisée n'a servi
qu'à en éprouver la fiabilité.

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
  les images d'un tick en un seul appel.
- **Équivalence PyTorch ↔ ONNX vérifiée** (`assert_allclose`, rtol 1e-3).

### 4.1 Taille et coût du modèle

| Poste | Valeur |
|---|---|
| Paramètres | **3,71 M** |
| Poids ONNX total (FP32) | **15,3 Mo** (`model_dualbranch.onnx` 1,0 Mo + `.onnx.data` 14,2 Mo) |
| Poids en FP16 (prod) | **~7,7 Mo** |

> ⚠️ L'ONNX est en **deux fichiers** (graphe + poids externes) : livrer les deux
> ensemble. Le backbone étant **partagé** entre les vues T et B, le nombre de
> paramètres ne double pas.

### 4.2 Empreinte mémoire (mesurée, onnxruntime)

| Situation | RAM (RSS pic) |
|---|---|
| Runtime + modèle chargé (repos) | ~75 Mo |
| **1 tick = 6 paires T/B (12 images)** | **~220 Mo** |
| Batch 12 paires | ~360 Mo |
| Batch 24 | ~660 Mo |

La mémoire est dominée par les **activations** (le poids ne fait que 15 Mo). Sur les
**3 Go de la GTX 1060**, la marge est d'un **facteur ~10** ; en FP16, l'empreinte est
encore ~divisée par deux.

### 4.3 Latence et débit (onnxruntime, CPU de dev — indicatif)

| Batch | p50 | débit |
|---|---|---|
| 1 (1 châtaigne) | 5,5 ms | 168 img/s |
| 12 | 47,9 ms | 240 img/s |
| 24 | 98,9 ms | 241 img/s |

**1 tick = 6 paires** → ~48 ms sur CPU de dev, soit ~**240 images/s** (≈ 120
châtaignes/s), très au-dessus de la cadence de la ligne (100 kg/h). Sur la cible
**GTX 1060 (CUDA/FP16)**, la latence sera nettement plus basse.

### 4.4 Bilan production

Le modèle traite **2 images par châtaigne**, **12 images (6 paires) par cycle**, pour
un coût total très modeste : **~15 Mo sur disque**, **~220 Mo de RAM en fonctionnement**,
**~48 ms par cycle**. Toutes les contraintes du cahier des charges (précision, rappel,
temps réel sur 12 flux, tenue en 3 Go) sont satisfaites, la mémoire étant le point le
plus confortable.

**Recommandation finale** : retenir le **dual-branch `concat_diff` + TTA** (respecte la
cible avec marge : P=0,953 / R=0,884). Si la latence GPU imposait de couper le TTA, le
modèle sans TTA respecte déjà la contrainte (P=0,951 / R=0,873) — c'est l'arbitrage
précision ↔ latence à trancher sur le matériel cible.

### 4.5 Utilisation (inférence)

Le modèle s'utilise via `src/training/predict.py`, qui charge l'ONNX (onnxruntime, sans
PyTorch) et classe **une châtaigne à partir de ses deux vues** :

```bash
source .venv/bin/activate

# à partir des deux images d'une même châtaigne
python src/training/predict.py --t vue_dessus.jpg --b vue_dessous.jpg

# ou à partir d'un identifiant du dataset (retrouve T et B automatiquement)
python src/training/predict.py --pairkey 2025_PIETRA_Cam_X_1_100.jpg
```

Sortie : les probabilités des 4 classes et la décision finale. La classe « Conforme »
n'est retenue que si sa probabilité dépasse le **seuil calibré (0,64)** ; sinon le fruit
est écarté — c'est le mécanisme qui garantit la précision exigée. En interne : même
prétraitement qu'à l'entraînement (center-crop + masque circulaire + normalisation),
TTA sur quelques rotations, puis application du seuil.

Étapes du traitement d'une châtaigne :
`2 images (T, B)` → prétraitement disque → backbone partagé → fusion `[T, B, |T−B|]`
→ tête → probabilités 4 classes → seuil Conforme → **décision**.

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
