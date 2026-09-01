# Choix techniques justifiés — CastagNet (§4.2 / §4.3)

> Document destiné au rapport / à la soutenance. Pour chaque décision : **ce qu'on
> fait**, **pourquoi**, et **pourquoi pas l'alternative**. Tout est ancré sur le
> cahier des charges (Annexe A) : classe **Conforme → rappel ≥ 85 % et précision
> ≥ 95 %** ; **12 flux** temps réel sur **GTX 1060 3 Go**.

---

## 1. Unité d'apprentissage : la châtaigne (paire T/B), pas l'image

**Choix** : un échantillon = une châtaigne = ses 2 vues (dessus T + dessous B).
**Pourquoi** : la machine trie des *fruits*, pas des images. L'analyse des données le
prouve : ~34 % des châtaignes présentes apparaissent « Vide » sur *une* des deux vues
(le fruit n'est pas dans le champ de cette caméra). Décider sur une seule image
conduirait donc à classer « Vide » un fruit bien présent 1 fois sur 3.
**Pourquoi pas l'image seule** : perte d'information systématique + ne reflète pas la
décision métier. On garde quand même un modèle *par image* comme **baseline** (voir §4)
pour démontrer chiffres à l'appui l'apport de la 2ᵉ vue.

## 2. Règle d'agrégation du label : « non-Vide gagne, puis le plus sévère »

**Choix** : label_châtaigne = la classe non-Vide parmi {T, B} ; si les deux vues
donnent deux vraies classes (45 cas après relecture), on garde la **plus sévère**
(PIETRA > NON Conforme > Conforme).
**Pourquoi** : (a) une face Vide = angle mort, pas une info ; (b) le cahier des charges
**privilégie la pureté du lot Conforme** → au moindre défaut vu d'un côté, on écarte.
**Pourquoi pas « majorité » ou « moyenne »** : avec 2 vues seulement il n'y a pas de
majorité ; et « moyenner » deux classes n'a pas de sens métier — un défaut est un fait,
pas une moyenne.

## 3. Découpage train / val / test : stratifié **et** groupé par châtaigne

**Choix** : 70 % / 15 % / 15 %, **stratifié** par (classe × année), **groupé** par
châtaigne (les 2 vues T/B d'un fruit vont toujours dans le même sous-ensemble).
**Pourquoi groupé** : sinon la vue T d'un fruit peut être en *train* et sa vue B en
*test* → le modèle « reconnaît le fruit », les scores explosent en test mais
s'effondrent en production (*data leakage*). Outil : `StratifiedGroupKFold` (sklearn).
**Pourquoi stratifié classe+année** : les classes sont déséquilibrées et 2025≠2026 en
distribution ; on veut les 4 classes et les 2 années représentées dans chaque split.
**Pourquoi pas un split aléatoire simple** : fuite T/B + risque qu'une classe rare
manque dans le test. **Splits figés** (random_state + fichier CSV) pour comparer les
modèles à égalité.

## 4. Modèles : baseline CNN « maison » ➜ modèle dual-branch T/B

**Choix** : deux modèles comparés.
1. **CNN maison** (codé à la main, quelques couches conv) sur image unique — **baseline
   exigée par le sujet**, sert de référence basse.
2. **Dual-branch T/B** : deux extracteurs de features (poids **partagés** = siamois) sur
   T et B → fusion par **concaténation** → tête de classification → 4 classes.

**Pourquoi le dual-branch** : c'est l'architecture qui correspond au problème (2 vues →
1 décision) ; la fusion par concat est simple, robuste et éprouvée (réf. multi-view
apple grading). Poids partagés → moitié moins de paramètres et de VRAM, et régularise
(T et B ont la même texture de châtaigne).
**Pourquoi pas un seul gros réseau pré-entraîné sur image concaténée** : coller T et B
côte à côte dans une image casse l'invariance et gaspille du compute.
**Pourquoi pas de la détection d'objets (YOLO)** : on n'a pas à *localiser* la châtaigne
(elle est déjà centrée dans le slot), juste à *classer* → un classifieur suffit, plus
léger et plus rapide (crucial pour les 12 flux).

## 5. Backbone : MobileNetV3-Small (torchvision)

**Choix** : `mobilenet_v3_small` pré-entraîné ImageNet comme extracteur des branches.
**Pourquoi** : meilleur rapport précision/latence de sa catégorie, très léger → tient à
l'aise en 3 Go même en traitant les 24 images (2×12) d'un tick en un seul batch.
**Pourquoi pas ResNet50 / EfficientNet-B3** : plus précis dans l'absolu mais trop lourds
pour la GTX 1060 en temps réel ; le gain de précision n'est pas nécessaire vu la
simplicité visuelle des classes.
**Pourquoi pré-entraîné** : *transfer learning* → convergence rapide et meilleure
généralisation avec un dataset modeste, vs. entraîner de zéro.

## 6. Prétraitement : center-crop + masque circulaire + resize 224²

**Choix** : recadrage carré central, **masque circulaire** (hors-cercle noirci), resize
224×224, normalisation ImageNet.
**Pourquoi** : l'information utile est dans le **disque central** (le slot). Masquer le
fond force le réseau à ignorer le plateau/l'éclairage et réduit le sur-apprentissage.
224² = entrée standard des backbones pré-entraînés.
**Pourquoi pas l'image brute 720×480** : beaucoup de fond inutile, plus lourd, et le
modèle risquerait d'apprendre des artefacts de fond.

## 7. Augmentation : rotation 0–360°, flips, jitter lumière — PAS de random-crop

**Choix** : rotation aléatoire complète, flips H/V, léger jitter luminosité/contraste
(train uniquement).
**Pourquoi** : le disque est **invariant en rotation** → la rotation 360° est une
augmentation « gratuite » et très efficace (une châtaigne reste la même quel que soit
son angle). Le jitter simule les variations d'éclairage de la ligne.
**Pourquoi pas de random-crop / zoom agressif** : ça sortirait la châtaigne du cadre ou
couperait un défaut → on détruirait l'info au lieu de l'augmenter.

## 8. Déséquilibre des classes : loss pondérée

**Choix** : `CrossEntropyLoss(weight=…)` avec poids inversement proportionnels à la
fréquence des classes.
**Pourquoi** : Conforme et Vide dominent ; sans pondération le modèle « joue la
majorité » et néglige NON Conforme / PIETRA — exactement les classes à ne pas rater
pour la pureté du lot.
**Pourquoi pas un simple oversampling** : la pondération est plus simple, sans dupliquer
d'images ni gonfler le temps d'epoch ; on garde l'oversampling en variante si besoin.

## 9. Métrique de décision : seuil réglé sur la précision Conforme (pas l'argmax)

**Choix** : suivi du **rappel et de la précision *par classe*** (pas l'accuracy
globale) ; en production, on n'accepte « Conforme » que si la **confiance dépasse un
seuil** calibré pour atteindre **précision ≥ 95 %** tout en gardant **rappel ≥ 85 %**.
**Pourquoi** : l'accuracy globale est trompeuse en déséquilibre. Le cahier des charges
impose une contrainte *asymétrique* (laisser passer un mauvais fruit coûte plus cher que
d'écarter un bon) → on règle le **point de fonctionnement** sur la précision Conforme,
au lieu de prendre bêtement la classe la plus probable.
**Pourquoi pas l'argmax brut** : il optimise l'accuracy, pas la contrainte métier.

## 10. Suivi d'expériences : MLflow

**Choix** : tout run loggé dans MLflow (hyperparamètres, métriques par classe, matrice
de confusion, courbes, artefacts .onnx + latence).
**Pourquoi** : exigé par le sujet, et indispensable pour **comparer les modèles à
égalité** et tracer le meilleur. Reproductibilité + traçabilité.

## 11. Export & production : ONNX opset 17, batch dynamique, FP16

**Choix** : export ONNX (opset 17) avec **axe batch dynamique**, vérification de
l'équivalence PyTorch↔ONNX, mesure de latence, FP16.
**Pourquoi batch dynamique** : on empile les **24 images (2 vues × 12 flux)** d'un tick
en **un seul appel GPU** → bien plus efficace que 12 inférences séparées.
**Pourquoi ONNX** : format d'inférence portable et rapide (onnxruntime), indépendant de
PyTorch pour la cible Linux de production.
**Pourquoi FP16** : sur GTX 1060 (3 Go), gain surtout **mémoire** (critique) ; on vérifie
que la précision de classification n'en souffre pas.

## 12. Pourquoi PyTorch « nu » et pas Lightning/Hydra

**Choix** : boucle d'entraînement PyTorch explicite + configs simples, sans
PyTorch-Lightning ni Hydra.
**Pourquoi** : le sujet demande un **CNN codé à la main** et la *compréhension* du
pipeline ; une boucle explicite est plus lisible et défendable en soutenance, avec moins
de dépendances (utile pour l'export et la cible modeste). Lightning/Hydra seraient un
plus « industriel » mais masqueraient la mécanique qu'on doit justement démontrer.

---

## Récapitulatif — traçabilité vers la grille de notation

| Compétence | Où c'est traité |
|---|---|
| **C28** (problématique data) | rapport qualité §1–6, déséquilibre, T/B, hétérogénéité diagnostic |
| **C29** (modèles & métriques) | §4 modèles, §8 loss, §9 métriques/seuil, comparatif MLflow |
| **C30** (exploitation/optim données) | §6 prétraitement, §7 augmentation, §11 export/latence |
| **C31** (organisation données) | unité châtaigne §1, agrégation §2, splits §3, dépôt Git |
| **C32** (conception programme IA) | architecture dual-branch §4–5, pipeline train/eval/export |
