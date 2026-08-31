# Références & choix techniques (recherche repos pro)

Synthèse des repos/patterns de référence retenus pour les §4.2/§4.3. Repos vérifiés
(README lus) sauf mention contraire.

## Structure projet (PyTorch + MLflow)
- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) (~5.3k★) — la référence : `configs/` (Hydra) + `src/` + loggers interchangeables (MLflow inclus).
- [hppRC/template-pytorch-lightning-hydra-mlflow-poetry](https://github.com/hppRC/template-pytorch-lightning-hydra-mlflow-poetry) — MLflow déjà câblé, minimaliste → point de départ.
- **Décision** : configs YAML (pas de constantes en dur), `MLFlowLogger`, log par run des métriques **par classe** (recall/precision Conforme surtout) + matrice de confusion + `.onnx` + latence en artifacts.

## Modèle 2-vues (T/B → 1 label) — cœur technique
- Stratégie retenue : **late fusion / feature-concat** avec **backbones à poids partagés** (siamois).
- Réf. applicative : *Enhancing Apple Cultivar Classification Using Multiview Images* (MDPI J. Imaging 2024) — 1 modèle/vue + fusion.
- Package utile pour benchmarker les fusions : [florencejt/fusilli](https://github.com/florencejt/fusilli) (concat/attention/late).
- Squelette `DualBranchNet` (timm, `num_classes=0` → features, `torch.cat` → tête MLP) — voir `src/training/models.py` (à venir).

## Analogues domaine (tri graines/fruits)
- **SeedSortNet** (PMC8356658) — CNN léger à attention **dédié tri de graines**, le plus proche.
- *Multi-Camera Sorting for Apple Surface Defects* (PMC10141532) — 3 caméras, **0,069 s/pomme, 93,8 %** → preuve temps-réel multi-vues sur matériel modeste.
- [veronicamorelli/Rice-Grain-Image-Classification](https://github.com/veronicamorelli/Rice-Grain-Image-Classification) — démarche custom-CNN vs transfer (Keras, pour la méthode).
- Enseignement : loss **pondérée par classe** (déséquilibre), suivi **recall par classe** ≠ accuracy.

## Backbones légers (cible GTX 1060 3 Go) — via [timm](https://github.com/huggingface/pytorch-image-models)
- Défaut : **`mobilenetv3_small_100`** (poids partagés → ~½ VRAM). Alternatives : `tf_efficientnet_lite0` (export ONNX/TRT propre), `mobilenetv4_conv_small`.

## ROI circulaire centrale (le disque utile)
- Center-crop 480² → resize 224² ; **masque circulaire** (cv2.circle) pour zéro-er le fond.
- Augmentations : **rotation 0–360°** (disque invariant → gros gain), flips, jitter brightness/contrast léger. **Pas** de random-crop (sortirait la ROI).

## Anti-leakage split (plusieurs images / même fruit)
- **`StratifiedGroupKFold`** (sklearn) : `groups = pairkey` → jamais le même fruit dans train & val, classes stratifiées.
- Modéliser **1 fruit = 1 échantillon (paire T+B)** fait disparaître le leak en amont. Figer les folds (artifact MLflow).

## Export ONNX + latence (§4.3)
- `torch.onnx.export(..., opset_version=17, dynamic_axes={axe 0 = batch})` → **batcher les 24 images/tick** (2 vues × 12 flux) en 1 appel GPU.
- Vérif équivalence torch↔onnx (`assert_allclose rtol=1e-3`), `onnx-simplifier`, bench `onnxruntime-gpu` providers **TensorRT→CUDA→CPU**, FP16, warmup + p50/p95.
- GTX 1060 = Pascal (pas de Tensor Cores) : FP16 surtout gain **mémoire** (3 Go critique) ; vitesse via TRT-EP + simplification.

## Stack cible (Python 3.13)
```
torch>=2.5 torchvision>=0.20 · timm>=1.0.29 · pytorch-lightning>=2.4
hydra-core>=1.3 omegaconf>=2.3 · mlflow>=2.16
onnx>=1.16 onnxruntime(-gpu)>=1.19 onnx-simplifier · scikit-learn>=1.5
opencv-python albumentations>=1.4
```
> Note : `pytorch-lightning`, `timm`, `hydra-core`, `albumentations` ne sont pas
> encore dans le venv (ajout à l'étape §4.2). Le reste est déjà installé.
