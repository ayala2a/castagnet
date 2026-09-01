# Pipeline d'entraînement (§4.2 / §4.3)

```bash
source ../../.venv/bin/activate     # depuis castagnet/src/training

python make_splits.py                        # -> data/splits_{chestnut,image}.csv (anti-leak)
python train.py --model simplecnn  --epochs 15   # baseline CNN maison (image unique)
python train.py --model dualbranch --epochs 15   # modèle 2 vues T/B (siamois)
mlflow ui                                    # tableau de bord des expériences
```

Justification de tous les choix : `../../reports/choix_justifies.md`.
À venir : `evaluate.py` (calibration seuil Conforme P≥95 %/R≥85 %) et `export_onnx.py` (+ latence).
onnxscript  # export ONNX torch 2.13+
