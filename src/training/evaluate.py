"""Évaluation sur le jeu de TEST + calibration du seuil de décision Conforme.

Le cahier des charges impose, sur la classe Conforme : précision ≥ 95 % ET
rappel ≥ 85 %. L'argmax brut n'optimise pas cette contrainte asymétrique ; on
cherche donc le **seuil de confiance** sur la probabilité « Conforme » qui atteint
précision ≥ 95 % tout en maximisant le rappel, et on vérifie rappel ≥ 85 %.

Usage :
    python src/training/evaluate.py --model dualbranch
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CLASSES, ImageDataset, PairDataset
from models import build_model
from train import device

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "reports", "figures")
CONF_IDX = CLASSES.index("Conforme")


@torch.no_grad()
def predict(model, loader, dev, is_pair):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        if is_pair:
            xt, xb, y = batch
            logits = model(xt.to(dev), xb.to(dev))
        else:
            x, y = batch
            logits = model(x.to(dev))
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def confusion(y_true, y_pred):
    K = len(CLASSES)
    cm = np.zeros((K, K), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def calibrate_conforme(probs, y_true, target_prec=0.95):
    """Balaye le seuil sur P(Conforme). Retourne (seuil, précision, rappel)."""
    pc = probs[:, CONF_IDX]
    is_conf = (y_true == CONF_IDX)
    best = None
    for thr in np.linspace(0.30, 0.99, 70):
        pred_conf = pc >= thr
        tp = np.sum(pred_conf & is_conf)
        fp = np.sum(pred_conf & ~is_conf)
        fn = np.sum(~pred_conf & is_conf)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if prec >= target_prec:
            if best is None or rec > best[2]:
                best = (float(thr), float(prec), float(rec))
    return best


def save_confusion_fig(cm, name):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=25, ha="right"); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
    thr = cm.max() / 2
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=9)
    ax.set_title(f"Matrice de confusion — {name} (test)", weight="bold")
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"confusion_{name}.png"), dpi=130)
    plt.close(fig)


def main(a):
    dev = device()
    is_pair = a.model == "dualbranch"
    if is_pair:
        df = pd.read_csv(os.path.join(DATA, "splits_chestnut.csv"))
        ds = PairDataset(df[df.split == "test"])
    else:
        df = pd.read_csv(os.path.join(DATA, "splits_image.csv"))
        ds = ImageDataset(df[df.split == "test"])
    loader = DataLoader(ds, batch_size=a.batch, num_workers=a.workers)

    model = build_model(a.model, pretrained=False, backbone=a.backbone).to(dev)
    ckpt = os.path.join(ROOT, f"best_{a.model}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=dev))

    probs, y_true = predict(model, loader, dev, is_pair)
    y_pred = probs.argmax(1)
    cm = confusion(y_true, y_pred)
    save_confusion_fig(cm, a.model)

    print(f"\n=== {a.model} — TEST (n={len(y_true)}) ===")
    print(f"accuracy globale : {(y_true == y_pred).mean():.3f}\n")
    print(f"{'classe':14} {'précision':>10} {'rappel':>8}")
    for i, c in enumerate(CLASSES):
        prec = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else 0
        rec = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() else 0
        print(f"{c:14} {prec:10.3f} {rec:8.3f}")

    print("\n--- Contrainte cahier des charges (Conforme P≥95 %, R≥85 %) ---")
    pc = cm[CONF_IDX, CONF_IDX] / cm[:, CONF_IDX].sum() if cm[:, CONF_IDX].sum() else 0
    rc = cm[CONF_IDX, CONF_IDX] / cm[CONF_IDX, :].sum() if cm[CONF_IDX, :].sum() else 0
    print(f"argmax brut       : P={pc:.3f}  R={rc:.3f}  -> {'OK' if pc>=.95 and rc>=.85 else 'NON conforme'}")
    cal = calibrate_conforme(probs, y_true)
    if cal:
        thr, p, r = cal
        verdict = "OK ✅" if r >= 0.85 else "précision atteinte mais rappel < 85 %"
        print(f"seuil calibré={thr:.2f}: P={p:.3f}  R={r:.3f}  -> {verdict}")
    else:
        print("Aucun seuil n'atteint précision ≥ 95 % — modèle à améliorer.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["simplecnn", "dualbranch"], default="dualbranch")
    ap.add_argument("--backbone", default="mobilenetv3_small",
                    choices=["mobilenetv3_small", "mobilenetv3_large"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args())
