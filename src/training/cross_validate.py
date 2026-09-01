"""Validation croisée du modèle final (dual-branch, fusion |T-B|).

Objectif : prouver que le résultat est STABLE, pas un coup de chance sur un seul
split. On fait un StratifiedGroupKFold (groupé par châtaigne = aucune fuite T/B,
stratifié par classe) ; pour chaque fold on entraîne, on sélectionne le meilleur
epoch sur une val interne, on calibre le seuil Conforme sur cette val, puis on
mesure précision/rappel Conforme sur le fold de test tenu à l'écart.

N'ÉCRIT QUE dans best_dualbranch_cv_fold{k}.pt — ne touche jamais au modèle final
ni à l'ONNX.

Usage :
    python src/training/cross_validate.py --folds 5 --epochs 12
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader

from dataset import CLASSES, PairDataset
from models import build_model
from train import CONF_IDX, class_weights, conforme_recall_at_precision, device, run_epoch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
SEED = 42
IDX = {c: i for i, c in enumerate(CLASSES)}


def conforme_at_threshold(probs, y_true, thr):
    """Précision/rappel de Conforme si on n'accepte 'Conforme' que si p>=thr."""
    pc = probs[:, CONF_IDX]; is_c = (y_true == CONF_IDX)
    pred = pc >= thr
    tp = np.sum(pred & is_c); fp = np.sum(pred & ~is_c); fn = np.sum(~pred & is_c)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec


def best_threshold_for_precision(probs, y_true, target_p=0.95):
    """Seuil sur P(Conforme) qui atteint precision>=target_p en maximisant le rappel."""
    pc = probs[:, CONF_IDX]; is_c = (y_true == CONF_IDX)
    best = (0.5, 0.0)
    for thr in np.linspace(0.30, 0.99, 70):
        prec, rec = conforme_at_threshold(probs, y_true, thr)
        if prec >= target_p and rec > best[1]:
            best = (float(thr), rec)
    return best[0]


def train_one_fold(tr, va, dev, a):
    ds_tr = PairDataset(tr, train=True, img_size=a.img_size)
    ds_va = PairDataset(va, img_size=a.img_size)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=a.workers)
    dl_va = DataLoader(ds_va, batch_size=a.batch, num_workers=a.workers)

    model = build_model("dualbranch", backbone=a.backbone, fusion=a.fusion).to(dev)
    labels = tr["label_chestnut"].map(IDX).values
    crit = nn.CrossEntropyLoss(weight=class_weights(labels).to(dev), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best_r, best_state = -1.0, None
    for ep in range(1, a.epochs + 1):
        run_epoch(model, dl_tr, dev, True, crit, opt)
        _, _, yt, _, vprobs = run_epoch(model, dl_va, dev, True, crit)
        sched.step()
        r = conforme_recall_at_precision(vprobs, yt)
        if r > best_r:
            best_r = r
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, dl_va


def main(a):
    dev = device()
    df = pd.read_csv(os.path.join(DATA, "labels_chestnut.csv"))
    df["strat"] = df["label_chestnut"] + "_" + df["year"].astype(str)
    y = df["label_chestnut"].map(IDX).values
    groups = df["pairkey"].values

    skf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=SEED)
    rows = []
    for k, (tv_idx, te_idx) in enumerate(skf.split(df, y, groups), 1):
        tv, te = df.iloc[tv_idx], df.iloc[te_idx]
        tr, va = train_test_split(tv, test_size=0.15, stratify=tv["strat"], random_state=SEED)

        model, _ = train_one_fold(tr, va, dev, a)

        # calibrer le seuil sur la val interne
        dl_va = DataLoader(PairDataset(va, img_size=a.img_size), batch_size=a.batch, num_workers=a.workers)
        _, _, yv, _, vprobs = run_epoch(model, dl_va, dev, True, nn.CrossEntropyLoss())
        thr = best_threshold_for_precision(vprobs, yv)

        # appliquer au fold de TEST tenu à l'écart
        dl_te = DataLoader(PairDataset(te, img_size=a.img_size), batch_size=a.batch, num_workers=a.workers)
        _, acc, yt, yp, tprobs = run_epoch(model, dl_te, dev, True, nn.CrossEntropyLoss())
        prec, rec = conforme_at_threshold(tprobs, yt, thr)
        acc = float((yt == yp).mean())
        rows.append({"fold": k, "seuil": thr, "conforme_P": prec, "conforme_R": rec, "acc": acc})
        print(f"fold {k}: seuil={thr:.2f}  Conforme P={prec:.3f} R={rec:.3f}  acc={acc:.3f}")
        torch.save(model.state_dict(), os.path.join(ROOT, f"best_dualbranch_cv_fold{k}.pt"))

    res = pd.DataFrame(rows)
    print("\n=== Validation croisée (moyenne ± écart-type sur", a.folds, "folds) ===")
    for col, name in [("conforme_P", "Conforme précision"), ("conforme_R", "Conforme rappel"), ("acc", "accuracy")]:
        print(f"  {name:20} {res[col].mean():.3f} ± {res[col].std():.3f}")
    res.to_csv(os.path.join(ROOT, "reports", "cross_validation.csv"), index=False)
    print("\n-> détail dans reports/cross_validation.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--backbone", default="mobilenetv3_large")
    ap.add_argument("--fusion", default="concat_diff")
    ap.add_argument("--img-size", type=int, default=224, dest="img_size")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=0)
    main(ap.parse_args())
