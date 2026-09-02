"""Entraînement — baseline SimpleCNN (image) ou DualBranchNet (paire T/B).

Boucle PyTorch explicite (pas de Lightning : lisibilité + moins de dépendances,
cf. reports/choix_justifies.md §12). Suivi MLflow, loss pondérée par classe,
métriques par classe + matrice de confusion.

Exemples :
    python src/training/train.py --model dualbranch --epochs 8
    python src/training/train.py --model simplecnn  --epochs 8
    python src/training/train.py --model dualbranch --subset 800 --epochs 1   # sanity
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mlflow
from dataset import CLASSES, ImageDataset, PairDataset
from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def class_weights(labels):
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float32)
    w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def per_class_metrics(y_true, y_pred):
    """Retourne dict {classe: (precision, recall)} + matrice de confusion."""
    K = len(CLASSES)
    cm = np.zeros((K, K), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    out = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        out[c] = (prec, rec)
    return out, cm


def run_epoch(model, loader, dev, is_pair, crit=None, opt=None):
    train = opt is not None
    model.train(train)
    ys, ps, tot = [], [], 0.0
    for batch in loader:
        if is_pair:
            xt, xb, y = batch
            xt, xb, y = xt.to(dev), xb.to(dev), y.to(dev)
            logits = model(xt, xb)
        else:
            x, y = batch
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
        loss = crit(logits, y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * y.size(0)
        ys.append(y.cpu().numpy())
        ps.append(torch.softmax(logits, 1).detach().cpu().numpy())
    y_true = np.concatenate(ys); probs = np.concatenate(ps)
    y_pred = probs.argmax(1)
    acc = (y_true == y_pred).mean()
    return tot / len(y_true), acc, y_true, y_pred, probs


CONF_IDX = CLASSES.index("Conforme")


def conforme_recall_at_precision(probs, y_true, target_p=0.95):
    """Meilleur rappel Conforme atteignable avec précision >= target_p (0 si aucun)."""
    pc = probs[:, CONF_IDX]; is_c = (y_true == CONF_IDX)
    best_r = 0.0
    for thr in np.linspace(0.30, 0.99, 70):
        pred = pc >= thr
        tp = np.sum(pred & is_c); fp = np.sum(pred & ~is_c); fn = np.sum(~pred & is_c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        if prec >= target_p:
            best_r = max(best_r, rec)
    return best_r


def main(a):
    dev = device()
    is_pair = a.model == "dualbranch"
    ch = pd.read_csv(os.path.join(DATA, "splits_chestnut.csv"))
    im = pd.read_csv(os.path.join(DATA, "splits_image.csv"))

    if is_pair:
        tr, va = ch[ch.split == "train"], ch[ch.split == "val"]
        if a.drop_ambiguous:  # nettoie le TRAIN seulement (val/test restent représentatifs)
            n0 = len(tr)
            tr = tr[~(tr["multiple"] | tr["chunk"])]
            print(f"drop-ambiguous : train {n0} -> {len(tr)} (exclut chunk/multiple)")
        if a.subset:
            tr = tr.sample(min(a.subset, len(tr)), random_state=0)
        ds_tr = PairDataset(tr, train=True, img_size=a.img_size, polar=a.polar)
        ds_va = PairDataset(va, img_size=a.img_size, polar=a.polar)
        labels = tr["label_chestnut"].map({c: i for i, c in enumerate(CLASSES)}).values
    else:
        tr, va = im[im.split == "train"], im[im.split == "val"]
        if a.subset:
            tr = tr.sample(min(a.subset, len(tr)), random_state=0)
        ds_tr = ImageDataset(tr, train=True, img_size=a.img_size, polar=a.polar)
        ds_va = ImageDataset(va, img_size=a.img_size, polar=a.polar)
        labels = tr["label"].map({c: i for i, c in enumerate(CLASSES)}).values

    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=a.workers)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=a.workers)

    model = build_model(a.model, backbone=a.backbone, fusion=a.fusion).to(dev)
    crit = nn.CrossEntropyLoss(weight=class_weights(labels).to(dev),
                               label_smoothing=a.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    mlflow.set_experiment("castagnet")
    with mlflow.start_run(run_name=f"{a.model}_{a.backbone}"):
        mlflow.log_params({"model": a.model, "backbone": a.backbone,
                           "epochs": a.epochs, "batch": a.batch, "lr": a.lr,
                           "scheduler": "cosine", "label_smoothing": a.label_smoothing,
                           "img_size": a.img_size, "drop_ambiguous": a.drop_ambiguous,
                           "fusion": a.fusion, "polar": a.polar,
                           "device": str(dev), "n_train": len(tr), "subset": a.subset or 0})
        best = -1.0
        for ep in range(1, a.epochs + 1):
            trl, tra, *_ = run_epoch(model, dl_tr, dev, is_pair, crit, opt)
            vl, vacc, yt, yp, vprobs = run_epoch(model, dl_va, dev, is_pair, crit)
            sched.step()
            met, cm = per_class_metrics(yt, yp)
            pc, rc = met["Conforme"]
            # objectif métier : rappel Conforme atteignable à précision >= 95 %
            r_at_p95 = conforme_recall_at_precision(vprobs, yt)
            mlflow.log_metrics({"train_loss": trl, "train_acc": tra,
                                "val_loss": vl, "val_acc": vacc,
                                "conforme_precision": pc, "conforme_recall": rc,
                                "conforme_recall_at_p95": r_at_p95,
                                "lr": sched.get_last_lr()[0]}, step=ep)
            print(f"ep{ep}: val_acc={vacc:.3f} | Conforme R@P95={r_at_p95:.3f} "
                  f"(argmax P={pc:.3f} R={rc:.3f})")
            # on sauve le modèle qui maximise l'objectif métier
            if r_at_p95 > best:
                best = r_at_p95
                torch.save(model.state_dict(),
                           os.path.join(ROOT, f"best_{a.model}{a.tag}.pt"))
        # matrice de confusion finale
        print("\nMatrice de confusion (val) — lignes=vrai, colonnes=prédit :")
        print("        " + "  ".join(f"{c[:6]:>6}" for c in CLASSES))
        for i, c in enumerate(CLASSES):
            print(f"{c[:8]:>8} " + "  ".join(f"{cm[i,j]:6d}" for j in range(len(CLASSES))))
        mlflow.log_metric("best_conforme_recall_at_p95", best)
    print(f"\nMeilleur rappel Conforme @P95 (val) : {best:.3f}  "
          f"(modèle sauvé best_{a.model}{a.tag}.pt)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["simplecnn", "dualbranch"], default="dualbranch")
    ap.add_argument("--backbone", default="mobilenetv3_small",
                    choices=["mobilenetv3_small", "mobilenetv3_large"])
    ap.add_argument("--fusion", default="concat",
                    choices=["concat", "sum", "concat_diff"])
    ap.add_argument("--tag", default="", help="suffixe du checkpoint (évite les collisions)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05, dest="label_smoothing")
    ap.add_argument("--img-size", type=int, default=224, dest="img_size")
    ap.add_argument("--polar", action="store_true",
                    help="représentation radiale (déroulé polaire du disque)")
    ap.add_argument("--drop-ambiguous", action="store_true", dest="drop_ambiguous",
                    help="exclut chunk/multiple du TRAIN (val/test intacts)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--subset", type=int, default=0, help="limiter le train (sanity)")
    main(ap.parse_args())
