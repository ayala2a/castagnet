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
        ys.append(y.cpu().numpy()); ps.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    acc = (y_true == y_pred).mean()
    return tot / len(y_true), acc, y_true, y_pred


def main(a):
    dev = device()
    is_pair = a.model == "dualbranch"
    ch = pd.read_csv(os.path.join(DATA, "splits_chestnut.csv"))
    im = pd.read_csv(os.path.join(DATA, "splits_image.csv"))

    if is_pair:
        tr, va = ch[ch.split == "train"], ch[ch.split == "val"]
        if a.subset:
            tr = tr.sample(min(a.subset, len(tr)), random_state=0)
        ds_tr, ds_va = PairDataset(tr, train=True), PairDataset(va)
        labels = tr["label_chestnut"].map({c: i for i, c in enumerate(CLASSES)}).values
    else:
        tr, va = im[im.split == "train"], im[im.split == "val"]
        if a.subset:
            tr = tr.sample(min(a.subset, len(tr)), random_state=0)
        ds_tr, ds_va = ImageDataset(tr, train=True), ImageDataset(va)
        labels = tr["label"].map({c: i for i, c in enumerate(CLASSES)}).values

    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, num_workers=a.workers)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=a.workers)

    model = build_model(a.model).to(dev)
    crit = nn.CrossEntropyLoss(weight=class_weights(labels).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    mlflow.set_experiment("castagnet")
    with mlflow.start_run(run_name=a.model):
        mlflow.log_params({"model": a.model, "epochs": a.epochs, "batch": a.batch,
                           "lr": a.lr, "device": str(dev),
                           "n_train": len(tr), "subset": a.subset or 0})
        best = 0.0
        for ep in range(1, a.epochs + 1):
            trl, tra, *_ = run_epoch(model, dl_tr, dev, is_pair, crit, opt)
            vl, vacc, yt, yp = run_epoch(model, dl_va, dev, is_pair, crit)
            met, cm = per_class_metrics(yt, yp)
            pc, rc = met["Conforme"]
            mlflow.log_metrics({"train_loss": trl, "train_acc": tra,
                                "val_loss": vl, "val_acc": vacc,
                                "conforme_precision": pc, "conforme_recall": rc}, step=ep)
            print(f"ep{ep}: train_acc={tra:.3f} val_acc={vacc:.3f} | "
                  f"Conforme P={pc:.3f} R={rc:.3f}")
            if vacc > best:
                best = vacc
                torch.save(model.state_dict(),
                           os.path.join(ROOT, f"best_{a.model}.pt"))
        # matrice de confusion finale
        print("\nMatrice de confusion (val) — lignes=vrai, colonnes=prédit :")
        print("        " + "  ".join(f"{c[:6]:>6}" for c in CLASSES))
        for i, c in enumerate(CLASSES):
            print(f"{c[:8]:>8} " + "  ".join(f"{cm[i,j]:6d}" for j in range(len(CLASSES))))
        mlflow.log_metric("best_val_acc", best)
    print(f"\nMeilleure val_acc : {best:.3f}  (modèle sauvé best_{a.model}.pt)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["simplecnn", "dualbranch"], default="dualbranch")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--subset", type=int, default=0, help="limiter le train (sanity)")
    main(ap.parse_args())
