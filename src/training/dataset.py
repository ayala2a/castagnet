"""Datasets PyTorch — image unique (baseline) et paire T/B (dual-branch).

Prétraitement commun : center-crop carré + masque circulaire (le fond hors du
slot est noirci) + resize 224 + normalisation ImageNet.
Augmentation (train) : rotation 0-360°, flips, jitter luminosité — cf. rapport
`reports/choix_justifies.md` §6-7.
"""

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(ROOT, "..", "castagnia_data-main", "images")

CLASSES = ["Conforme", "NON Conforme", "PIETRA", "Vide"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _circular_preprocess(bgr, size=224, r_ratio=0.49):
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sq = bgr[y0:y0 + s, x0:x0 + s]
    sq = cv2.resize(sq, (size, size))
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * r_ratio), 255, -1)
    return cv2.bitwise_and(sq, sq, mask=mask)


def _augment(bgr):
    # rotation 0-360° (le disque est invariant en rotation)
    ang = np.random.uniform(0, 360)
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    bgr = cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if np.random.rand() < 0.5:
        bgr = cv2.flip(bgr, 1)
    if np.random.rand() < 0.5:
        bgr = cv2.flip(bgr, 0)
    # jitter luminosité/contraste léger
    alpha = np.random.uniform(0.85, 1.15)   # contraste
    beta = np.random.uniform(-15, 15)       # luminosité
    return cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)


def _to_tensor(bgr, train):
    if train:
        bgr = _augment(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


def _load(filename, size=224):
    path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(path)
    if img is None:
        return np.zeros((size, size, 3), np.uint8)
    return _circular_preprocess(img, size=size)


class ImageDataset(Dataset):
    """Baseline mono-vue : renvoie (image, label)."""

    def __init__(self, df, train=False, img_size=224):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = _to_tensor(_load(r["filename"], self.img_size), self.train)
        return x, CLASS_TO_IDX[r["label"]]


class PairDataset(Dataset):
    """Dual-branch : renvoie (image_T, image_B, label).

    Vue manquante (orphelin) -> image noire (le masque circulaire fait déjà du
    hors-cercle un fond noir, donc une vue absente = 'rien vu de ce côté').
    """

    def __init__(self, df, train=False, img_size=224):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def _view(self, filename):
        if isinstance(filename, str) and filename:
            return _to_tensor(_load(filename, self.img_size), self.train)
        return torch.zeros(3, self.img_size, self.img_size)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        xt = self._view(r["T_filename"])
        xb = self._view(r["B_filename"])
        return xt, xb, CLASS_TO_IDX[r["label_chestnut"]]
