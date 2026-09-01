"""Extraction d'images de châtaignes depuis les vidéos brutes de la machine (§4.1).

Chaque vidéo = flux d'une caméra filmant un créneau circulaire où défilent les
châtaignes. Pipeline :
  1. lecture des frames (cv2)
  2. détection du créneau circulaire (HoughCircles, position ~fixe par vidéo)
  3. score de présence (le slot est-il rempli par une châtaigne ?)
  4. crop carré autour du cercle + masque circulaire (style dataset)
  5. déduplication temporelle (une châtaigne occupe plusieurs frames consécutives)

Sorties :
  data/video_frames/<video>/crop_<frame>.jpg   crops retenus
  data/video_frames/<video>_contact.png        planche-contact de contrôle
"""

import argparse
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO_DIR = os.path.join(ROOT, "..")  # marron/
OUT = os.path.join(ROOT, "data", "video_frames")


def detect_circle(gray):
    """Détecte le créneau circulaire dominant. Retourne (x, y, r) ou None."""
    g = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
        param1=100, param2=45, minRadius=70, maxRadius=200,
    )
    if circles is None:
        return None
    x, y, r = np.round(circles[0][0]).astype(int)
    return int(x), int(y), int(r)


def circular_crop(img, x, y, r, pad=1.15):
    """Crop carré centré sur le cercle + masque circulaire (hors-cercle noirci)."""
    R = int(r * pad)
    h, w = img.shape[:2]
    x0, x1 = max(0, x - R), min(w, x + R)
    y0, y1 = max(0, y - R), min(h, y + R)
    crop = img[y0:y1, x0:x1].copy()
    ch, cw = crop.shape[:2]
    mask = np.zeros((ch, cw), np.uint8)
    cv2.circle(mask, (cw // 2, ch // 2), min(ch, cw) // 2, 255, -1)
    return cv2.bitwise_and(crop, crop, mask=cv2.cvtColor(
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY))


def fill_score(gray, x, y, r):
    """Fraction de pixels 'clairs' (châtaigne) dans le disque intérieur.

    Un slot vide est sombre (trou) ; une châtaigne renvoie la lumière -> plus clair.
    """
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x, y), int(r * 0.75), 255, -1)
    inside = gray[mask == 255]
    if inside.size == 0:
        return 0.0
    return float((inside > 90).mean())


def extract(video_path, out_dir, step=2, fill_thr=0.20, dedup_gap=8, max_crops=60):
    name = os.path.splitext(os.path.basename(video_path))[0].replace(" ", "")
    vdir = os.path.join(out_dir, name)
    os.makedirs(vdir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kept, last_kept = [], -999
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = detect_circle(gray)
        if c is None:
            continue
        x, y, r = c
        score = fill_score(gray, x, y, r)
        if score < fill_thr:
            continue  # slot vide
        if i - last_kept < dedup_gap:
            continue  # même châtaigne, frame trop proche
        crop = circular_crop(img, x, y, r)
        path = os.path.join(vdir, f"crop_{i:04d}.jpg")
        cv2.imwrite(path, crop)
        kept.append((i, score, crop))
        last_kept = i
        if len(kept) >= max_crops:
            break
    cap.release()

    # planche-contact
    if kept:
        thumbs = [cv2.resize(c, (120, 120)) for _, _, c in kept]
        cols = 10
        rows = (len(thumbs) + cols - 1) // cols
        sheet = np.zeros((rows * 122, cols * 122, 3), np.uint8)
        for k, t in enumerate(thumbs):
            rr, cc = divmod(k, cols)
            sheet[rr * 122:rr * 122 + 120, cc * 122:cc * 122 + 120] = t
        cv2.imwrite(os.path.join(out_dir, f"{name}_contact.png"), sheet)
    print(f"{name}: {len(kept)} crops retenus (sur {n} frames, pas={step})")
    return kept


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*",
                    default=["attachment.avi", "attachment (1).avi"])
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--fill-thr", type=float, default=0.20)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for v in args.videos:
        extract(os.path.join(VIDEO_DIR, v), OUT, step=args.step, fill_thr=args.fill_thr)
