"""Appariement T/B des deux vidéos de la machine (§4.1).

Les deux vidéos filment le MÊME créneau sous deux angles (une vue « haut », une
vue « bas »), avec un micro-décalage temporel : la vidéo B est en avance de
OFFSET frames sur la vidéo A (mesuré par corrélation des séquences d'arrivée des
châtaignes : même cadence de 27 frames, décalage médian de 5 frames = 0,2 s).

Pour chaque châtaigne détectée dans la vidéo A à la frame f, sa vue appariée est
dans la vidéo B à la frame f - OFFSET. On crope les deux et on écrit la table de
correspondance.

Sorties :
    data/video_frames/pairs/<id>_A.jpg  /  <id>_B.jpg
    data/video_pairs.csv        (id, frame_A, frame_B, crop_A, crop_B)
    data/video_frames/pairs_contact.png  (planche de contrôle)
"""

import argparse
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO_DIR = os.path.join(ROOT, "..")
OUT = os.path.join(ROOT, "data", "video_frames", "pairs")
VIDEO_A = "attachment.avi"       # vue 1
VIDEO_B = "attachment (1).avi"   # vue 2 (en avance de OFFSET frames)
OFFSET = 5                        # A = B + 5  (mesuré)


def tan_signal(path):
    """Fraction de pixels 'châtaigne' (beige/brun) par frame -> présence."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sig = []
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = ((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 32) &
             (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 60))
        sig.append(float(m.mean()))
    cap.release()
    return np.array(sig)


def detect_events(sig, thr_ratio=0.5, min_gap=10):
    thr = sig.min() + (sig.max() - sig.min()) * thr_ratio
    on = sig > thr
    ev, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if not v and start is not None:
            ev.append((start, i - 1)); start = None
    if start is not None:
        ev.append((start, len(on) - 1))
    merged = []
    for a, b in ev:
        if merged and a - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [int((a + b) / 2) for a, b in merged if b - a >= 1]


def detect_circle(gray):
    circles = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT,
                               dp=1.2, minDist=300, param1=100, param2=40,
                               minRadius=80, maxRadius=220)
    if circles is None:
        h, w = gray.shape[:2]
        return w // 2, h // 2, min(h, w) // 3
    return tuple(np.round(circles[0][0]).astype(int))


def presence(img, x, y, r):
    """Fraction de pixels 'châtaigne' (beige/brun) dans le disque intérieur."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (x, y), int(r * 0.85), 255, -1)
    tan = ((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 32) &
           (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 60) & (mask == 255))
    inside = (mask == 255).sum()
    return tan.sum() / inside if inside else 0.0


def crop_from(img, x, y, r):
    R = int(r * 1.15)
    h, w = img.shape[:2]
    crop = img[max(0, y - R):min(h, y + R), max(0, x - R):min(w, x + R)].copy()
    ch, cw = crop.shape[:2]
    if ch < 10 or cw < 10:
        return None
    m = np.zeros((ch, cw), np.uint8)
    cv2.circle(m, (cw // 2, ch // 2), min(ch, cw) // 2, 255, -1)
    return cv2.bitwise_and(crop, crop, mask=m)


def best_in_window(path, center, w=3):
    """Dans [center-w, center+w], renvoie (frame, crop, score) où le fruit est le
    plus présent dans le créneau — évite les vues vides."""
    cap = cv2.VideoCapture(path)
    best = (center, None, -1.0)
    for f in range(max(0, center - w), center + w + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        x, y, r = detect_circle(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        s = presence(img, x, y, r)
        if s > best[2]:
            best = (f, crop_from(img, x, y, r), s)
    cap.release()
    return best


def main(a):
    os.makedirs(OUT, exist_ok=True)
    sigA = tan_signal(os.path.join(VIDEO_DIR, VIDEO_A))
    events = detect_events(sigA)
    print(f"{len(events)} châtaignes détectées dans la vidéo A")

    rows, thumbs = [], []
    for i, fa0 in enumerate(events):
        fb0 = fa0 - a.offset
        if fb0 < 0:
            continue
        # meilleure vue du fruit dans chaque vidéo, dans une fenêtre ±w
        fa, cropA, sa = best_in_window(os.path.join(VIDEO_DIR, VIDEO_A), fa0, a.window)
        fb, cropB, sb = best_in_window(os.path.join(VIDEO_DIR, VIDEO_B), fb0, a.window)
        if cropA is None or cropB is None:
            continue
        # on ne garde que les paires où le fruit est présent des DEUX côtés
        if min(sa, sb) < a.min_presence:
            continue
        pid = f"vid_{i:03d}"
        pa, pb = f"{pid}_A.jpg", f"{pid}_B.jpg"
        cv2.imwrite(os.path.join(OUT, pa), cropA)
        cv2.imwrite(os.path.join(OUT, pb), cropB)
        rows.append((pid, fa, fb, pa, pb))
        thumbs.append((cv2.resize(cropA, (120, 120)), cv2.resize(cropB, (120, 120))))

    # table de correspondance
    import csv
    with open(os.path.join(ROOT, "data", "video_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "frame_A", "frame_B", "crop_A", "crop_B"])
        w.writerows(rows)

    # planche de contrôle : A au-dessus, B en-dessous, par colonne
    if thumbs:
        cols = min(len(thumbs), 12)
        sheet = np.zeros((2 * 122 + 10, cols * 122, 3), np.uint8)
        for k, (ta, tb) in enumerate(thumbs[:cols]):
            sheet[0:120, k * 122:k * 122 + 120] = ta
            sheet[132:252, k * 122:k * 122 + 120] = tb
        cv2.imwrite(os.path.join(ROOT, "data", "video_frames", "pairs_contact.png"), sheet)

    print(f"{len(rows)} paires T/B écrites -> data/video_pairs.csv")
    print(f"décalage utilisé : A = B + {a.offset} frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=OFFSET)
    ap.add_argument("--window", type=int, default=3, help="fenêtre de recherche ±frames")
    ap.add_argument("--min-presence", type=float, default=0.04, dest="min_presence",
                    help="présence minimale du fruit dans les 2 vues pour garder la paire")
    main(ap.parse_args())
