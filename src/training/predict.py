"""Inférence sur UNE châtaigne (ses 2 vues T et B) avec le modèle ONNX.

Utilise onnxruntime uniquement (pas besoin de PyTorch) — c'est le mode production.

Exemples :
    # à partir des deux images d'une même châtaigne
    python src/training/predict.py --t chemin/vue_T.jpg --b chemin/vue_B.jpg

    # ou en donnant juste une châtaigne du dataset (pairkey) : retrouve T et B
    python src/training/predict.py --pairkey 2025_Conforme_1_Cam_X_1_0.jpg
"""

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGES_DIR = os.path.join(ROOT, "..", "castagnia_data-main", "images")
ONNX = os.path.join(ROOT, "model_dualbranch.onnx")

CLASSES = ["Conforme", "NON Conforme", "PIETRA", "Vide"]
CONF_IDX = 0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# seuil calibré : on n'accepte "Conforme" que si sa proba dépasse ce seuil
# (garantit la précision >= 95 % exigée par le cahier des charges)
SEUIL_CONFORME = 0.64


def preprocess(path, size=224, r_ratio=0.49):
    """Même prétraitement qu'à l'entraînement : center-crop + masque circulaire."""
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((size, size, 3), np.uint8)
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = cv2.resize(img, (size, size))
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * r_ratio), 255, -1)
    img = cv2.bitwise_and(img, img, mask=mask)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return rgb.transpose(2, 0, 1)[None]  # (1,3,H,W)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(t_path, b_path, tta=4):
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    xt, xb = preprocess(t_path), preprocess(b_path)
    # TTA simple : on moyenne la vue nette + quelques rotations
    probs = np.zeros(len(CLASSES), np.float32)
    for k in range(max(1, tta)):
        if k == 0:
            it, ib = xt, xb
        else:
            ang = 90 * k
            M = cv2.getRotationMatrix2D((112, 112), ang, 1.0)
            it = np.stack([cv2.warpAffine(xt[0, c], M, (224, 224)) for c in range(3)])[None]
            ib = np.stack([cv2.warpAffine(xb[0, c], M, (224, 224)) for c in range(3)])[None]
        logits = sess.run(None, {"view_t": it, "view_b": ib})[0][0]
        probs += softmax(logits)
    probs /= max(1, tta)

    # décision : argmax, MAIS "Conforme" seulement si sûr (seuil)
    if probs[CONF_IDX] >= SEUIL_CONFORME:
        decision = "Conforme"
    else:
        # sinon on prend la meilleure classe hors Conforme
        order = np.argsort(-probs)
        decision = CLASSES[order[0]] if order[0] != CONF_IDX else CLASSES[order[1]]
    return decision, probs


def main(a):
    if a.pairkey:
        t = os.path.join(IMAGES_DIR, a.pairkey.replace("_Cam_X_", "_Cam_T_"))
        b = os.path.join(IMAGES_DIR, a.pairkey.replace("_Cam_X_", "_Cam_B_"))
    else:
        t, b = a.t, a.b
    decision, probs = predict(t, b, tta=a.tta)
    print(f"Vue T : {os.path.basename(t)}")
    print(f"Vue B : {os.path.basename(b)}")
    print("\nProbabilités :")
    for c, p in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
        print(f"  {c:14} {p:6.1%}")
    print(f"\n=> DÉCISION : {decision}")
    if decision == "Conforme":
        print(f"   (Conforme accepté : proba {probs[CONF_IDX]:.1%} >= seuil {SEUIL_CONFORME:.0%})")
    else:
        print(f"   (Conforme écarté : proba {probs[CONF_IDX]:.1%} < seuil {SEUIL_CONFORME:.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", help="chemin vue T (dessus)")
    ap.add_argument("--b", help="chemin vue B (dessous)")
    ap.add_argument("--pairkey", help="nom de fichier avec _Cam_X_ (retrouve T et B)")
    ap.add_argument("--tta", type=int, default=4)
    main(ap.parse_args())
