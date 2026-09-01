"""Export ONNX + vérification + mesure de latence (§4.3).

- Export opset 17, axe batch dynamique -> permet d'empiler les 24 images d'un tick
  (2 vues × 12 flux) en un seul appel.
- Vérifie l'équivalence PyTorch ↔ ONNX (assert_allclose).
- Mesure la latence onnxruntime à plusieurs tailles de batch et discute la
  compatibilité avec les 12 flux (cadence 100 kg/h).

Usage :
    python src/training/export_onnx.py --model dualbranch
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import torch

from models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def export(model_name, out_path):
    model = build_model(model_name, pretrained=False)
    ckpt = os.path.join(ROOT, f"best_{model_name}.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    if model_name == "dualbranch":
        dummy = (torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224))
        names = ["view_t", "view_b"]
        dyn = {"view_t": {0: "batch"}, "view_b": {0: "batch"}, "logits": {0: "batch"}}
    else:
        dummy = (torch.randn(1, 3, 224, 224),)
        names = ["image"]
        dyn = {"image": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model, dummy, out_path, input_names=names, output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamic_axes=dyn,
    )
    print(f"Export ONNX -> {out_path}")
    return model, dummy, names


def verify(model, dummy, names, onnx_path):
    with torch.no_grad():
        torch_out = model(*dummy).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feeds = {n: d.numpy() for n, d in zip(names, dummy)}
    onnx_out = sess.run(None, feeds)[0]
    np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-3)
    print("Équivalence PyTorch ↔ ONNX : OK (rtol=1e-3)")
    return sess


def bench(sess, names, batches=(1, 12, 24), n=100, warm=10):
    print("\n=== Latence onnxruntime (CPU sur ce Mac — indicatif) ===")
    print(f"{'batch':>6} {'p50(ms)':>9} {'p95(ms)':>9} {'img/s':>8}")
    results = {}
    for b in batches:
        feeds = {nm: np.random.randn(b, 3, 224, 224).astype(np.float32) for nm in names}
        for _ in range(warm):
            sess.run(None, feeds)
        ts = []
        for _ in range(n):
            s = time.perf_counter(); sess.run(None, feeds); ts.append((time.perf_counter() - s) * 1e3)
        ts = np.array(ts)
        thr = b / (ts.mean() / 1e3)
        results[b] = (np.percentile(ts, 50), np.percentile(ts, 95), thr)
        print(f"{b:6d} {np.percentile(ts,50):9.2f} {np.percentile(ts,95):9.2f} {thr:8.0f}")
    return results


def main(a):
    onnx_path = os.path.join(ROOT, f"model_{a.model}.onnx")
    model, dummy, names = export(a.model, onnx_path)
    sess = verify(model, dummy, names, onnx_path)
    res = bench(sess, names)

    # discussion 12 flux : 1 châtaigne = 1 inférence dual-branch ; 12 flux = 6 paires
    # -> au pire 12 images (mono) ou 6 paires (dual) par tick.
    b = 12 if a.model == "simplecnn" else 6
    if b in res or 12 in res:
        p50 = res.get(b, res.get(12))[0]
        print(f"\n>> Pour 1 tick ({'12 images' if a.model=='simplecnn' else '6 paires T/B'}) : "
              f"~{p50:.1f} ms (CPU Mac). Sur GTX 1060 (CUDA/FP16) la latence sera "
              f"nettement plus basse — mesure à refaire sur la cible.")
    size = os.path.getsize(onnx_path) / 1e6
    print(f"Taille du modèle ONNX : {size:.1f} Mo")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["simplecnn", "dualbranch"], default="dualbranch")
    main(ap.parse_args())
