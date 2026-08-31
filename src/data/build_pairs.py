"""Appariement des vues Top (T) / Bottom (B) d'une même châtaigne.

Une châtaigne physique = 2 photos (caméra du dessus T + caméra du dessous B).
La clé d'appariement est le nom de fichier privé de la position de caméra :

    2025_Conforme_1_Cam_B_1_0.jpg  ->  2025_Conforme_1_Cam_X_1_0.jpg
    2026_Conforme_Cam_B_1_0.jpg    ->  2026_Conforme_Cam_X_1_0.jpg

Règle d'agrégation du label par châtaigne (justifiée par l'analyse : les
désaccords T/B impliquent TOUJOURS "Vide" d'un côté, jamais deux vraies
classes en conflit) :

    label_chataigne = classe NON-Vide parmi {T, B}
        - T et B d'accord            -> cette classe
        - une face Vide, l'autre X   -> X   (le fruit est là, une caméra ne l'a pas capté)
        - les deux Vide              -> Vide

Sorties :
    data/pairs_TB.csv        une ligne par châtaigne (paire ou orphelin)
    data/labels_chestnut.csv labels agrégés par châtaigne, prêts pour le split
"""

import argparse
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LABELS = os.path.join(ROOT, "..", "castagnia_data-main", "labels_principal.csv")
DEFAULT_OUT = os.path.join(ROOT, "data")

VIDE = "Vide"
CLASSES = ["Conforme", "NON Conforme", "PIETRA", VIDE]

PAIRKEY_RE = re.compile(r"_Cam_[TB]_")


def pairkey(filename: str) -> str:
    """Clé identifiant une châtaigne, indépendante de la caméra T/B."""
    return PAIRKEY_RE.sub("_Cam_X_", filename)


def aggregate_label(labels: list[str]) -> tuple[str, bool]:
    """Retourne (label_chataigne, conflit_reel).

    conflit_reel = True s'il existe DEUX vraies classes différentes (hors Vide)
    parmi les vues — cas qui, dans ce dataset, ne se produit jamais mais qu'on
    trace par sécurité.
    """
    non_vide = [l for l in labels if l != VIDE]
    if not non_vide:
        return VIDE, False
    uniques = set(non_vide)
    conflict = len(uniques) > 1
    # non-Vide gagne ; en cas de conflit théorique, on prend la classe la plus
    # sévère selon la priorité qualité (protéger la pureté du lot Conforme).
    severity = {"PIETRA": 3, "NON Conforme": 2, "Conforme": 1}
    best = max(non_vide, key=lambda l: severity.get(l, 0))
    return best, conflict


def build(labels_csv: str, out_dir: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv, dtype={"labeled_by": str})
    df["labeled_by"] = df["labeled_by"].fillna("")
    df["pairkey"] = df["filename"].map(pairkey)

    records = []
    for key, sub in df.groupby("pairkey"):
        by_pos = {r.cam_position: r for r in sub.itertuples(index=False)}
        t, b = by_pos.get("T"), by_pos.get("B")
        views = [r.label_principal for r in (t, b) if r is not None]
        chestnut_label, conflict = aggregate_label(views)

        ref = t or b  # métadonnées communes à la paire
        records.append(
            {
                "pairkey": key,
                "year": ref.year,
                "cam_num": ref.cam_num,
                "sample_num": ref.sample_num,
                "label_filename": ref.label_filename,
                "T_filename": t.filename if t is not None else "",
                "B_filename": b.filename if b is not None else "",
                "T_label": t.label_principal if t is not None else "",
                "B_label": b.label_principal if b is not None else "",
                "has_T": t is not None,
                "has_B": b is not None,
                "n_views": len(views),
                "agreed": (t is not None and b is not None and t.label_principal == b.label_principal),
                "real_conflict": conflict,
                "label_chestnut": chestnut_label,
                # tags : OR logique des deux vues
                "multiple": bool(getattr(t, "multiple", False)) or bool(getattr(b, "multiple", False)),
                "chunk": bool(getattr(t, "chunk", False)) or bool(getattr(b, "chunk", False)),
                "reviewed": bool(getattr(t, "reviewed", False)) and bool(getattr(b, "reviewed", False))
                if (t is not None and b is not None)
                else bool(getattr(ref, "reviewed", False)),
            }
        )

    pairs = pd.DataFrame.from_records(records).sort_values("pairkey").reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    pairs.to_csv(os.path.join(out_dir, "pairs_TB.csv"), index=False)

    chestnut = pairs[
        ["pairkey", "year", "cam_num", "T_filename", "B_filename", "has_T", "has_B",
         "label_chestnut", "multiple", "chunk", "reviewed"]
    ].copy()
    chestnut.to_csv(os.path.join(out_dir, "labels_chestnut.csv"), index=False)

    # --- résumé console ---
    n = len(pairs)
    complete = pairs[(pairs.has_T) & (pairs.has_B)]
    print(f"Châtaignes (paires + orphelins) : {n}")
    print(f"  paires complètes T+B : {len(complete)} ({len(complete)/n:.1%})")
    print(f"  orphelins T seul     : {int(((pairs.has_T) & ~(pairs.has_B)).sum())}")
    print(f"  orphelins B seul     : {int((~(pairs.has_T) & (pairs.has_B)).sum())}")
    print(f"  conflits réels T/B (2 vraies classes) : {int(pairs.real_conflict.sum())}")
    print("Distribution label_chestnut :")
    for k, v in pairs["label_chestnut"].value_counts().items():
        print(f"    {k:14} {v:6} ({v/n:.1%})")
    return pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    build(args.labels, args.out)
