"""Découpage train / val / test — au niveau CHÂTAIGNE, stratifié et anti-fuite.

- Unité = châtaigne (paire T/B) -> les 2 vues d'un fruit tombent dans le MÊME split
  (pas de fuite T/B).
- Stratification par (classe × année) pour équilibrer classes et années.
- Proportions 70/15/15, figées par random_state pour comparer les modèles à égalité.

Sorties :
    data/splits_chestnut.csv  (1 ligne / châtaigne + colonne `split`)
    data/splits_image.csv     (1 ligne / image T ou B, `split` hérité de la châtaigne)
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
SEED = 42


def make(chestnut_csv: str, out_dir: str, val=0.15, test=0.15,
         drop_ambiguous=False):
    df = pd.read_csv(chestnut_csv)
    if drop_ambiguous:
        df = df[~(df["multiple"] | df["chunk"])].reset_index(drop=True)

    # clé de stratification : classe + année
    df["strat"] = df["label_chestnut"] + "_" + df["year"].astype(str)

    # 1) train vs (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=val + test, stratify=df["strat"], random_state=SEED)
    # 2) val vs test à l'intérieur du reste
    rel_test = test / (val + test)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["strat"], random_state=SEED)

    for part, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        df.loc[part.index, "split"] = name

    os.makedirs(out_dir, exist_ok=True)
    cols = ["pairkey", "split", "label_chestnut", "year", "cam_num",
            "T_filename", "B_filename", "has_T", "has_B", "multiple", "chunk"]
    df[cols].to_csv(os.path.join(out_dir, "splits_chestnut.csv"), index=False)

    # --- version image-level (baseline mono-vue), split hérité ---
    rows = []
    for r in df.itertuples(index=False):
        for pos, fn in (("T", r.T_filename), ("B", r.B_filename)):
            if isinstance(fn, str) and fn:
                rows.append({"filename": fn, "cam_position": pos,
                             "label": r.label_chestnut, "split": r.split,
                             "pairkey": r.pairkey})
    img = pd.DataFrame(rows)
    img.to_csv(os.path.join(out_dir, "splits_image.csv"), index=False)

    # --- rapport console : vérifier stratification + absence de fuite ---
    print("=== Répartition châtaignes par split ===")
    print(df["split"].value_counts().reindex(["train", "val", "test"]).to_string())
    print("\n=== Distribution des classes par split (%) ===")
    ct = pd.crosstab(df["split"], df["label_chestnut"], normalize="index") * 100
    print(ct.reindex(["train", "val", "test"]).round(1).to_string())
    # anti-fuite : aucune châtaigne (pairkey) dans 2 splits
    leak = img.groupby("pairkey")["split"].nunique().gt(1).sum()
    print(f"\nFuite T/B (pairkey dans >1 split) : {leak}  ->  {'OK' if leak==0 else 'PROBLÈME'}")
    print(f"Images : {len(img)}  |  châtaignes : {len(df)}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chestnut", default=os.path.join(DATA, "labels_chestnut.csv"))
    ap.add_argument("--out", default=DATA)
    ap.add_argument("--drop-ambiguous", action="store_true",
                    help="exclure les images taguées multiple/chunk du dataset")
    args = ap.parse_args()
    make(args.chestnut, args.out, drop_ambiguous=args.drop_ambiguous)
