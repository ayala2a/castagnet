"""Génère les figures et statistiques du rapport qualité §4.1.

Compare l'état "AVANT" (label_filename = 3 classes, base du rapport PDF fourni,
obsolète) et "APRÈS" (label_principal = 4 classes réelles), au niveau image ET
au niveau châtaigne (paires T/B agrégées).

Usage :
    python src/data/quality_report.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELS = os.path.join(ROOT, "..", "castagnia_data-main", "labels_principal.csv")
MASKED = os.path.join(ROOT, "..", "castagnia_data-main", "labels_masked.csv")
PAIRS = os.path.join(ROOT, "data", "pairs_TB.csv")
FIGDIR = os.path.join(ROOT, "reports", "figures")

COLORS = {"Conforme": "#2e7d32", "NON Conforme": "#c62828",
          "PIETRA": "#ef6c00", "Vide": "#607d8b"}
ORDER = ["Conforme", "NON Conforme", "PIETRA", "Vide"]


def _bar(ax, series, title, order=None):
    order = order or list(series.index)
    vals = [series.get(k, 0) for k in order]
    ax.bar(order, vals, color=[COLORS.get(k, "#888") for k in order])
    ax.set_title(title, fontsize=11, weight="bold")
    ax.tick_params(axis="x", rotation=20)
    tot = sum(vals)
    for i, v in enumerate(vals):
        if tot:
            ax.text(i, v, f"{v}\n{v/tot:.0%}", ha="center", va="bottom", fontsize=8)
    ax.margins(y=0.15)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df = pd.read_csv(LABELS, dtype={"labeled_by": str})
    df["labeled_by"] = df["labeled_by"].fillna("")
    df["label_filename"] = df["label_filename"].str.replace("_", " ")
    pairs = pd.read_csv(PAIRS)
    stats = {}

    # === FIG 1 : AVANT / APRÈS / PAR CHÂTAIGNE ===============================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _bar(axes[0], df["label_filename"].value_counts(),
         "AVANT — label_filename\n(base du rapport PDF fourni)", ORDER[:3])
    _bar(axes[1], df["label_principal"].value_counts(),
         "APRÈS — label_principal\n(par image, 4 classes)", ORDER)
    _bar(axes[2], pairs["label_chestnut"].value_counts(),
         "PAR CHÂTAIGNE\n(paires T/B agrégées)", ORDER)
    fig.suptitle("Répartition des labels — l'apparition de la classe « Vide »",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGDIR, "01_avant_apres.png"), dpi=130)
    plt.close(fig)

    stats["avant_label_filename"] = df["label_filename"].value_counts().to_dict()
    stats["apres_label_principal"] = df["label_principal"].value_counts().to_dict()
    stats["par_chataigne"] = pairs["label_chestnut"].value_counts().to_dict()

    # === FIG 2 : PAR ANNÉE et PAR CAMÉRA (label_principal) ==================
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ct_year = pd.crosstab(df["year"], df["label_principal"])[ORDER]
    ct_year.plot(kind="bar", stacked=True, ax=axes[0],
                 color=[COLORS[c] for c in ORDER])
    axes[0].set_title("Par année (label_principal)", weight="bold")
    axes[0].tick_params(axis="x", rotation=0)
    ct_cam = pd.crosstab(df["cam_num"], df["label_principal"])[ORDER]
    ct_cam.plot(kind="bar", stacked=True, ax=axes[1],
                color=[COLORS[c] for c in ORDER])
    axes[1].set_title("Par caméra (label_principal)", weight="bold")
    axes[1].tick_params(axis="x", rotation=0)
    for a in axes:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "02_annee_camera.png"), dpi=130)
    plt.close(fig)
    stats["par_annee"] = {int(y): row.to_dict() for y, row in ct_year.iterrows()}
    stats["par_camera"] = {int(c): row.to_dict() for c, row in ct_cam.iterrows()}

    # === FIG 3 : COMPOSITION DES DÉSACCORDS T/B ============================
    comp = pairs[(pairs.has_T) & (pairs.has_B)].copy()
    from collections import Counter
    combo = Counter(
        tuple(sorted([t, b])) for t, b in zip(comp["T_label"], comp["B_label"])
    )
    labels_c = [f"{a}\n+ {b}" for (a, b) in combo.keys()]
    vals_c = list(combo.values())
    cols = ["#4caf50" if a == b else "#ff9800" for (a, b) in combo.keys()]
    order_idx = sorted(range(len(vals_c)), key=lambda i: -vals_c[i])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([labels_c[i] for i in order_idx], [vals_c[i] for i in order_idx],
           color=[cols[i] for i in order_idx])
    ax.set_title("Composition des couples de vues T/B (16 176 paires)\n"
                 "vert = accord · orange = une face « Vide »", weight="bold")
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    tot = sum(vals_c)
    for i, idx in enumerate(order_idx):
        ax.text(i, vals_c[idx], f"{vals_c[idx]/tot:.0%}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "03_desaccords_TB.png"), dpi=130)
    plt.close(fig)
    stats["combinaisons_TB"] = {f"{a}+{b}": v for (a, b), v in combo.items()}
    disc = comp[comp["T_label"] != comp["B_label"]]
    stats["desaccords_TB"] = {
        "total": int(len(disc)),
        "avec_Vide": int(((disc.T_label == "Vide") | (disc.B_label == "Vide")).sum()),
        "conflits_reels": int(comp.real_conflict.sum()),
    }

    # === FIG 4 : TRAÇABILITÉ / RELECTURE ===================================
    lb = df["labeled_by"].replace("", "(non tracé)").value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].bar(lb.index.astype(str), lb.values, color="#3f51b5")
    axes[0].set_title("Origine des labels (labeled_by)", weight="bold")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    rev = df["reviewed"].value_counts()
    axes[1].bar(["reviewed=True", "reviewed=False"],
                [int(rev.get(True, 0)), int(rev.get(False, 0))],
                color=["#4caf50", "#c62828"])
    axes[1].set_title("Taux de relecture", weight="bold")
    for i, v in enumerate([int(rev.get(True, 0)), int(rev.get(False, 0))]):
        axes[1].text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "04_tracabilite.png"), dpi=130)
    plt.close(fig)
    stats["labeled_by"] = {str(k): int(v) for k, v in lb.items()}
    stats["reviewed_true"] = int(rev.get(True, 0))
    stats["reviewed_false"] = int(rev.get(False, 0))
    m = (df["reviewed"] == True) & (df["labeled_by"] == "")
    stats["reviewed_sans_auteur"] = int(m.sum())

    # tags
    stats["tags"] = {t: int(df[t].fillna(False).astype(bool).sum())
                     for t in ["multiple", "chunk", "mixed_quality"]}

    # masked coverage
    mk = pd.read_csv(MASKED)
    stats["masked"] = {
        "lignes": int(len(mk)),
        "couverture_pct": round(100 * len(set(mk.filename) & set(df.filename)) / len(df), 1),
        "valeurs": mk["label"].value_counts().to_dict(),
    }

    with open(os.path.join(ROOT, "reports", "quality_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("Figures + reports/quality_stats.json générés.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
