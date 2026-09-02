"""Exporte des figures depuis les runs MLflow (historique + comparatif) — §4.2.

Lit mlflow.db (SQLite) et génère :
    reports/figures/mlflow_history.png     courbes val_acc + rappel Conforme @P95
    reports/figures/mlflow_comparatif.png  meilleur score par run (barres)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "reports", "figures")
DB = "sqlite:///" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow.db")


def main():
    os.makedirs(FIG, exist_ok=True)
    mlflow.set_tracking_uri(DB)
    client = mlflow.tracking.MlflowClient()
    exp = mlflow.get_experiment_by_name("castagnet")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])

    # on garde les runs "complets" (>= 10 epochs loggées)
    def history(run_id, metric):
        h = client.get_metric_history(run_id, metric)
        h = sorted(h, key=lambda m: m.step)
        return [m.step for m in h], [m.value for m in h]

    # --- FIG 1 : historique du meilleur run (le plus d'epochs) ---
    best = None
    for _, r in runs.iterrows():
        steps, _ = history(r.run_id, "val_acc")
        if best is None or len(steps) > best[1]:
            best = (r.run_id, len(steps), r.get("tags.mlflow.runName", "run"))
    rid, _, rname = best
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for metric, label, col in [("val_acc", "val_acc", "#1976d2"),
                               ("conforme_recall_at_p95", "Conforme rappel @P95", "#c62828"),
                               ("conforme_precision", "Conforme précision", "#2e7d32")]:
        s, v = history(rid, metric)
        if s:
            ax.plot(s, v, marker="o", ms=3, label=label, color=col)
    ax.axhline(0.85, ls="--", c="#c62828", alpha=0.4)
    ax.axhline(0.95, ls="--", c="#2e7d32", alpha=0.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("score")
    ax.set_title(f"Historique d'entraînement — {rname}", weight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "mlflow_history.png"), dpi=130)
    plt.close(fig)

    # --- FIG 2 : comparatif des runs (meilleur val_acc) ---
    rows = []
    for _, r in runs.iterrows():
        s, v = history(r.run_id, "val_acc")
        if len(s) >= 10:  # runs complets seulement
            rows.append((r.get("tags.mlflow.runName", "run"),
                         r.get("params.backbone", ""), r.get("params.fusion", ""),
                         max(v)))
    rows.sort(key=lambda x: x[3])
    labels = [f"{n}\n{b} {f}".strip() for n, b, f, _ in rows]
    vals = [x[3] for x in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(range(len(vals)), vals, color="#1976d2")
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels, fontsize=7)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("meilleure val_acc"); ax.set_xlim(0, 1)
    ax.set_title("Comparatif des runs (MLflow)", weight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "mlflow_comparatif.png"), dpi=130)
    plt.close(fig)

    print(f"Figures générées depuis {len(runs)} runs MLflow.")
    print(f"  historique : run '{rname}'")
    print(f"  comparatif : {len(rows)} runs complets")


if __name__ == "__main__":
    main()
