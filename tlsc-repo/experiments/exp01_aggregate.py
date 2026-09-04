"""Agrégation des runs de l'expérience 1 en tableaux et figures de synthèse.

Parcourt ``outputs/``, retient les runs réels complets, recalcule depuis les
``scores.npz`` l'AUROC du détecteur bilatéral ``|F - médiane_source(F)|``
(et son homologue pour H) ainsi que du détecteur bivarié Mahalanobis (F, H),
puis écrit dans ``outputs/aggregate/`` :

- ``summary.json``   : une entrée par (dataset, corruption, seed, template)
- ``summary.csv``    : le même contenu, aplati par sévérité
- ``fig_grid_detection.png`` : grille AUROC(F) vs AUROC(H) par corruption
- ``fig_two_sided.png``      : unilatéral vs bilatéral à sévérité max

Chaque valeur reste rattachée à son run_id d'origine.

Usage
-----
    python -m experiments.exp01_aggregate outputs
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tlsc.eval.metrics import (
    bivariate_shift_score,
    shift_detection_auroc,
    two_sided_shift_score,
)


def collect_run(run_dir: Path) -> dict | None:
    """Charge un run réel complet ; None sinon."""
    metrics_path, scores_path = run_dir / "metrics.json", run_dir / "scores.npz"
    if not metrics_path.is_file() or not scores_path.is_file():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        # runs antérieurs au passage systématique en UTF-8 (Windows, cp1252)
        metrics = json.loads(metrics_path.read_text(encoding="cp1252"))
    if metrics.get("config", {}).get("dry_run"):
        return None

    scores = np.load(scores_path)
    severities = metrics["config"]["severities"]
    s0 = severities[0]
    detection: dict[str, dict[str, float]] = {}
    for s in severities[1:]:
        entry = dict(metrics["shift_detection"].get(str(s), {}))
        f2_src, f2_tgt = two_sided_shift_score(scores[f"F_{s0}"], scores[f"F_{s}"])
        h2_src, h2_tgt = two_sided_shift_score(scores[f"H_{s0}"], scores[f"H_{s}"])
        entry["auroc_F_two_sided"] = shift_detection_auroc(f2_src, f2_tgt)
        entry["auroc_H_two_sided"] = shift_detection_auroc(h2_src, h2_tgt)
        fh_src, fh_tgt = bivariate_shift_score(
            scores[f"F_{s0}"], scores[f"H_{s0}"], scores[f"F_{s}"], scores[f"H_{s}"])
        entry["auroc_FH_bivariate"] = shift_detection_auroc(fh_src, fh_tgt)
        detection[str(s)] = entry

    analysis_path = run_dir / "analysis.json"
    analysis = (json.loads(analysis_path.read_text(encoding="utf-8"))
                if analysis_path.is_file() else None)

    cfg = metrics["config"]
    return {
        "run_id": metrics["run_id"],
        "dataset": cfg["dataset"],
        "corruption": cfg["corruption"],
        "corruption_source": cfg["corruption_source"],
        "seed": cfg["seed"],
        "template_index": cfg.get("template_index"),
        "prompt_ensemble": cfg.get("prompt_ensemble", True),
        "severities": severities,
        "git_sha": metrics["environment"]["git_sha"],
        "git_dirty": metrics["environment"]["git_dirty"],
        "per_severity_stats": {
            str(s): {k: metrics["per_severity"][str(s)][k]
                     for k in ("accuracy", "auroc", "H_mean", "F_mean", "chi_mean", "ece")}
            for s in severities
        },
        "detection": detection,
        "analysis": ({s: a for s, a in analysis["per_severity"].items()}
                     if analysis else None),
    }


def is_canonical(run: dict) -> bool:
    """Run de la configuration principale : ensemble d'invites, graine 0."""
    return (run["prompt_ensemble"] and run["template_index"] is None
            and run["seed"] == 0 and len(run["severities"]) == 6)


def write_csv(runs: list[dict], path: Path) -> None:
    fields = ["run_id", "dataset", "corruption", "seed", "template_index",
              "prompt_ensemble", "severity", "auroc_F", "auroc_H",
              "auroc_F_two_sided", "auroc_H_two_sided", "auroc_FH_bivariate",
              "delta_auroc", "delta_ci_low", "delta_ci_high", "delong_p"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for sev, det in run["detection"].items():
                ana = (run["analysis"] or {}).get(sev, {})
                writer.writerow({
                    "run_id": run["run_id"], "dataset": run["dataset"],
                    "corruption": run["corruption"], "seed": run["seed"],
                    "template_index": run["template_index"],
                    "prompt_ensemble": run["prompt_ensemble"], "severity": sev,
                    "auroc_F": det.get("auroc_F"), "auroc_H": det.get("auroc_H"),
                    "auroc_F_two_sided": det.get("auroc_F_two_sided"),
                    "auroc_H_two_sided": det.get("auroc_H_two_sided"),
                    "auroc_FH_bivariate": det.get("auroc_FH_bivariate"),
                    "delta_auroc": ana.get("delta_auroc"),
                    "delta_ci_low": ana.get("delta_ci_low"),
                    "delta_ci_high": ana.get("delta_ci_high"),
                    "delong_p": ana.get("delong_p"),
                })


def make_figures(runs: list[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cold, hot, ink = "#0B6B7A", "#B04A22", "#0E1620"
    canonical = [r for r in runs if is_canonical(r)]
    datasets = sorted({r["dataset"] for r in canonical})

    # 1 — grille de détection par corruption
    for ds in datasets:
        rs = sorted((r for r in canonical if r["dataset"] == ds),
                    key=lambda r: r["corruption"])
        if not rs:
            continue
        ncols = min(4, len(rs))
        nrows = -(-len(rs) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows),
                                 sharey=True, squeeze=False)
        for ax in axes.flat[len(rs):]:
            ax.axis("off")
        for ax, run in zip(axes.flat, rs, strict=False):
            sevs = sorted(int(s) for s in run["detection"])
            ax.plot(sevs, [run["detection"][str(s)]["auroc_F"] for s in sevs],
                    "o-", color=cold, label="F")
            ax.plot(sevs, [run["detection"][str(s)]["auroc_H"] for s in sevs],
                    "s--", color=hot, label="H")
            ax.axhline(0.5, color="#999999", lw=1, ls=":")
            ax.set_ylim(0.0, 1.03)
            ax.set_title(run["corruption"], fontsize=10)
            ax.grid(alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes.flat[0].legend(frameon=False, fontsize=9)
        for ax in axes[-1]:
            ax.set_xlabel("sévérité")
        for row in axes:
            row[0].set_ylabel("AUROC détection")
        fig.suptitle(f"Détection du décalage par corruption — {ds}", y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / f"fig_grid_detection_{ds}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # 2 — unilatéral vs bilatéral vs bivarié à sévérité maximale
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    labels, one_sided, two_sided, bivariate = [], [], [], []
    for run in sorted(canonical, key=lambda r: (r["dataset"], r["corruption"])):
        smax = str(max(int(s) for s in run["detection"]))
        det = run["detection"][smax]
        labels.append(f"{run['dataset'][:6]}·{run['corruption']}")
        one_sided.append(det["auroc_F"])
        two_sided.append(det["auroc_F_two_sided"])
        bivariate.append(det["auroc_FH_bivariate"])
    x = np.arange(len(labels))
    ax.bar(x - 0.27, one_sided, 0.27, color=ink, label="F unilatéral")
    ax.bar(x, two_sided, 0.27, color=cold, label="|F − méd. source|")
    ax.bar(x + 0.27, bivariate, 0.27, color=hot, label="Mahalanobis (F, H)")
    ax.axhline(0.5, color="#999999", lw=1, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUROC à sévérité max")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / "fig_two_sided.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", default="outputs")
    args = p.parse_args()

    root = Path(args.path).resolve()
    runs = [r for d in sorted(root.iterdir()) if d.is_dir() and d.name != "aggregate"
            for r in [collect_run(d)] if r is not None]
    if not runs:
        raise SystemExit("aucun run réel agrégeable trouvé")

    outdir = root / "aggregate"
    outdir.mkdir(exist_ok=True)
    (outdir / "summary.json").write_text(
        json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(runs, outdir / "summary.csv")
    make_figures(runs, outdir)

    canonical = [r for r in runs if is_canonical(r)]
    print(f"{len(runs)} runs agrégés dont {len(canonical)} canoniques.")
    for run in sorted(canonical, key=lambda r: (r["dataset"], r["corruption"])):
        smax = str(max(int(s) for s in run["detection"]))
        det = run["detection"][smax]
        print(f"  {run['dataset']:15s} {run['corruption']:18s} sév.{smax} : "
              f"F={det['auroc_F']:.3f} H={det['auroc_H']:.3f} "
              f"F2s={det['auroc_F_two_sided']:.3f} "
              f"FH={det['auroc_FH_bivariate']:.3f} · {run['run_id']}")
    print(f"\nécrit dans {outdir}")


if __name__ == "__main__":
    main()
