"""Expérience 2 — trajectoires des observables de Gibbs en fonction de la profondeur.

Origine : le croquis fondateur du projet fait de la profondeur N une variable
d'optimisation (``argmin_N H``). Cette expérience mesure, sans aucun
entraînement, la trajectoire de H, F, chi et de l'exactitude après chaque bloc
du ViT visuel (sonde « logit lens », voir ``tlsc/models/layer_probe.py``), sur
cohorte source et corrompue.

Caractère EXPLORATOIRE, consigné a priori : aucun seuil de verdict n'est
pré-enregistré. Attente qualitative non contraignante — sur la source, H_n
décroît avec la profondeur (« cristallisation en profondeur ») ; les couches
intermédiaires n'ayant jamais été alignées à l'espace texte, une trajectoire
plate jusqu'aux dernières couches serait un résultat, pas un échec.

Usage
-----
    python -m experiments.exp02_layer_trajectories --dataset breastmnist \
        --source medmnistc --corruption speckle_noise --severities 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.exp01_zero_training import (
    git_dirty,
    git_sha,
    make_run_id,
    source_sha256,
)
from tlsc.core.gibbs import observables
from tlsc.eval.metrics import shift_detection_auroc


def encode_cohort_layers(clip, cohort, batch_size: int) -> torch.Tensor:
    """Encode une cohorte par couche. Retourne (N_couches, N_images, d)."""
    out = []
    for i in range(0, len(cohort.images), batch_size):
        batch = cohort.images[i:i + batch_size]
        px = torch.stack([clip.preprocess(im) for im in batch])
        out.append(clip.encode_layers(px).cpu())
    return torch.cat(out, dim=1)


def validate_args(args: argparse.Namespace) -> None:
    severities = list(args.severities)
    if not severities or severities[0] != 0:
        raise ValueError("les sévérités doivent commencer par 0, le domaine source")
    if severities != sorted(set(severities)):
        raise ValueError("les sévérités doivent être uniques et strictement croissantes")
    if any(severity not in range(6) for severity in severities):
        raise ValueError("chaque sévérité doit appartenir à l'intervalle [0, 5]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size doit être strictement positif")


def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    started = time.perf_counter()
    cfg = {
        "experiment": "exp02_layer_trajectories",
        "dataset": args.dataset, "model": args.model, "pretrained": args.pretrained,
        "corruption": args.corruption, "corruption_source": args.source,
        "severities": list(args.severities), "prompt_ensemble": True,
        "image_size": args.size, "batch_size": args.batch_size, "seed": args.seed,
        "limit": args.limit, "exploratory": True,
    }
    run_id = make_run_id(cfg)
    outdir = Path(args.outdir) / run_id
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    env = {
        "git_sha": git_sha(), "git_dirty": git_dirty(),
        "source_sha256": source_sha256(), "python": sys.version.split()[0],
        "torch": torch.__version__, "device": args.device,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    print(f"run_id : {run_id}\nsortie : {outdir}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from tlsc.data.medmnistc import build
    from tlsc.models.clip_anchors import PROMPTS, FrozenCLIP

    if args.dataset not in PROMPTS:
        raise SystemExit(f"aucun jeu d'invites pour '{args.dataset}'")
    pset = PROMPTS[args.dataset]
    clip = FrozenCLIP(args.model, args.pretrained, args.device)
    anchors = clip.anchors(pset, ensemble=True).cpu()
    T = clip.temperature
    print(f"température de Gibbs T = 2*tau = {T:.6f}")

    per_severity: dict[str, dict] = {}
    layer_scores: dict[int, dict[str, np.ndarray]] = {}
    n_layers = None

    for s in args.severities:
        coh = build(args.dataset, severity=s, corruption=args.corruption,
                    source=args.source, size=args.size, root=args.root, seed=args.seed)
        if args.limit:
            coh.images, coh.labels = coh.images[:args.limit], coh.labels[:args.limit]
        z_layers = encode_cohort_layers(clip, coh, args.batch_size)  # (N, B, d)
        y = coh.labels
        n_layers = z_layers.shape[0]

        rows = []
        H_all = np.empty((n_layers, len(y)), dtype=np.float32)
        F_all = np.empty((n_layers, len(y)), dtype=np.float32)
        for n in range(n_layers):
            obs = observables(z_layers[n], anchors, T)
            p = obs["p"].numpy()
            acc = float((p.argmax(1) == y).mean())
            rows.append({
                "layer": n + 1, "accuracy": acc,
                "H_mean": float(obs["H"].mean()), "F_mean": float(obs["F"].mean()),
                "chi_mean": float(obs["chi"].mean()),
            })
            H_all[n] = obs["H"].numpy()
            F_all[n] = obs["F"].numpy()
        per_severity[str(s)] = {"n": int(len(y)), "corruption": coh.corruption,
                                "layers": rows}
        layer_scores[s] = {"H": H_all, "F": F_all}
        last = rows[-1]
        print(f"  sévérité {s} · n={len(y):5d} · couche 12 : acc={last['accuracy']:.3f} "
              f"H={last['H_mean']:.4f} F={last['F_mean']:.4f} chi={last['chi_mean']:.3f}")

    # détection source vs cible, couche par couche
    s0 = args.severities[0]
    detection: dict[str, list[dict]] = {}
    for s in args.severities[1:]:
        rows = []
        for n in range(n_layers):
            rows.append({
                "layer": n + 1,
                "auroc_F": shift_detection_auroc(layer_scores[s0]["F"][n],
                                                 layer_scores[s]["F"][n]),
                "auroc_H": shift_detection_auroc(layer_scores[s0]["H"][n],
                                                 layer_scores[s]["H"][n]),
            })
        detection[str(s)] = rows
    smax = args.severities[-1]
    best = max(detection[str(smax)], key=lambda r: r["auroc_F"])
    print(f"  détection sév. {smax} : meilleure couche pour F = {best['layer']} "
          f"(AUROC {best['auroc_F']:.3f})")

    results = {
        "run_id": run_id, "config": cfg, "environment": env,
        "temperature_T": float(T), "n_classes": int(pset.n_classes),
        "n_layers": int(n_layers),
        "elapsed_seconds": time.perf_counter() - started,
        "prompts": [p for row in pset.phrases() for p in row],
        "per_severity": per_severity,
        "per_layer_detection": detection,
        "note": ("Expérience exploratoire : pas de seuil pré-enregistré. "
                 "Les couches intermédiaires sont lues par logit lens ; seule la "
                 "couche finale a été alignée à l'espace texte par l'entraînement."),
    }
    (outdir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(outdir / "scores_layers.npz",
                        **{f"{k}_{s}": v for s, d_ in layer_scores.items()
                           for k, v in d_.items()})
    make_figures(per_severity, detection, outdir / "figures", args)
    print(f"\nmétriques écrites : {outdir/'metrics.json'}")
    return results


def make_figures(per_severity, detection, figdir: Path, args) -> None:
    """Trajectoires par couche et pouvoir de détection par couche."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cold, hot = "#0B6B7A", "#B04A22"
    sevs = sorted(int(s) for s in per_severity)
    cmap = plt.get_cmap("plasma")

    def color(s: int) -> tuple:
        return cmap(0.15 + 0.7 * s / max(sevs[-1], 1))

    def tidy(a) -> None:
        a.grid(alpha=0.25)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.set_xlabel("profondeur n (bloc)")

    # 1 — trajectoires H, chi, F, accuracy par couche
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for s in sevs:
        rows = per_severity[str(s)]["layers"]
        ns = [r["layer"] for r in rows]
        for ax, key, title in zip(
                axes, ("H_mean", "chi_mean", "F_mean", "accuracy"),
                ("Entropie H_n", "Cristallinité chi_n", "Énergie libre F_n", "Exactitude"),
                strict=True):
            ax.plot(ns, [r[key] for r in rows], "o-", ms=3, color=color(s),
                    label=f"sév. {s}")
            ax.set_title(title)
    for ax in axes:
        tidy(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Trajectoires par couche — {args.dataset} · {args.corruption}", y=1.05)
    fig.tight_layout()
    fig.savefig(figdir / "01_trajectoires_couches.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2 — détection par couche, à chaque sévérité
    if detection:
        smax = max(int(s) for s in detection)
        rows = detection[str(smax)]
        ns = [r["layer"] for r in rows]
        fig, a = plt.subplots(figsize=(6.2, 3.8))
        a.plot(ns, [r["auroc_F"] for r in rows], "o-", color=cold, label="énergie libre F")
        a.plot(ns, [r["auroc_H"] for r in rows], "s--", color=hot, label="entropie H")
        a.axhline(0.5, color="#999999", lw=1, ls=":", label="hasard")
        a.set_ylim(0.0, 1.03)
        a.set_ylabel("AUROC de détection")
        a.set_title(f"Détection du décalage par couche (sévérité {smax})")
        a.legend(frameon=False)
        tidy(a)
        fig.tight_layout()
        fig.savefig(figdir / "02_detection_par_couche.png", dpi=150)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="breastmnist")
    p.add_argument("--model", default="ViT-B-16-quickgelu")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--corruption", default="speckle_noise")
    p.add_argument("--source", default="medmnistc", choices=["local", "medmnistc"])
    p.add_argument("--severities", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0, help="0 = tout le split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="data/raw")
    p.add_argument("--outdir", default="outputs")
    args = p.parse_args()
    try:
        run(args)
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
