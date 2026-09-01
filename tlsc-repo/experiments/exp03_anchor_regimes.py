"""Expérience 3 — régimes d'ancrage : invites textuelles vs centroïdes de données.

Origine : le croquis fondateur définit les ancres comme des centroïdes de classe
calculés depuis les données (mu_0, mu_1) et la distance inter-classes D_inter
comme grandeur à maximiser. Le protocole v3 (§2.3) organise cette comparaison en
trois régimes :

R1  invites textuelles pures (zero-shot) ;
R2  centroïdes few-shot (k exemples étiquetés par classe, domaine source) ;
R3  centroïdes sur le split train source complet — plafond de référence.

Pour chaque régime : D_inter, classification par sévérité, et détection du
décalage (AUROC de F et H). Anti-fuite : les centroïdes ne voient que le split
TRAIN source à sévérité 0 ; le split test reste en lecture seule.

Caractère EXPLORATOIRE déclaré : aucun seuil pré-enregistré.

Usage
-----
    python -m experiments.exp03_anchor_regimes --dataset breastmnist \
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
    encode_cohort,
    git_dirty,
    git_sha,
    make_run_id,
    source_sha256,
)
from tlsc.core.gibbs import observables
from tlsc.data.medmnistc import build
from tlsc.eval.metrics import diagnostic_metrics, shift_detection_auroc
from tlsc.models.data_anchors import class_centroids, inter_class_distance


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
    if args.k_shot <= 0:
        raise ValueError("--k-shot doit être strictement positif")


def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    started = time.perf_counter()
    cfg = {
        "experiment": "exp03_anchor_regimes",
        "dataset": args.dataset, "model": args.model, "pretrained": args.pretrained,
        "corruption": args.corruption, "corruption_source": args.source,
        "severities": list(args.severities), "k_shot": args.k_shot,
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

    from tlsc.models.clip_anchors import PROMPTS, FrozenCLIP

    if args.dataset not in PROMPTS:
        raise SystemExit(f"aucun jeu d'invites pour '{args.dataset}'")
    pset = PROMPTS[args.dataset]
    clip = FrozenCLIP(args.model, args.pretrained, args.device)
    T = clip.temperature
    print(f"température de Gibbs T = 2*tau = {T:.6f}")

    # ── ancres des trois régimes (source uniquement) ─────────────────────
    train = build(args.dataset, severity=0, corruption=args.corruption,
                  source=args.source, split="train", size=args.size,
                  root=args.root, seed=args.seed)
    z_train = encode_cohort(clip, train, args.batch_size)
    y_train = torch.as_tensor(train.labels, dtype=torch.long)
    print(f"split train source encodé : n={len(y_train)}")

    regimes = {
        "R1_texte": clip.anchors(pset, ensemble=True).cpu(),
        "R2_fewshot": class_centroids(z_train, y_train, k_shot=args.k_shot,
                                      seed=args.seed),
        "R3_centroides": class_centroids(z_train, y_train),
    }
    d_inter = {name: inter_class_distance(mu) for name, mu in regimes.items()}
    for name, value in d_inter.items():
        print(f"  D_inter[{name}] = {value:.4f}")

    # ── balayage test par sévérité, pour chaque régime ───────────────────
    per_regime: dict[str, dict] = {name: {"D_inter": d_inter[name], "per_severity": {}}
                                   for name in regimes}
    scores: dict[str, dict[int, dict[str, np.ndarray]]] = {n: {} for n in regimes}

    for s in args.severities:
        coh = build(args.dataset, severity=s, corruption=args.corruption,
                    source=args.source, size=args.size, root=args.root, seed=args.seed)
        if args.limit:
            coh.images, coh.labels = coh.images[:args.limit], coh.labels[:args.limit]
        z = encode_cohort(clip, coh, args.batch_size)
        y = coh.labels
        for name, mu in regimes.items():
            obs = observables(z, mu, T)
            p = obs["p"].numpy()
            m = diagnostic_metrics(y, p)
            m |= {"H_mean": float(obs["H"].mean()), "F_mean": float(obs["F"].mean()),
                  "chi_mean": float(obs["chi"].mean()), "n": int(len(y)),
                  "corruption": coh.corruption}
            per_regime[name]["per_severity"][str(s)] = m
            scores[name][s] = {"H": obs["H"].numpy(), "F": obs["F"].numpy()}
        r3 = per_regime["R3_centroides"]["per_severity"][str(s)]
        print(f"  sévérité {s} · n={len(y):5d} · R3 : acc={r3['accuracy']:.3f} "
              f"bal={r3['balanced_accuracy']:.3f} H={r3['H_mean']:.4f}")

    # ── détection source vs cible, par régime ────────────────────────────
    s0 = args.severities[0]
    for name in regimes:
        detection = {}
        for s in args.severities[1:]:
            detection[str(s)] = {
                "auroc_F": shift_detection_auroc(scores[name][s0]["F"],
                                                 scores[name][s]["F"]),
                "auroc_H": shift_detection_auroc(scores[name][s0]["H"],
                                                 scores[name][s]["H"]),
            }
        per_regime[name]["shift_detection"] = detection
    smax = str(args.severities[-1])
    for name in regimes:
        det = per_regime[name]["shift_detection"][smax]
        acc0 = per_regime[name]["per_severity"]["0"]["balanced_accuracy"]
        print(f"  {name:14s} : bal.acc(0)={acc0:.3f} · sév.{smax} "
              f"AUROC(F)={det['auroc_F']:.3f} AUROC(H)={det['auroc_H']:.3f}")

    results = {
        "run_id": run_id, "config": cfg, "environment": env,
        "temperature_T": float(T), "n_classes": int(pset.n_classes),
        "n_train_source": int(len(y_train)),
        "elapsed_seconds": time.perf_counter() - started,
        "prompts_R1": [p for row in pset.phrases() for p in row],
        "regimes": per_regime,
        "note": ("Expérience exploratoire : pas de seuil pré-enregistré. Les "
                 "centroïdes R2/R3 sont calculés sur le split train SOURCE à "
                 "sévérité 0 uniquement (anti-fuite)."),
    }
    (outdir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        outdir / "scores_regimes.npz",
        **{f"{name}_{k}_{s}": v for name, d_ in scores.items()
           for s, kv in d_.items() for k, v in kv.items()})
    make_figures(per_regime, args, outdir / "figures")
    print(f"\nmétriques écrites : {outdir/'metrics.json'}")
    return results


def make_figures(per_regime: dict, args, figdir: Path) -> None:
    """Détection et classification par sévérité, un panneau par grandeur."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"R1_texte": ("#B04A22", "o-"), "R2_fewshot": ("#8a6bbf", "s--"),
              "R3_centroides": ("#0B6B7A", "d-")}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, block in per_regime.items():
        color, style = styles[name]
        sevs = sorted(int(s) for s in block["shift_detection"])
        axes[0].plot(sevs, [block["shift_detection"][str(s)]["auroc_F"] for s in sevs],
                     style, color=color, label=name)
        axes[1].plot(sevs, [block["shift_detection"][str(s)]["auroc_H"] for s in sevs],
                     style, color=color, label=name)
        all_sevs = sorted(int(s) for s in block["per_severity"])
        axes[2].plot(all_sevs,
                     [block["per_severity"][str(s)]["balanced_accuracy"] for s in all_sevs],
                     style, color=color, label=name)
    titles = ("Détection : AUROC(F)", "Détection : AUROC(H)", "Exactitude équilibrée")
    for ax, title in zip(axes, titles, strict=True):
        ax.axhline(0.5, color="#999999", lw=1, ls=":")
        ax.set_xlabel("sévérité")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.03)
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Régimes d'ancrage — {args.dataset} · {args.corruption}", y=1.04)
    fig.tight_layout()
    fig.savefig(figdir / "01_regimes_ancrage.png", dpi=150, bbox_inches="tight")
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
    p.add_argument("--k-shot", type=int, default=16)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0, help="0 = tout le split test")
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
