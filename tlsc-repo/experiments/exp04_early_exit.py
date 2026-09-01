"""Expérience 4 — arrêt anticipé par échantillon, calibré sur la source.

Origine : le croquis fondateur (`argmin_N H`) et sa règle online « sortir quand
H < epsilon ». Chaque image sort à la première couche où son observable (H ou F)
passe sous un seuil calibré sur un split source dédié ; on mesure la profondeur
moyenne et l'exactitude sous décalage croissant.

Protocole anti-fuite : le split test source (sévérité 0) est coupé en deux
moitiés stratifiées — CALIBRATION (choix des seuils et de la couche fixe de
référence) et ÉVALUATION (tout le reste). Les sévérités > 0 ne sont évaluées
que sur les indices d'évaluation. Aucune décision n'utilise la cible.

Caractère EXPLORATOIRE déclaré : aucun seuil de verdict pré-enregistré.

Usage
-----
    python -m experiments.exp04_early_exit --dataset breastmnist \
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
from experiments.exp02_layer_trajectories import encode_cohort_layers
from tlsc.core.gibbs import observables
from tlsc.data.medmnistc import build
from tlsc.eval.early_exit import calibrate_epsilon, simulate_early_exit


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
    if not 0.0 <= args.tolerance < 1.0:
        raise ValueError("--tolerance doit appartenir à [0, 1[")


def stratified_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deux moitiés stratifiées par classe : (calibration, évaluation)."""
    rng = np.random.default_rng(seed)
    calib, evaluation = [], []
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        calib.append(idx[:half])
        evaluation.append(idx[half:])
    return np.sort(np.concatenate(calib)), np.sort(np.concatenate(evaluation))


def layer_scores_and_correct(z_layers: torch.Tensor, anchors: torch.Tensor,
                             T: float, y: np.ndarray) -> dict[str, np.ndarray]:
    """H, F et exactitude par couche : tableaux (n_layers, n)."""
    n_layers = z_layers.shape[0]
    out = {"H": [], "F": [], "correct": []}
    for n in range(n_layers):
        obs = observables(z_layers[n], anchors, T)
        out["H"].append(obs["H"].numpy())
        out["F"].append(obs["F"].numpy())
        out["correct"].append(obs["p"].numpy().argmax(1) == y)
    return {k: np.stack(v) for k, v in out.items()}


def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    started = time.perf_counter()
    cfg = {
        "experiment": "exp04_early_exit",
        "dataset": args.dataset, "model": args.model, "pretrained": args.pretrained,
        "corruption": args.corruption, "corruption_source": args.source,
        "severities": list(args.severities), "tolerance": args.tolerance,
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
    anchors = clip.anchors(pset, ensemble=True).cpu()
    T = clip.temperature
    print(f"température de Gibbs T = 2*tau = {T:.6f}")

    # ── encodage par couche, toutes sévérités ────────────────────────────
    per_sev_arrays: dict[int, dict[str, np.ndarray]] = {}
    y_ref: np.ndarray | None = None
    for s in args.severities:
        coh = build(args.dataset, severity=s, corruption=args.corruption,
                    source=args.source, size=args.size, root=args.root, seed=args.seed)
        if args.limit:
            coh.images, coh.labels = coh.images[:args.limit], coh.labels[:args.limit]
        z_layers = encode_cohort_layers(clip, coh, args.batch_size)
        per_sev_arrays[s] = layer_scores_and_correct(z_layers, anchors, T, coh.labels)
        if s == 0:
            y_ref = coh.labels
        print(f"  sévérité {s} encodée · n={len(coh.labels)}")

    # ── calibration sur la moitié source ─────────────────────────────────
    calib_idx, eval_idx = stratified_split(y_ref, args.seed)
    src = per_sev_arrays[0]
    calibration = {}
    for rule in ("H", "F"):
        calibration[rule] = calibrate_epsilon(
            src[rule][:, calib_idx], src["correct"][:, calib_idx],
            tolerance=args.tolerance)
    acc_calib_layers = src["correct"][:, calib_idx].mean(axis=1)
    fixed_layer = int(acc_calib_layers.argmax())  # 0-based
    calibration["fixed_layer"] = {"layer": fixed_layer + 1,
                                  "accuracy_calib": float(acc_calib_layers[fixed_layer])}
    for rule in ("H", "F"):
        c = calibration[rule]
        print(f"  calibration {rule}-stop : epsilon={c['epsilon']:.4f} · "
              f"profondeur={c['mean_depth']:.2f} · acc={c['accuracy']:.3f} "
              f"(pleine profondeur {c['accuracy_full']:.3f})")
    print(f"  couche fixe de référence (calibration) : {fixed_layer + 1}")

    # ── évaluation sous décalage, indices d'évaluation seulement ─────────
    per_severity: dict[str, dict] = {}
    for s in args.severities:
        arrays = per_sev_arrays[s]
        block: dict[str, dict | float | int] = {"n_eval": int(len(eval_idx))}
        for rule in ("H", "F"):
            out = simulate_early_exit(arrays[rule][:, eval_idx],
                                      arrays["correct"][:, eval_idx],
                                      calibration[rule]["epsilon"])
            block[f"{rule}_stop"] = out
        block["full_depth"] = {"accuracy": float(arrays["correct"][-1, eval_idx].mean()),
                               "mean_depth": float(arrays["correct"].shape[0])}
        block["fixed_layer"] = {"layer": fixed_layer + 1,
                                "accuracy": float(arrays["correct"][fixed_layer, eval_idx].mean()),
                                "mean_depth": float(fixed_layer + 1)}
        per_severity[str(s)] = block
        print(f"  sévérité {s} : full={block['full_depth']['accuracy']:.3f} · "
              f"H-stop={block['H_stop']['accuracy']:.3f}@{block['H_stop']['mean_depth']:.2f} · "
              f"F-stop={block['F_stop']['accuracy']:.3f}@{block['F_stop']['mean_depth']:.2f} · "
              f"fixe({fixed_layer + 1})={block['fixed_layer']['accuracy']:.3f}")

    results = {
        "run_id": run_id, "config": cfg, "environment": env,
        "temperature_T": float(T), "n_classes": int(pset.n_classes),
        "n_layers": int(per_sev_arrays[0]["H"].shape[0]),
        "split": {"n_calib": int(len(calib_idx)), "n_eval": int(len(eval_idx)),
                  "stratifie": True, "seed": args.seed},
        "elapsed_seconds": time.perf_counter() - started,
        "calibration": calibration,
        "per_severity": per_severity,
        "note": ("Expérience exploratoire : pas de seuil de verdict pré-enregistré. "
                 "Seuils et couche fixe calibrés sur la moitié CALIBRATION du split "
                 "test source (sévérité 0) ; toutes les évaluations portent sur la "
                 "moitié ÉVALUATION. Ancres textuelles R1 (ensemble d'invites)."),
    }
    (outdir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figures(per_severity, outdir / "figures", args)
    print(f"\nmétriques écrites : {outdir/'metrics.json'}")
    return results


def make_figures(per_severity: dict, figdir: Path, args) -> None:
    """Exactitude et profondeur moyenne par sévérité, pour chaque règle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sevs = sorted(int(s) for s in per_severity)
    series = {
        "H-stop": ("#B04A22", "s--", "H_stop"),
        "F-stop": ("#0B6B7A", "o-", "F_stop"),
        "pleine profondeur": ("#59627a", "d:", "full_depth"),
        "couche fixe": ("#8a6bbf", "^-.", "fixed_layer"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for label, (color, style, key) in series.items():
        axes[0].plot(sevs, [per_severity[str(s)][key]["accuracy"] for s in sevs],
                     style, color=color, label=label)
        if key in ("H_stop", "F_stop"):
            axes[1].plot(sevs, [per_severity[str(s)][key]["mean_depth"] for s in sevs],
                         style, color=color, label=label)
    axes[0].set_title("Exactitude (split évaluation)")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_title("Profondeur moyenne de sortie N*")
    axes[1].set_ylim(0.0, 12.5)
    for ax in axes:
        ax.set_xlabel("sévérité")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Arrêt anticipé calibré — {args.dataset} · {args.corruption}", y=1.04)
    fig.tight_layout()
    fig.savefig(figdir / "01_arret_anticipe.png", dpi=150, bbox_inches="tight")
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
    p.add_argument("--tolerance", type=float, default=0.01,
                   help="perte d'exactitude admise à la calibration")
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
