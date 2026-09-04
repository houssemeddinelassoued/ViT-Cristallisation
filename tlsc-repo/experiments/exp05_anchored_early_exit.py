"""Expérience 5 — arrêt anticipé calibré sous les trois régimes d'ancrage.

Croisement des chantiers exp03 (régimes d'ancrage) et exp04 (arrêt anticipé).

Motivation. L'exp04 a mesuré la règle « sortir quand H < epsilon » sous ancres
textuelles R1, et l'a trouvée dégénérée : la couche 1 suffisait déjà, non pas
parce que la cristallisation y était acquise, mais parce que les ancres
textuelles ne classent presque rien sur ce substrat (exactitude équilibrée
proche du hasard, exp01). L'exp03 a ensuite montré que quelques dizaines
d'étiquettes source suffisent à rendre la classification réelle. Le test non
trivial de « argmin_N H » exige donc de rejouer l'arrêt sous ancres R2/R3, où
l'exactitude varie effectivement avec la profondeur.

Trois questions, toutes exploratoires.

Q1  Sous ancres R2/R3, la profondeur optimale cesse-t-elle d'être N* = 1 ?
Q2  La profondeur de sortie est-elle un détecteur de décalage utilisable
    (AUROC des N* source contre cible) ? L'exp04 avait observé la montée de la
    profondeur moyenne avec la sévérité sans la quantifier.
Q3  Le critère de qualité d'ancrage (D_inter rapportée à la dispersion,
    ``tlsc.models.data_anchors.anchor_quality``) classe-t-il les régimes dans
    l'ordre de leur utilité réelle, là où D_inter seule les classait à l'envers ?

Ancres par couche. La sonde logit lens produit un embedding à chaque
profondeur ; les centroïdes R2/R3 sont donc recalculés couche par couche sur le
split TRAIN source. Les ancres textuelles R1 sont les mêmes à toutes les
profondeurs — c'est précisément l'hypothèse que le logit lens met à l'épreuve.

Protocole anti-fuite, en trois barrières :
1. centroïdes R2/R3 : split TRAIN source, sévérité 0 uniquement ;
2. seuils epsilon et couche fixe : moitié CALIBRATION du split TEST source ;
3. toutes les évaluations, source comme cible : moitié ÉVALUATION seulement.

Caractère EXPLORATOIRE déclaré : aucun seuil de verdict pré-enregistré.

Usage
-----
    python -m experiments.exp05_anchored_early_exit --dataset breastmnist \
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
from experiments.exp04_early_exit import layer_scores_and_correct, stratified_split
from tlsc.data.medmnistc import build
from tlsc.eval.early_exit import calibrate_epsilon, exit_depth_auroc, simulate_early_exit
from tlsc.models.data_anchors import anchor_quality, class_centroids

REGIMES = ("R1_texte", "R2_fewshot", "R3_centroides")
RULES = ("H", "F")


def validate_args(args: argparse.Namespace) -> None:
    severities = list(args.severities)
    if not severities or severities[0] != 0:
        raise ValueError("les sévérités doivent commencer par 0, le domaine source")
    if severities != sorted(set(severities)):
        raise ValueError("les sévérités doivent être uniques et strictement croissantes")
    if any(severity not in range(6) for severity in severities):
        raise ValueError("chaque sévérité doit appartenir à l'intervalle [0, 5]")
    if len(severities) < 2:
        raise ValueError("au moins une sévérité cible est requise en plus de la source")
    if args.batch_size <= 0:
        raise ValueError("--batch-size doit être strictement positif")
    if not 0.0 <= args.tolerance < 1.0:
        raise ValueError("--tolerance doit appartenir à [0, 1[")
    if args.k_shot <= 0:
        raise ValueError("--k-shot doit être strictement positif")
    if args.train_limit < 0:
        raise ValueError("--train-limit doit être positif ou nul")


def subsample_stratified(y: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Indices d'un sous-échantillon stratifié de taille ~limit (0 = tout).

    Parameters
    ----------
    y : ndarray, shape (N,) — étiquettes entières.
    limit : int — taille visée ; 0 ou >= N retourne tous les indices.
    seed : int — graine du tirage.

    Returns
    -------
    ndarray d'indices triés.
    """
    if limit <= 0 or limit >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        rng.shuffle(idx)
        keep.append(idx[:max(1, round(limit * len(idx) / len(y)))])
    return np.sort(np.concatenate(keep))


def build_layer_anchors(z_train_layers: torch.Tensor, y_train: torch.Tensor,
                        text_anchors: torch.Tensor, k_shot: int, seed: int,
                        ) -> tuple[dict[str, torch.Tensor], dict[str, list[dict]]]:
    """Ancres par couche pour les trois régimes, et leur qualité géométrique.

    Parameters
    ----------
    z_train_layers : Tensor, shape (n_layers, N, d)
        Embeddings TRAIN source par couche, L2-normalisés.
    y_train : Tensor, shape (N,)
    text_anchors : Tensor, shape (K, d) — ancres R1, communes à toutes les couches.
    k_shot : int — exemples par classe pour R2.
    seed : int

    Returns
    -------
    anchors : dict régime -> Tensor (n_layers, K, d)
    quality : dict régime -> liste par couche des dicts de qualité d'ancrage,
        évaluée sur le nuage TRAIN source de la couche correspondante.
    """
    n_layers = z_train_layers.shape[0]
    anchors: dict[str, list[torch.Tensor]] = {name: [] for name in REGIMES}
    quality: dict[str, list[dict]] = {name: [] for name in REGIMES}
    for n in range(n_layers):
        z_n = z_train_layers[n]
        per_layer = {
            "R1_texte": text_anchors,
            "R2_fewshot": class_centroids(z_n, y_train, k_shot=k_shot, seed=seed),
            "R3_centroides": class_centroids(z_n, y_train),
        }
        for name, mu in per_layer.items():
            anchors[name].append(mu)
            quality[name].append(anchor_quality(z_n, mu, y_train) | {"layer": n + 1})
    return {name: torch.stack(mus) for name, mus in anchors.items()}, quality


def scores_par_regime(z_layers: torch.Tensor, anchors: dict[str, torch.Tensor],
                      T: float, y: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """H, F et exactitude par couche, pour chaque régime d'ancrage.

    Chaque couche est lue avec les ancres de SA couche : ``anchors[name][n]``.

    Returns
    -------
    dict régime -> dict avec les clés ``H``, ``F``, ``correct``, toutes de
    forme (n_layers, n).
    """
    out = {}
    for name, mu_layers in anchors.items():
        blocks = [
            layer_scores_and_correct(z_layers[n:n + 1], mu_layers[n], T, y)
            for n in range(z_layers.shape[0])
        ]
        out[name] = {
            k: np.concatenate([b[k] for b in blocks]) for k in ("H", "F", "correct")
        }
    return out


def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    started = time.perf_counter()
    cfg = {
        "experiment": "exp05_anchored_early_exit",
        "dataset": args.dataset, "model": args.model, "pretrained": args.pretrained,
        "corruption": args.corruption, "corruption_source": args.source,
        "severities": list(args.severities), "tolerance": args.tolerance,
        "k_shot": args.k_shot, "train_limit": args.train_limit,
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
    text_anchors = clip.anchors(pset, ensemble=True).cpu()
    T = clip.temperature
    print(f"temperature de Gibbs T = 2*tau = {T:.6f}")

    # -- barriere 1 : ancres par couche depuis le TRAIN source, severite 0 --
    train = build(args.dataset, severity=0, corruption=args.corruption,
                  source=args.source, split="train", size=args.size,
                  root=args.root, seed=args.seed)
    keep = subsample_stratified(np.asarray(train.labels), args.train_limit, args.seed)
    train.images = [train.images[i] for i in keep]
    train.labels = np.asarray(train.labels)[keep]
    z_train_layers = encode_cohort_layers(clip, train, args.batch_size)
    y_train = torch.as_tensor(train.labels, dtype=torch.long)
    print(f"train source encode par couche : n={len(y_train)} | "
          f"{z_train_layers.shape[0]} couches")

    anchors, quality = build_layer_anchors(z_train_layers, y_train, text_anchors,
                                           args.k_shot, args.seed)
    n_train_source = int(len(y_train))
    del z_train_layers
    for name in REGIMES:
        q12 = quality[name][-1]
        print(f"  qualite[{name:14s}] couche 12 : D_inter={q12['D_inter']:.4f} "
              f"d_cloud={q12['d_cloud']:.4f} Q_gap={q12['Q_gap']:.4f} "
              f"Q_fisher={q12['Q_fisher']:.4f}")

    # -- encodage du split test, toutes severites --------------------------
    per_sev: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    y_ref: np.ndarray | None = None
    for s in args.severities:
        coh = build(args.dataset, severity=s, corruption=args.corruption,
                    source=args.source, size=args.size, root=args.root, seed=args.seed)
        if args.limit:
            coh.images, coh.labels = coh.images[:args.limit], coh.labels[:args.limit]
        z_layers = encode_cohort_layers(clip, coh, args.batch_size)
        per_sev[s] = scores_par_regime(z_layers, anchors, T, coh.labels)
        if s == 0:
            y_ref = coh.labels
        del z_layers
        print(f"  severite {s} encodee | n={len(coh.labels)}")

    # -- barriere 2 : calibration sur la moitie source ----------------------
    calib_idx, eval_idx = stratified_split(y_ref, args.seed)
    n_layers = per_sev[0][REGIMES[0]]["H"].shape[0]
    per_regime: dict[str, dict] = {}

    for name in REGIMES:
        src = per_sev[0][name]
        calibration: dict[str, dict] = {}
        for rule in RULES:
            calibration[rule] = calibrate_epsilon(
                src[rule][:, calib_idx], src["correct"][:, calib_idx],
                tolerance=args.tolerance)
        acc_calib = src["correct"][:, calib_idx].mean(axis=1)
        fixed_layer = int(acc_calib.argmax())
        calibration["fixed_layer"] = {
            "layer": fixed_layer + 1,
            "accuracy_calib": float(acc_calib[fixed_layer]),
        }
        calibration["accuracy_par_couche_calib"] = [float(v) for v in acc_calib]

        # -- barriere 3 : evaluation, indices d'evaluation seulement --------
        per_severity: dict[str, dict] = {}
        for s in args.severities:
            arrays = per_sev[s][name]
            block: dict = {"n_eval": int(len(eval_idx))}
            for rule in RULES:
                block[f"{rule}_stop"] = simulate_early_exit(
                    arrays[rule][:, eval_idx], arrays["correct"][:, eval_idx],
                    calibration[rule]["epsilon"])
            block["full_depth"] = {
                "accuracy": float(arrays["correct"][-1, eval_idx].mean()),
                "mean_depth": float(n_layers)}
            block["fixed_layer"] = {
                "layer": fixed_layer + 1,
                "accuracy": float(arrays["correct"][fixed_layer, eval_idx].mean()),
                "mean_depth": float(fixed_layer + 1)}
            block["accuracy_par_couche"] = [
                float(v) for v in arrays["correct"][:, eval_idx].mean(axis=1)]
            block["oracle_layer"] = int(np.argmax(block["accuracy_par_couche"])) + 1
            per_severity[str(s)] = block

        # -- Q2 : la profondeur de sortie comme detecteur de decalage -------
        s0 = args.severities[0]
        depth_detection: dict[str, dict] = {}
        for s in args.severities[1:]:
            depth_detection[str(s)] = {
                rule: exit_depth_auroc(
                    per_sev[s0][name][rule][:, eval_idx],
                    per_sev[s][name][rule][:, eval_idx],
                    calibration[rule]["epsilon"])
                for rule in RULES
            }

        per_regime[name] = {
            "anchor_quality_par_couche": quality[name],
            "calibration": calibration,
            "per_severity": per_severity,
            "depth_detection": depth_detection,
        }
        smax = str(args.severities[-1])
        blk, det = per_severity[smax], depth_detection[smax]
        print(f"  {name:14s} : couche fixe={fixed_layer + 1} | oracle(sev0)="
              f"{per_severity['0']['oracle_layer']} | sev{smax} F-stop "
              f"acc={blk['F_stop']['accuracy']:.3f}@{blk['F_stop']['mean_depth']:.2f} "
              f"| AUROC(N*|F)={det['F']['auroc']:.3f}")

    results = {
        "run_id": run_id, "config": cfg, "environment": env,
        "temperature_T": float(T), "n_classes": int(pset.n_classes),
        "n_layers": int(n_layers), "n_train_source": n_train_source,
        "split": {"n_calib": int(len(calib_idx)), "n_eval": int(len(eval_idx)),
                  "stratifie": True, "seed": args.seed},
        "elapsed_seconds": time.perf_counter() - started,
        "regimes": per_regime,
        "note": ("Experience exploratoire : aucun seuil de verdict pre-enregistre. "
                 "Ancres R2/R3 recalculees couche par couche sur le split TRAIN "
                 "source a severite 0 ; seuils epsilon et couche fixe calibres sur "
                 "la moitie CALIBRATION du split test source ; toutes les "
                 "evaluations portent sur la moitie EVALUATION. Les couches "
                 "intermediaires n'ont jamais ete alignees a l'espace texte : la "
                 "lecture R1 en profondeur intermediaire reste un logit lens."),
    }
    (outdir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        outdir / "scores_layers_regimes.npz",
        **{f"{name}_{k}_{s}": v for s, per_name in per_sev.items()
           for name, kv in per_name.items() for k, v in kv.items()},
        eval_idx=eval_idx, calib_idx=calib_idx)
    make_figures(per_regime, args, outdir / "figures")
    print(f"\nmetriques ecrites : {outdir/'metrics.json'}")
    return results


def make_figures(per_regime: dict, args, figdir: Path) -> None:
    """Exactitude, profondeur de sortie, detecteur N* et qualite d'ancrage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"R1_texte": ("#B04A22", "o-"), "R2_fewshot": ("#8a6bbf", "s--"),
              "R3_centroides": ("#0B6B7A", "d-")}
    sevs = sorted(int(s) for s in per_regime[REGIMES[0]]["per_severity"])
    tgt = sevs[1:]

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    for name, block in per_regime.items():
        color, style = styles[name]
        ps = block["per_severity"]
        axes[0].plot(sevs, [ps[str(s)]["F_stop"]["accuracy"] for s in sevs],
                     style, color=color, label=name)
        axes[0].plot(sevs, [ps[str(s)]["full_depth"]["accuracy"] for s in sevs],
                     ":", color=color, alpha=0.5, lw=1.2)
        axes[1].plot(sevs, [ps[str(s)]["F_stop"]["mean_depth"] for s in sevs],
                     style, color=color, label=name)
        axes[2].plot(tgt, [block["depth_detection"][str(s)]["F"]["auroc"] for s in tgt],
                     style, color=color, label=name)
        layers = range(1, len(ps["0"]["accuracy_par_couche"]) + 1)
        axes[3].plot(layers, ps["0"]["accuracy_par_couche"], style, color=color,
                     label=name)

    axes[0].set_title("Exactitude : F-stop (trait) vs pleine profondeur (pointille)")
    axes[0].set_xlabel("severite")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_title("Profondeur moyenne de sortie N*")
    axes[1].set_xlabel("severite")
    axes[1].set_ylim(0.0, 12.5)
    axes[2].set_title("AUROC de N* comme detecteur de decalage")
    axes[2].set_xlabel("severite")
    axes[2].set_ylim(0.0, 1.03)
    axes[2].axhline(0.5, color="#999999", lw=1, ls=":")
    axes[3].set_title("Exactitude par couche (source, split evaluation)")
    axes[3].set_xlabel("couche")
    axes[3].set_ylim(0.0, 1.0)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Ancres x arret calibre — {args.dataset} · {args.corruption}", y=1.05)
    fig.tight_layout()
    fig.savefig(figdir / "01_ancres_x_arret.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for name, block in per_regime.items():
        color, style = styles[name]
        q = block["anchor_quality_par_couche"]
        layers = [d["layer"] for d in q]
        axes[0].plot(layers, [d["D_inter"] for d in q], style, color=color, label=name)
        axes[1].plot(layers, [d["Q_fisher"] for d in q], style, color=color, label=name)
    axes[0].set_title("D_inter seule (trompeuse)")
    axes[1].set_title("Q_fisher = D_inter / dispersion intra")
    for ax in axes:
        ax.set_xlabel("couche")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Qualite d'ancrage par couche — {args.dataset}", y=1.04)
    fig.tight_layout()
    fig.savefig(figdir / "02_qualite_ancrage.png", dpi=150, bbox_inches="tight")
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
                   help="perte d'exactitude admise a la calibration")
    p.add_argument("--k-shot", type=int, default=16,
                   help="exemples par classe pour les centroides R2")
    p.add_argument("--train-limit", type=int, default=0,
                   help="sous-echantillon stratifie du train source (0 = tout)")
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
