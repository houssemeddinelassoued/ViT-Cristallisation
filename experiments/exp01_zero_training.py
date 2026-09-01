"""Expérience 1 — détection du décalage d'acquisition, sans aucun entraînement.

Prédiction à falsifier, issue du corollaire de la Proposition 2 : sous décalage
croissant, l'énergie libre F sépare le domaine source du domaine corrompu tandis
que l'entropie H échoue. Autrement dit ``AUROC(F) >> AUROC(H) ~ 0,5``.

Si la prédiction ne tient pas, le résultat se consigne tel quel : c'est une
réfutation de l'axe, pas un défaut d'implémentation.

Usage
-----
    python -m experiments.exp01_zero_training --dataset breastmnist
    python -m experiments.exp01_zero_training --dry-run     # chaîne complète, sans modèle
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from tlsc.core.gibbs import observables
from tlsc.eval.metrics import (
    diagnostic_metrics,
    expected_calibration_error,
    shift_detection_auroc,
)

VERDICT_CRITERIA = {
    "min_auroc_F": 0.70,
    "max_auroc_H": 0.60,
    "min_auroc_gap": 0.05,
}


def validate_args(args: argparse.Namespace) -> None:
    """Valide les paramètres qui déterminent l'interprétation scientifique."""
    severities = list(args.severities)
    if not severities or severities[0] != 0:
        raise ValueError("les sévérités doivent commencer par 0, le domaine source")
    if severities != sorted(set(severities)):
        raise ValueError("les sévérités doivent être uniques et strictement croissantes")
    if any(severity not in range(6) for severity in severities):
        raise ValueError("chaque sévérité doit appartenir à l'intervalle [0, 5]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size doit être strictement positif")
    if args.limit < 0:
        raise ValueError("--limit doit être positif ou nul")


# ── traçabilité ──────────────────────────────────────────────────────────────
def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "non-versionné"


def git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).strip())
    except Exception:
        return None


def source_sha256() -> str:
    """Empreinte du code Python et de la configuration de dépendances du run."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "experiments").rglob("*.py"))
    paths += sorted((root / "tlsc").rglob("*.py"))
    paths += [root / "pyproject.toml", root / "requirements.txt"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_run_id(cfg: dict) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return f"{stamp}_{cfg['dataset']}_{h}"


# ── encodage ─────────────────────────────────────────────────────────────────
def encode_cohort(clip, cohort, batch_size: int) -> torch.Tensor:
    """Encode une cohorte en embeddings normalisés (N, d)."""
    out = []
    for i in range(0, len(cohort.images), batch_size):
        batch = cohort.images[i:i + batch_size]
        px = torch.stack([clip.preprocess(im) for im in batch])
        out.append(clip.encode(px).cpu())
    return torch.cat(out)


def synthetic_cohort(n: int, d: int, K: int, severity: int, anchors: torch.Tensor,
                     seed: int) -> tuple[torch.Tensor, np.ndarray]:
    """Embeddings synthétiques pour le mode --dry-run.

    La sévérité éloigne progressivement les embeddings de leur ancre et les
    rapproche d'une direction parasite commune. Cela permet de valider la chaîne
    de bout en bout SANS modèle ni données. Les sorties sont marquées
    ``dry_run: true`` et ne constituent en aucun cas un résultat.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, K, (n,), generator=g)
    drift = torch.nn.functional.normalize(torch.randn(d, generator=g), dim=-1)
    alpha = severity / 5.0
    base = anchors[y]
    noise = torch.randn(n, d, generator=g) * (0.35 + 0.55 * alpha)
    z = (1.0 - 0.55 * alpha) * base + noise + 1.4 * alpha * drift
    return torch.nn.functional.normalize(z, dim=-1), y.numpy()


# ── expérience ───────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    started = time.perf_counter()
    cfg = {
        "experiment": "exp01_zero_training",
        "dataset": args.dataset, "model": args.model, "pretrained": args.pretrained,
        "corruption": args.corruption, "corruption_source": args.source,
        "severities": list(args.severities), "prompt_ensemble": not args.no_ensemble,
        "template_index": args.template_index,
        "image_size": args.size, "batch_size": args.batch_size, "seed": args.seed,
        "limit": args.limit, "dry_run": args.dry_run,
        "verdict_criteria": VERDICT_CRITERIA,
    }
    run_id = make_run_id(cfg)
    outdir = Path(args.outdir) / run_id
    (outdir / "figures").mkdir(parents=True, exist_ok=True)

    env = {
        "git_sha": git_sha(), "git_dirty": git_dirty(),
        "source_sha256": source_sha256(), "python": sys.version.split()[0],
        "torch": torch.__version__, "platform": platform.platform(),
        "device": args.device, "cuda_available": torch.cuda.is_available(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    print(f"run_id : {run_id}\nsortie : {outdir}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── modèle et ancres ─────────────────────────────────────────────────
    if args.dry_run:
        d, K, T = 512, 2, 0.02
        g = torch.Generator().manual_seed(args.seed)
        anchors = torch.nn.functional.normalize(torch.randn(K, d, generator=g), dim=-1)
        prompts_used = ["<dry-run : ancres aléatoires, aucun sens sémantique>"]
        clip = None
    else:
        from tlsc.models.clip_anchors import PROMPTS, FrozenCLIP, PromptSet

        if args.dataset not in PROMPTS:
            raise SystemExit(
                f"Aucun jeu d'invites défini pour '{args.dataset}'. "
                f"Ajoutez-le dans tlsc/models/clip_anchors.py (jeux connus : "
                f"{', '.join(PROMPTS)})."
            )
        pset = PROMPTS[args.dataset]
        if args.template_index is not None:
            if not 0 <= args.template_index < len(pset.templates):
                raise SystemExit(
                    f"--template-index doit appartenir à [0, {len(pset.templates) - 1}] "
                    f"pour '{args.dataset}'"
                )
            pset = PromptSet(name=pset.name, classes=pset.classes,
                             templates=[pset.templates[args.template_index]])
        clip = FrozenCLIP(args.model, args.pretrained, args.device)
        anchors = clip.anchors(pset, ensemble=not args.no_ensemble).cpu()
        T, K = clip.temperature, pset.n_classes
        prompts_used = [p for row in pset.phrases() for p in row]
        print(f"température de Gibbs T = 2*tau = {T:.6f}")

    # ── balayage de sévérité ─────────────────────────────────────────────
    per_sev: dict[int, dict] = {}
    scores: dict[int, dict[str, np.ndarray]] = {}

    for s in args.severities:
        severity_started = time.perf_counter()
        if args.dry_run:
            z, y = synthetic_cohort(args.limit or 800, anchors.shape[1], K, s, anchors, args.seed)
            corruption_label = "dry-run"
        else:
            from tlsc.data.medmnistc import build

            coh = build(args.dataset, severity=s, corruption=args.corruption,
                        source=args.source, size=args.size, root=args.root, seed=args.seed)
            if args.limit:
                coh.images, coh.labels = coh.images[:args.limit], coh.labels[:args.limit]
            z, y = encode_cohort(clip, coh, args.batch_size), coh.labels
            corruption_label = coh.corruption

        obs = observables(z, anchors, T)
        p = obs["p"].numpy()
        m = diagnostic_metrics(y, p)
        m |= {
            "ece": expected_calibration_error(y, p),
            "H_mean": float(obs["H"].mean()), "H_std": float(obs["H"].std()),
            "F_mean": float(obs["F"].mean()), "F_std": float(obs["F"].std()),
            "E_mean": float(obs["E"].mean()), "chi_mean": float(obs["chi"].mean()),
            "n": int(len(y)), "corruption": corruption_label,
            "elapsed_seconds": time.perf_counter() - severity_started,
        }
        per_sev[s] = m
        scores[s] = {"H": obs["H"].numpy(), "F": obs["F"].numpy()}
        print(f"  sévérité {s} · n={m['n']:5d} · acc={m['accuracy']:.3f} · "
              f"H={m['H_mean']:.4f} · F={m['F_mean']:.4f} · chi={m['chi_mean']:.3f}")

    # ── détection du décalage : source contre chaque sévérité ────────────
    s0 = 0
    detection = {}
    for s in args.severities[1:]:
        detection[s] = {
            "auroc_F": shift_detection_auroc(scores[s0]["F"], scores[s]["F"]),
            "auroc_H": shift_detection_auroc(scores[s0]["H"], scores[s]["H"]),
        }
        print(f"  détection sév. {s0} vs {s} : AUROC(F)={detection[s]['auroc_F']:.3f} · "
              f"AUROC(H)={detection[s]['auroc_H']:.3f}")

    smax = args.severities[-1]
    verdict = {
        "auroc_F_max_severity": detection[smax]["auroc_F"] if detection else None,
        "auroc_H_max_severity": detection[smax]["auroc_H"] if detection else None,
    }
    if detection:
        auroc_F = detection[smax]["auroc_F"]
        auroc_H = detection[smax]["auroc_H"]
        verdict |= {
            "F_detecte_decalage": bool(auroc_F >= VERDICT_CRITERIA["min_auroc_F"]),
            "H_proche_hasard": bool(auroc_H <= VERDICT_CRITERIA["max_auroc_H"]),
            "F_domine_H": bool(auroc_F - auroc_H >= VERDICT_CRITERIA["min_auroc_gap"]),
        }
        verdict["prediction_confirmee"] = all(
            verdict[key] for key in ("F_detecte_decalage", "H_proche_hasard", "F_domine_H")
        )

    results = {
        "run_id": run_id, "config": cfg, "environment": env,
        "temperature_T": float(T), "n_classes": int(K),
        "elapsed_seconds_before_artifacts": time.perf_counter() - started,
        "prompts": prompts_used,
        "per_severity": {str(k): v for k, v in per_sev.items()},
        "shift_detection": {str(k): v for k, v in detection.items()},
        "verdict": verdict,
        "avertissement": (
            "MODE DRY-RUN : embeddings synthétiques, aucune valeur ci-dessus n'est "
            "un résultat expérimental." if args.dry_run else None
        ),
    }
    (outdir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(outdir / "scores.npz",
                        **{f"{k}_{s}": v for s, d_ in scores.items() for k, v in d_.items()})
    make_figures(per_sev, detection, scores, outdir / "figures", args)
    print(f"\nmétriques écrites : {outdir/'metrics.json'}")
    return results


def make_figures(per_sev, detection, scores, figdir: Path, args) -> None:
    """Trois figures : trajectoires, pouvoir de détection, distributions de F."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cold, hot, ink = "#0B6B7A", "#B04A22", "#0E1620"
    sev = sorted(per_sev)

    def tidy(a) -> None:
        a.grid(alpha=0.25)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    # 1 — trajectoires des trois observables
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(sev, [per_sev[s]["accuracy"] for s in sev], "o-", color=ink)
    ax[0].set_title("Exactitude")
    ax[1].plot(sev, [per_sev[s]["H_mean"] for s in sev], "o-", color=hot)
    ax[1].set_title("Entropie moyenne")
    ax[2].plot(sev, [per_sev[s]["F_mean"] for s in sev], "o-", color=cold)
    ax[2].set_title("Énergie libre moyenne")
    for a in ax:
        a.set_xlabel("sévérité du décalage")
        tidy(a)
    fig.suptitle(f"Trajectoires sous décalage — {args.dataset}", y=1.04)
    fig.tight_layout()
    fig.savefig(figdir / "01_trajectoires.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2 — pouvoir de détection : la figure qui teste la Proposition 2
    if detection:
        ds = sorted(detection)
        fig, a = plt.subplots(figsize=(5.2, 3.8))
        a.plot(ds, [detection[s]["auroc_F"] for s in ds], "o-", color=cold,
               label="énergie libre F")
        a.plot(ds, [detection[s]["auroc_H"] for s in ds], "s--", color=hot,
               label="entropie H")
        a.axhline(0.5, color="#999999", lw=1, ls=":", label="hasard")
        a.set_ylim(0.4, 1.02)
        a.set_xlabel("sévérité du décalage")
        a.set_ylabel("AUROC de détection")
        a.set_title("Pouvoir de détection du décalage")
        a.legend(frameon=False)
        tidy(a)
        fig.tight_layout()
        fig.savefig(figdir / "02_detection.png", dpi=150)
        plt.close(fig)

    # 3 — distributions de l'énergie libre, source contre cible
    smax = sev[-1]
    fig, a = plt.subplots(figsize=(5.2, 3.8))
    a.hist(scores[sev[0]]["F"], bins=40, alpha=0.65, color=cold,
           label=f"source (sévérité {sev[0]})")
    a.hist(scores[smax]["F"], bins=40, alpha=0.65, color=hot,
           label=f"corrompu (sévérité {smax})")
    a.set_xlabel("énergie libre F")
    a.set_ylabel("effectif")
    a.set_title("Distribution de l'énergie libre")
    a.legend(frameon=False)
    tidy(a)
    fig.tight_layout()
    fig.savefig(figdir / "03_distribution_F.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="breastmnist")
    p.add_argument("--model", default="ViT-B-16-quickgelu")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--corruption", default="speckle")
    p.add_argument("--source", default="local", choices=["local", "medmnistc"],
                   help="'medmnistc' pour les corruptions officielles ; 'local' pour le repli")
    p.add_argument("--severities", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0, help="0 = tout le split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ensemble", action="store_true", help="une seule invite par classe")
    p.add_argument("--template-index", type=int, default=None,
                   help="restreint les ancres à une seule formulation (sensibilité aux invites)")
    p.add_argument("--root", default="data/raw")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--dry-run", action="store_true",
                   help="valide la chaîne avec des embeddings synthétiques, sans modèle")
    args = p.parse_args()
    try:
        run(args)
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
