"""Analyse statistique a posteriori des runs de l'expérience 1.

Pour chaque run réel (non dry-run), calcule par sévérité la différence
AUROC(F) - AUROC(H) avec test de DeLong apparié et IC bootstrap stratifié,
puis écrit ``analysis.json`` à côté de ``metrics.json``.

Usage
-----
    python -m experiments.exp01_analysis outputs
    python -m experiments.exp01_analysis outputs/<run_id> [--n-boot 10000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tlsc.eval.metrics import bootstrap_auroc_delta, delong_test


def analyse_run(run_dir: Path, n_boot: int, alpha: float, seed: int) -> dict | None:
    """Analyse un run ; retourne None s'il est inanalysable ou synthétique."""
    metrics_path, scores_path = run_dir / "metrics.json", run_dir / "scores.npz"
    if not metrics_path.is_file() or not scores_path.is_file():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("config", {}).get("dry_run"):
        return None

    scores = np.load(scores_path)
    severities = metrics["config"]["severities"]
    s0 = severities[0]
    per_severity: dict[str, dict] = {}
    for s in severities[1:]:
        y = np.concatenate([np.zeros(len(scores[f"F_{s0}"])), np.ones(len(scores[f"F_{s}"]))])
        f = np.concatenate([scores[f"F_{s0}"], scores[f"F_{s}"]])
        h = np.concatenate([scores[f"H_{s0}"], scores[f"H_{s}"]])
        delong = delong_test(y, f, h)
        delta, lo, hi = bootstrap_auroc_delta(y, f, h, n=n_boot, alpha=alpha, seed=seed)
        per_severity[str(s)] = {
            "auroc_F": delong["auroc_a"], "auroc_H": delong["auroc_b"],
            "delta_auroc": delta, "delta_ci_low": lo, "delta_ci_high": hi,
            "delong_z": delong["z"], "delong_p": delong["p_value"],
        }

    analysis = {
        "run_id": metrics["run_id"],
        "dataset": metrics["config"]["dataset"],
        "corruption": metrics["config"]["corruption"],
        "corruption_source": metrics["config"]["corruption_source"],
        "seed": metrics["config"]["seed"],
        "n_boot": n_boot, "alpha": alpha, "bootstrap_seed": seed,
        "per_severity": per_severity,
    }
    (run_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return analysis


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="répertoire outputs ou répertoire d'un run précis")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.path)
    run_dirs = [root] if (root / "metrics.json").is_file() else sorted(
        d for d in root.iterdir() if d.is_dir()
    )
    analysed = 0
    for run_dir in run_dirs:
        result = analyse_run(run_dir, args.n_boot, args.alpha, args.seed)
        if result is None:
            continue
        analysed += 1
        smax = max(result["per_severity"], key=int)
        stats = result["per_severity"][smax]
        print(f"{result['run_id']} · {result['dataset']} · {result['corruption']} · "
              f"seed={result['seed']} · sév.{smax} : ΔAUROC={stats['delta_auroc']:+.3f} "
              f"IC95%=[{stats['delta_ci_low']:+.3f}, {stats['delta_ci_high']:+.3f}] "
              f"p={stats['delong_p']:.2e}")
    if analysed == 0:
        raise SystemExit("aucun run réel analysable trouvé")


if __name__ == "__main__":
    main()
