"""Agrégation multi-graines de l'expérience 5.

Les chiffres d'un run unique de l'exp05 ne portent aucune incertitude. Ce script
regroupe les runs citables par (cohorte, corruption, régime) et rapporte, sur les
graines disponibles, la moyenne, l'écart-type et l'étendue des grandeurs de tête :

- ``oracle_layer_balanced`` — la profondeur utile ; sa **dispersion** dit si
  l'optimum intérieur est un fait ou un artefact d'une graine ;
- l'exactitude équilibrée à cette couche, à la couche fixe calibrée et en pleine
  profondeur, à sévérité 0 et à sévérité maximale ;
- l'AUROC de la profondeur de sortie comme détecteur, unilatérale et bilatérale.

Un run n'entre dans l'agrégation que s'il est **complet**, **propre**
(``environment.git_dirty`` faux) et calibré sur l'exactitude **équilibrée** : les
runs antérieurs au 2026-09-05, calibrés sur l'exactitude brute, ne sont pas
comparables et sont écartés en le disant.

Usage
-----
    python -m experiments.exp05_aggregate outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

REGIMES = ("R1_texte", "R2_fewshot", "R3_centroides")


def lire_metrics(chemin: Path) -> dict:
    """Lit un ``metrics.json``, avec repli cp1252 pour les runs anciens."""
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(chemin.read_text(encoding="cp1252"))


def raison_rejet(run_dir: Path) -> str | None:
    """Motif d'exclusion d'un run de l'agrégation exp05, ou None s'il est retenu."""
    m = run_dir / "metrics.json"
    if not m.is_file():
        return "incomplet"
    j = lire_metrics(m)
    if j.get("config", {}).get("experiment") != "exp05_anchored_early_exit":
        return "autre_experience"
    if j.get("environment", {}).get("git_dirty"):
        return "git_dirty"
    if j.get("critere_calibration") != "balanced_accuracy":
        return "critere_brut"
    return None


def _stats(valeurs: list[float]) -> dict[str, float]:
    """Moyenne, écart-type, étendue et effectif d'une liste de mesures."""
    return {
        "moyenne": float(statistics.fmean(valeurs)),
        "ecart_type": float(statistics.stdev(valeurs)) if len(valeurs) > 1 else 0.0,
        "min": float(min(valeurs)),
        "max": float(max(valeurs)),
        "n_graines": len(valeurs),
    }


def collecte(root: Path) -> tuple[list[dict], dict[str, list[str]]]:
    """Runs exp05 retenus et dictionnaire des runs écartés par motif."""
    retenus, ecartes = [], defaultdict(list)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "aggregate"):
        motif = raison_rejet(d)
        if motif == "autre_experience":
            continue
        if motif is not None:
            ecartes[motif].append(d.name)
            continue
        retenus.append(lire_metrics(d / "metrics.json"))
    return retenus, dict(ecartes)


def agrege(runs: list[dict]) -> list[dict]:
    """Une entrée par (cohorte, corruption, régime), agrégée sur les graines."""
    groupes: dict[tuple[str, str, str], list[tuple[int, str, dict]]] = defaultdict(list)
    for m in runs:
        c = m["config"]
        for regime in REGIMES:
            groupes[(c["dataset"], c["corruption"], regime)].append(
                (c["seed"], m["run_id"], m["regimes"][regime]))

    sorties = []
    for (dataset, corruption, regime), entrees in sorted(groupes.items()):
        entrees.sort(key=lambda e: e[0])
        sorties.append(_agrege_groupe(
            dataset, corruption, regime,
            graines=[s for s, _, _ in entrees],
            identifiants=[r for _, r, _ in entrees],
            blocs=[b for _, _, b in entrees]))
    return sorties


def _agrege_groupe(dataset: str, corruption: str, regime: str, *,
                   graines: list[int], identifiants: list[str],
                   blocs: list[dict]) -> dict:
    """Agrège sur les graines les grandeurs d'un (cohorte, corruption, régime)."""
    smax = str(max((int(s) for s in blocs[0]["per_severity"]), default=0))
    sev0 = [b["per_severity"]["0"] for b in blocs]
    sevm = [b["per_severity"][smax] for b in blocs]
    det = [b["depth_detection"][smax] for b in blocs]

    entree = {
        "dataset": dataset, "corruption": corruption, "regime": regime,
        "seeds": graines, "run_ids": identifiants, "severite_max": int(smax),
        "couche_oracle": _stats([s["oracle_layer_balanced"] for s in sev0]),
        "couche_fixe": _stats([b["calibration"]["fixed_layer"]["layer"] for b in blocs]),
        "bal_oracle_sev0": _stats([max(s["balanced_accuracy_par_couche"]) for s in sev0]),
        "bal_couche12_sev0": _stats([s["balanced_accuracy_par_couche"][-1] for s in sev0]),
        "bal_couche_fixe_sevmax": _stats(
            [s["fixed_layer"]["balanced_accuracy"] for s in sevm]),
        "bal_pleine_prof_sevmax": _stats(
            [s["full_depth"]["balanced_accuracy"] for s in sevm]),
        "auroc_Nstar_F_sevmax": _stats([d["F"]["auroc"] for d in det]),
        "auroc_Nstar_F_bilateral_sevmax": _stats([d["F"]["auroc_two_sided"] for d in det]),
        "auroc_Nstar_H_sevmax": _stats([d["H"]["auroc"] for d in det]),
        "auroc_Nstar_H_bilateral_sevmax": _stats([d["H"]["auroc_two_sided"] for d in det]),
        "Q_gap_couche12": _stats(
            [b["anchor_quality_par_couche"][-1]["Q_gap"] for b in blocs]),
    }
    entree["gain_couche_fixe_sevmax"] = {
        "moyenne": entree["bal_couche_fixe_sevmax"]["moyenne"]
        - entree["bal_pleine_prof_sevmax"]["moyenne"]}
    return entree


def ecrit_csv(entrees: list[dict], chemin: Path) -> None:
    grandeurs = [k for k, v in entrees[0].items() if isinstance(v, dict) and "moyenne" in v]
    champs = ["dataset", "corruption", "regime", "n_graines"]
    for g in grandeurs:
        champs += [f"{g}_moyenne", f"{g}_ecart_type", f"{g}_min", f"{g}_max"]
    with chemin.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=champs)
        w.writeheader()
        for e in entrees:
            ligne = {"dataset": e["dataset"], "corruption": e["corruption"],
                     "regime": e["regime"], "n_graines": e["couche_oracle"]["n_graines"]}
            for g in grandeurs:
                for stat in ("moyenne", "ecart_type", "min", "max"):
                    ligne[f"{g}_{stat}"] = f"{e[g][stat]:.4f}"
            w.writerow(ligne)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", default="outputs")
    args = p.parse_args()

    root = Path(args.path).resolve()
    runs, ecartes = collecte(root)
    for motif, noms in sorted(ecartes.items()):
        print(f"ecartes ({motif}) : {len(noms)}")
        for n in noms:
            print(f"    {n}")
    if not runs:
        raise SystemExit("aucun run exp05 citable et calibre sur l'equilibree")

    entrees = agrege(runs)
    outdir = root / "aggregate"
    outdir.mkdir(exist_ok=True)
    (outdir / "exp05_summary.json").write_text(
        json.dumps(entrees, indent=2, ensure_ascii=False), encoding="utf-8")
    ecrit_csv(entrees, outdir / "exp05_summary.csv")

    graines = sorted({s for m in runs for s in [m["config"]["seed"]]})
    print(f"\n{len(runs)} runs exp05 agreges | graines : {graines}")
    print(f"{'cohorte/corruption':34s} {'regime':15s} "
          f"{'couche oracle':>16s} {'bal oracle sev0':>17s} {'gain sev max':>13s}")
    for e in entrees:
        o, b = e["couche_oracle"], e["bal_oracle_sev0"]
        print(f"{e['dataset'] + '/' + e['corruption']:34s} {e['regime']:15s} "
              f"{o['moyenne']:6.1f} +/- {o['ecart_type']:<5.1f} "
              f"{b['moyenne']:7.3f} +/- {b['ecart_type']:<5.3f} "
              f"{e['gain_couche_fixe_sevmax']['moyenne']:+12.3f}")
    print(f"\necrit dans {outdir}")


if __name__ == "__main__":
    main()
