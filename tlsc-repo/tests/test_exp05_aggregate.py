"""Tests de l'agrégation multi-graines de l'expérience 5."""
from __future__ import annotations

import json

import pytest

from experiments.exp05_aggregate import agrege, collecte, raison_rejet

REGIMES = ("R1_texte", "R2_fewshot", "R3_centroides")


def _bloc(oracle: int, bal: list[float], fixe: int, auroc_f: float) -> dict:
    """Bloc de régime minimal, aux seules clés lues par l'agrégation."""
    return {
        "anchor_quality_par_couche": [{"Q_gap": 0.2, "layer": i + 1} for i in range(len(bal))],
        "calibration": {"fixed_layer": {"layer": fixe}},
        "per_severity": {
            "0": {"balanced_accuracy_par_couche": bal, "oracle_layer_balanced": oracle,
                  "fixed_layer": {"balanced_accuracy": 0.8},
                  "full_depth": {"balanced_accuracy": 0.7}},
            "5": {"balanced_accuracy_par_couche": bal, "oracle_layer_balanced": oracle,
                  "fixed_layer": {"balanced_accuracy": 0.75},
                  "full_depth": {"balanced_accuracy": 0.55}},
        },
        "depth_detection": {
            "5": {"F": {"auroc": auroc_f, "auroc_two_sided": 0.9},
                  "H": {"auroc": 0.4, "auroc_two_sided": 0.85}},
        },
    }


def _run(tmp_path, nom: str, *, seed: int = 0, dirty: bool = False,
         critere: str = "balanced_accuracy", oracle: int = 9, auroc_f: float = 1.0,
         experiment: str = "exp05_anchored_early_exit", metrics: bool = True,
         bilateral: bool = True):
    d = tmp_path / nom
    d.mkdir()
    if metrics:
        bal = [0.7, 0.75, 0.8, 0.84, 0.71]
        regimes = {r: _bloc(oracle, bal, 9, auroc_f) for r in REGIMES}
        if not bilateral:
            for r in regimes.values():
                for s in r["depth_detection"].values():
                    for obs in ("F", "H"):
                        s[obs].pop("auroc_two_sided", None)
        (d / "metrics.json").write_text(json.dumps({
            "run_id": nom,
            "config": {"experiment": experiment, "dataset": "pneumoniamnist",
                       "corruption": "gaussian_blur", "seed": seed},
            "environment": {"git_dirty": dirty},
            "critere_calibration": critere,
            "regimes": regimes,
        }), encoding="utf-8")
    return d


def test_run_propre_retenu(tmp_path) -> None:
    assert raison_rejet(_run(tmp_path, "bon")) is None


def test_run_sali_ecarte(tmp_path) -> None:
    assert raison_rejet(_run(tmp_path, "sali", dirty=True)) == "git_dirty"


def test_run_au_critere_brut_ecarte(tmp_path) -> None:
    """Les runs calibrés sur l'exactitude brute ne sont pas comparables."""
    assert raison_rejet(_run(tmp_path, "brut", critere="accuracy")) == "critere_brut"


def test_autre_experience_ignoree(tmp_path) -> None:
    d = _run(tmp_path, "exp01", experiment="exp01_zero_training")
    assert raison_rejet(d) == "autre_experience"


def test_run_incomplet_ecarte(tmp_path) -> None:
    assert raison_rejet(_run(tmp_path, "vide", metrics=False)) == "incomplet"


def test_collecte_separe_retenus_et_ecartes(tmp_path) -> None:
    _run(tmp_path, "a", seed=0)
    _run(tmp_path, "b", seed=1, dirty=True)
    _run(tmp_path, "c", seed=2, critere="accuracy")
    _run(tmp_path, "d", experiment="exp01_zero_training")
    retenus, ecartes = collecte(tmp_path)
    assert len(retenus) == 1
    assert set(ecartes) == {"git_dirty", "critere_brut"}
    assert "autre_experience" not in ecartes  # ignoree, pas ecartee


def test_agregation_sur_plusieurs_graines(tmp_path) -> None:
    _run(tmp_path, "s0", seed=0, oracle=9, auroc_f=1.0)
    _run(tmp_path, "s1", seed=1, oracle=8, auroc_f=0.8)
    _run(tmp_path, "s2", seed=2, oracle=10, auroc_f=0.9)
    retenus, _ = collecte(tmp_path)
    entrees = agrege(retenus)
    assert len(entrees) == len(REGIMES)
    e = entrees[0]
    assert e["seeds"] == [0, 1, 2]
    assert e["couche_oracle"]["moyenne"] == pytest.approx(9.0)
    assert e["couche_oracle"]["min"] == 8 and e["couche_oracle"]["max"] == 10
    assert e["couche_oracle"]["n_graines"] == 3
    assert e["auroc_Nstar_F_sevmax"]["moyenne"] == pytest.approx(0.9)
    assert e["auroc_Nstar_F_sevmax"]["ecart_type"] > 0.0


def test_gain_de_la_couche_fixe(tmp_path) -> None:
    """Le gain est l'écart entre couche fixe et pleine profondeur à sévérité max."""
    _run(tmp_path, "s0", seed=0)
    retenus, _ = collecte(tmp_path)
    e = agrege(retenus)[0]
    assert e["gain_couche_fixe_sevmax"]["moyenne"] == pytest.approx(0.75 - 0.55)


def test_ecart_type_nul_sur_une_seule_graine(tmp_path) -> None:
    _run(tmp_path, "s0", seed=0)
    retenus, _ = collecte(tmp_path)
    e = agrege(retenus)[0]
    assert e["couche_oracle"]["ecart_type"] == 0.0
    assert e["couche_oracle"]["n_graines"] == 1


def test_run_sans_detecteur_bilateral_ecarte(tmp_path) -> None:
    """Les runs anterieurs a l'ajout du bilateral sur N* sont remplaces, pas ignores."""
    d = _run(tmp_path, "ancien", bilateral=False)
    assert raison_rejet(d) == "sans_bilateral"


def test_collecte_nomme_le_motif_sans_bilateral(tmp_path) -> None:
    _run(tmp_path, "neuf", seed=0)
    _run(tmp_path, "ancien", seed=0, bilateral=False)
    retenus, ecartes = collecte(tmp_path)
    assert len(retenus) == 1
    assert ecartes["sans_bilateral"] == ["ancien"]
