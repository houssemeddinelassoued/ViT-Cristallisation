"""Tests des métriques rapportées par le protocole expérimental."""
from __future__ import annotations

import numpy as np
import pytest

from tlsc.eval.metrics import (
    bootstrap_auroc_delta,
    brier_score,
    delong_test,
    diagnostic_metrics,
    paired_bootstrap_ci,
    sensitivity_at_specificity,
)


def test_predictions_parfaites() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]])
    metrics = diagnostic_metrics(y, p)
    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["sensitivity_at_95_specificity"] == 1.0
    assert metrics["brier"] == pytest.approx(0.01)


def test_brier_multiclasse_nul_pour_prediction_exacte() -> None:
    y = np.array([0, 2, 1])
    p = np.eye(3)[y]
    assert brier_score(y, p) == 0.0


def test_specificite_cible_validee() -> None:
    with pytest.raises(ValueError):
        sensitivity_at_specificity(np.array([0, 1]), np.array([0.0, 1.0]), 1.1)


def test_bootstrap_apparie_reproductible() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    first = paired_bootstrap_ci(a, b, n=2050, seed=7)
    second = paired_bootstrap_ci(a, b, n=2050, seed=7)
    assert first == second == (1.0, 1.0, 1.0)


def test_delong_scores_identiques() -> None:
    rng = np.random.default_rng(3)
    y = np.array([0] * 50 + [1] * 50)
    s = rng.normal(size=100) + y
    out = delong_test(y, s, s)
    assert out["delta"] == 0.0
    assert out["p_value"] == 1.0


def test_delong_detecte_un_detecteur_superieur() -> None:
    rng = np.random.default_rng(4)
    y = np.array([0] * 200 + [1] * 200)
    fort = y * 2.0 + rng.normal(size=400) * 0.4
    aleatoire = rng.normal(size=400)
    out = delong_test(y, fort, aleatoire)
    assert out["auroc_a"] > 0.95 > 0.7 > out["auroc_b"]
    assert out["p_value"] < 1e-6


def test_delong_exige_les_deux_classes() -> None:
    with pytest.raises(ValueError):
        delong_test(np.ones(4), np.arange(4.0), np.arange(4.0))


def test_bootstrap_delta_auroc_reproductible_et_coherent() -> None:
    rng = np.random.default_rng(5)
    y = np.array([0] * 120 + [1] * 120)
    fort = y * 2.0 + rng.normal(size=240) * 0.5
    faible = y * 0.2 + rng.normal(size=240)
    first = bootstrap_auroc_delta(y, fort, faible, n=500, seed=11)
    second = bootstrap_auroc_delta(y, fort, faible, n=500, seed=11)
    assert first == second
    observed, lo, hi = first
    assert lo <= observed <= hi
    assert lo > 0.0  # le détecteur fort domine avec un IC excluant zéro