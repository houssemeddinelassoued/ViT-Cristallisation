"""Tests de la règle d'arrêt anticipé par échantillon."""
from __future__ import annotations

import numpy as np
import pytest

from tlsc.eval.early_exit import calibrate_epsilon, exit_layers, simulate_early_exit


def test_exit_layers_premiere_couche_sous_le_seuil() -> None:
    scores = np.array([[0.9, 0.2, 0.9],
                       [0.5, 0.9, 0.9],
                       [0.1, 0.9, 0.9]])
    exits = exit_layers(scores, 0.5)
    # éch. 0 : couche 2 (0,5 <= 0,5) ; éch. 1 : couche 1 ; éch. 2 : jamais -> dernière
    assert exits.tolist() == [1, 0, 2]


def test_simulate_accuracy_lue_a_la_couche_de_sortie() -> None:
    scores = np.array([[0.1, 0.9],
                       [0.9, 0.1]])
    correct = np.array([[True, False],
                        [False, True]])
    out = simulate_early_exit(scores, correct, 0.5)
    # éch. 0 sort couche 1 (exact), éch. 1 sort couche 2 (exact)
    assert out["accuracy"] == 1.0
    assert out["mean_depth"] == pytest.approx(1.5)


def test_calibrate_preserve_l_exactitude() -> None:
    rng = np.random.default_rng(0)
    n, n_layers = 400, 6
    # scores décroissants avec la profondeur ; prédictions exactes dès la couche 3
    scores = np.linspace(1.0, 0.1, n_layers)[:, None] + 0.05 * rng.normal(size=(n_layers, n))
    correct = np.zeros((n_layers, n), dtype=bool)
    correct[2:] = True
    out = calibrate_epsilon(scores, correct, tolerance=0.0)
    assert out["accuracy"] >= out["accuracy_full"]
    assert out["mean_depth"] < n_layers  # sortie anticipée effective


def test_calibrate_repli_si_contrainte_infaisable() -> None:
    # seule la dernière couche est exacte : aucun arrêt anticipé admissible
    scores = np.tile(np.linspace(1.0, 0.1, 4)[:, None], (1, 50))
    correct = np.zeros((4, 50), dtype=bool)
    correct[-1] = True
    out = calibrate_epsilon(scores, correct, tolerance=0.0)
    assert out["mean_depth"] == 4.0
    assert out["accuracy"] == 1.0


def test_validations() -> None:
    with pytest.raises(ValueError):
        exit_layers(np.zeros(3), 0.5)
    with pytest.raises(ValueError):
        simulate_early_exit(np.zeros((2, 3)), np.zeros((3, 2), dtype=bool), 0.5)
    with pytest.raises(ValueError):
        calibrate_epsilon(np.zeros((2, 3)), np.zeros((2, 3), dtype=bool), tolerance=1.0)
