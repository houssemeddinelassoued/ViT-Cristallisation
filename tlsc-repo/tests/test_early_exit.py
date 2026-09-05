"""Tests de la règle d'arrêt anticipé par échantillon."""
from __future__ import annotations

import numpy as np
import pytest

from tlsc.eval.early_exit import (
    balanced_accuracy_at_exit,
    calibrate_epsilon,
    exit_depth_auroc,
    exit_layers,
    simulate_early_exit,
)


def _cohorte_desequilibree() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/20, couche 1 = predicteur majoritaire, couche 3 = vrai signal.

    Les deux couches ont la MEME exactitude brute (0,80) ; seule l'exactitude
    equilibree les separe (0,50 contre 0,80). C'est la configuration rencontree
    sur BreastMNIST, ou la proportion de la classe majoritaire vaut 0,731.
    """
    y = np.array([0] * 20 + [1] * 80)
    correct = np.zeros((3, len(y)), dtype=bool)
    correct[0] = y == 1                       # couche 1 : classe majoritaire seule
    correct[1] = correct[0]
    correct[2, :16] = True                    # couche 3 : 16/20 et 64/80
    correct[2, 20:84] = True
    scores = np.tile(np.array([0.1, 0.5, 0.9])[:, None], (1, len(y)))
    return scores, correct, y


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


def test_profondeur_de_sortie_separe_les_domaines() -> None:
    # source : passe sous le seuil dès la couche 1 ; cible : seulement en couche 3
    source = np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
    target = np.array([[0.9, 0.9], [0.9, 0.9], [0.1, 0.1]])
    out = exit_depth_auroc(source, target, 0.5)
    assert out["auroc"] == 1.0
    assert out["mean_depth_source"] == pytest.approx(1.0)
    assert out["mean_depth_target"] == pytest.approx(3.0)
    assert out["delta_depth"] == pytest.approx(2.0)
    assert out["n_source"] == 2 and out["n_target"] == 2


def test_regle_degeneree_donne_un_auroc_de_hasard() -> None:
    # toutes les images sortent à la même couche : le compteur n'informe pas
    scores = np.full((4, 20), 0.1)
    out = exit_depth_auroc(scores, scores.copy(), 0.5)
    assert out["auroc"] == pytest.approx(0.5)
    assert out["delta_depth"] == pytest.approx(0.0)


def test_exit_depth_auroc_validations() -> None:
    with pytest.raises(ValueError):
        exit_depth_auroc(np.zeros(4), np.zeros((4, 2)), 0.5)
    with pytest.raises(ValueError):
        exit_depth_auroc(np.zeros((4, 2)), np.zeros((3, 2)), 0.5)


def test_exactitude_equilibree_par_classe() -> None:
    hit = np.array([True, False, True, True, True])
    y = np.array([0, 0, 1, 1, 1])
    # classe 0 : 1/2 = 0,5 ; classe 1 : 3/3 = 1,0 -> moyenne 0,75
    assert balanced_accuracy_at_exit(hit, y) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        balanced_accuracy_at_exit(hit, y[:-1])


def test_critere_brut_valide_l_effondrement_majoritaire() -> None:
    """Sans etiquettes, la calibration sort en couche 1 sur un predicteur nul."""
    scores, correct, y = _cohorte_desequilibree()
    out = calibrate_epsilon(scores, correct, tolerance=0.01)
    assert out["criterion"] == "accuracy"
    assert out["mean_depth"] == pytest.approx(1.0)
    assert out["accuracy"] == pytest.approx(0.80)
    # l'exactitude equilibree de ce choix est celle du hasard
    assert balanced_accuracy_at_exit(
        correct[exit_layers(scores, out["epsilon"]), np.arange(len(y))], y
    ) == pytest.approx(0.5)


def test_critere_equilibre_refuse_l_effondrement() -> None:
    scores, correct, y = _cohorte_desequilibree()
    out = calibrate_epsilon(scores, correct, tolerance=0.01, y=y)
    assert out["criterion"] == "balanced_accuracy"
    assert out["mean_depth"] == pytest.approx(3.0)
    assert out["balanced_accuracy"] == pytest.approx(0.80)
    assert out["balanced_accuracy_full"] == pytest.approx(0.80)


def test_simulate_ajoute_l_equilibree_si_etiquettes() -> None:
    scores, correct, y = _cohorte_desequilibree()
    sans = simulate_early_exit(scores, correct, 0.2)
    avec = simulate_early_exit(scores, correct, 0.2, y)
    assert "balanced_accuracy" not in sans
    assert avec["balanced_accuracy"] == pytest.approx(0.5)
    assert avec["accuracy"] == sans["accuracy"]


def test_bilateral_recupere_un_detecteur_inverse() -> None:
    """Cible sortant plus TOT : l'unilateral s'effondre, le bilateral recupere.

    Configuration rencontree a l'exp05 sous ancres R3 (AUROC 0,059) : les images
    corrompues cristallisent plus tot, ce que le detecteur unilateral lit comme
    l'inverse d'un decalage.
    """
    # source : sortie en couche 3 ; cible : sortie en couche 1
    source = np.array([[0.9] * 10, [0.9] * 10, [0.1] * 10])
    target = np.array([[0.1] * 10, [0.1] * 10, [0.1] * 10])
    out = exit_depth_auroc(source, target, 0.5)
    assert out["mean_depth_source"] == pytest.approx(3.0)
    assert out["mean_depth_target"] == pytest.approx(1.0)
    assert out["auroc"] == pytest.approx(0.0)            # unilateral : totalement inverse
    assert out["auroc_two_sided"] == pytest.approx(1.0)  # bilateral : recupere


def test_bilateral_neutre_quand_les_domaines_coincident() -> None:
    scores = np.array([[0.9] * 8, [0.1] * 8, [0.1] * 8])
    out = exit_depth_auroc(scores, scores.copy(), 0.5)
    assert out["auroc"] == pytest.approx(0.5)
    assert out["auroc_two_sided"] == pytest.approx(0.5)
