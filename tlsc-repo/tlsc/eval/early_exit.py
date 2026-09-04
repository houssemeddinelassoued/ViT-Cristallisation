"""Règle d'arrêt anticipé par échantillon sur les observables de Gibbs.

Origine : le croquis fondateur (`argmin_N H`, arrêt quand H < epsilon). Une
image sort à la PREMIÈRE couche où son observable passe sous un seuil epsilon ;
sinon elle parcourt tout le réseau. Le seuil est calibré sur un split source
dédié — jamais sur la cible (règle anti-fuite n° 4 du protocole).

Convention : un score BAS signifie « confiant / proche des ancres » (H comme F).
"""
from __future__ import annotations

import numpy as np


def exit_layers(scores: np.ndarray, epsilon: float) -> np.ndarray:
    """Couche de sortie de chaque échantillon pour un seuil donné.

    Parameters
    ----------
    scores : ndarray, shape (n_layers, n)
        Observable par couche et par échantillon (H ou F).
    epsilon : float
        Seuil d'arrêt : sortie à la première couche où ``score <= epsilon``.

    Returns
    -------
    ndarray, shape (n,), int — indices 0-based de la couche de sortie ;
    ``n_layers - 1`` si le seuil n'est jamais atteint.
    """
    if scores.ndim != 2 or scores.shape[0] < 1:
        raise ValueError("scores doit être (n_layers, n)")
    below = scores <= epsilon
    first = below.argmax(axis=0)
    never = ~below.any(axis=0)
    first[never] = scores.shape[0] - 1
    return first


def simulate_early_exit(scores: np.ndarray, correct: np.ndarray,
                        epsilon: float) -> dict[str, float]:
    """Profondeur moyenne et exactitude sous la règle d'arrêt.

    Parameters
    ----------
    scores : ndarray, shape (n_layers, n)
    correct : ndarray, shape (n_layers, n), booléen
        ``correct[l, i]`` : la prédiction de l'échantillon i à la couche l
        est-elle exacte ?
    epsilon : float

    Returns
    -------
    dict : ``mean_depth`` (1-based), ``accuracy``, ``epsilon``.
    """
    if scores.shape != correct.shape:
        raise ValueError("scores et correct doivent avoir la même forme")
    exits = exit_layers(scores, epsilon)
    idx = np.arange(scores.shape[1])
    return {
        "mean_depth": float(exits.mean() + 1.0),
        "accuracy": float(correct[exits, idx].mean()),
        "epsilon": float(epsilon),
    }


def calibrate_epsilon(scores: np.ndarray, correct: np.ndarray,
                      tolerance: float = 0.01,
                      quantiles: np.ndarray | None = None) -> dict[str, float]:
    """Choisit le seuil de profondeur moyenne minimale à exactitude préservée.

    Sur le split de CALIBRATION source : parmi une grille de seuils (quantiles
    des scores calibrés), retient celui qui minimise la profondeur moyenne sous
    la contrainte ``accuracy >= accuracy_pleine_profondeur - tolerance``.
    Repli : si aucun seuil ne satisfait la contrainte, retourne un seuil
    inatteignable (sortie systématique à la dernière couche).

    Returns
    -------
    dict : ``epsilon``, ``mean_depth``, ``accuracy``, ``accuracy_full``.
    """
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("tolerance doit appartenir à [0, 1[")
    if quantiles is None:
        quantiles = np.arange(0.05, 1.0, 0.05)
    acc_full = float(correct[-1].mean())
    grid = np.quantile(scores, quantiles)
    best: dict[str, float] | None = None
    for eps in np.unique(grid):
        out = simulate_early_exit(scores, correct, float(eps))
        if out["accuracy"] >= acc_full - tolerance and (
                best is None or out["mean_depth"] < best["mean_depth"]):
            best = out
    if best is None:
        best = simulate_early_exit(scores, correct, float(scores.min()) - 1.0)
    best["accuracy_full"] = acc_full
    return best


def exit_depth_auroc(scores_source: np.ndarray, scores_target: np.ndarray,
                     epsilon: float) -> dict[str, float]:
    """AUROC de la profondeur de sortie comme détecteur de décalage.

    Constat d'exp04 : sous un seuil calibré sur la source, les images corrompues
    « retardent » leur cristallisation — la profondeur de sortie croît avec la
    sévérité. Cette fonction quantifie ce constat en détecteur : les profondeurs
    N* du domaine source et du domaine cible sont-elles séparables ?

    Le compteur de couches est disponible gratuitement en production (aucun
    calcul supplémentaire : la règle d'arrêt le produit déjà), ce qui en ferait
    un moniteur de dérive sans surcoût.

    Convention : une profondeur PLUS ÉLEVÉE indique la cible, conformément à
    :func:`tlsc.eval.metrics.shift_detection_auroc`. Les profondeurs étant des
    entiers fortement ex aequo, l'AUROC est calculée avec la gestion standard
    des ex aequo (crédit 1/2). Une règle dégénérée qui fait sortir toutes les
    images à la même couche donne exactement 0,5.

    Parameters
    ----------
    scores_source, scores_target : ndarray, shape (n_layers, n)
        Observable par couche, domaine source et domaine cible.
    epsilon : float
        Seuil d'arrêt calibré sur la source.

    Returns
    -------
    dict : ``auroc``, ``mean_depth_source``, ``mean_depth_target``,
    ``delta_depth``, ``n_source``, ``n_target``.
    """
    from tlsc.eval.metrics import shift_detection_auroc

    if scores_source.ndim != 2 or scores_target.ndim != 2:
        raise ValueError("scores_source et scores_target doivent être (n_layers, n)")
    if scores_source.shape[0] != scores_target.shape[0]:
        raise ValueError("les deux domaines doivent avoir le même nombre de couches")
    src = exit_layers(scores_source, epsilon).astype(np.float64) + 1.0
    tgt = exit_layers(scores_target, epsilon).astype(np.float64) + 1.0
    return {
        "auroc": shift_detection_auroc(src, tgt),
        "mean_depth_source": float(src.mean()),
        "mean_depth_target": float(tgt.mean()),
        "delta_depth": float(tgt.mean() - src.mean()),
        "n_source": int(src.size),
        "n_target": int(tgt.size),
    }
