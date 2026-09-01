"""Métriques d'évaluation.

Aucune valeur de référence n'est codée en dur : ce module calcule, il n'affirme
rien.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve

__all__ = [
    "bivariate_shift_score", "bootstrap_auroc_delta", "brier_score", "delong_test",
    "diagnostic_metrics", "expected_calibration_error", "paired_bootstrap_ci",
    "sensitivity_at_specificity", "shift_detection_auroc", "two_sided_shift_score",
]


def sensitivity_at_specificity(y_true: np.ndarray, score: np.ndarray,
                               target: float = 0.95) -> float:
    """Sensibilité maximale parmi les seuils atteignant la spécificité cible."""
    if not 0.0 <= target <= 1.0:
        raise ValueError("target doit appartenir à [0, 1]")
    try:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, score)
    except ValueError:
        return float("nan")
    feasible = true_positive_rate[false_positive_rate <= 1.0 - target]
    return float(feasible.max()) if len(feasible) else 0.0


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Score de Brier multiclasse, somme quadratique moyenne par exemple."""
    if p.ndim != 2:
        raise ValueError("p doit avoir la forme (N, K)")
    one_hot = np.eye(p.shape[1], dtype=p.dtype)[y_true]
    return float(np.square(p - one_hot).sum(axis=1).mean())


def diagnostic_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Exactitude, exactitude équilibrée, F1 macro et AUROC.

    Parameters
    ----------
    y_true : (N,) entiers
    p : (N, K) postérieurs
    """
    pred = p.argmax(1)
    out = {
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "brier": brier_score(y_true, p),
    }
    try:
        out["auroc"] = float(
            roc_auc_score(y_true, p[:, 1]) if p.shape[1] == 2
            else roc_auc_score(y_true, p, multi_class="ovr")
        )
    except ValueError:  # une seule classe présente
        out["auroc"] = float("nan")
    out["sensitivity_at_95_specificity"] = (
        sensitivity_at_specificity(y_true, p[:, 1]) if p.shape[1] == 2 else float("nan")
    )
    return out


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """ECE à bacs de largeur égale sur la confiance maximale."""
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def shift_detection_auroc(score_source: np.ndarray, score_target: np.ndarray) -> float:
    """AUROC d'un score comme détecteur de décalage source contre cible.

    Convention : un score PLUS ÉLEVÉ doit indiquer la cible. Une valeur proche de
    0,5 signifie que le score ne distingue pas les deux domaines.
    """
    y = np.concatenate([np.zeros(len(score_source)), np.ones(len(score_target))])
    s = np.concatenate([score_source, score_target])
    return float(roc_auc_score(y, s))


def two_sided_shift_score(score_source: np.ndarray,
                          score_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score de décalage bilatéral : écart absolu à la médiane SOURCE.

    Motivation empirique : certains décalages photométriques rapprochent les
    images de toutes les ancres et font *baisser* F, ce qui inverse l'AUROC du
    détecteur unilatéral. La statistique ``|s - médiane(source)|`` détecte les
    départs dans les deux directions. La médiane est calculée sur la source
    seule : aucune information cible n'entre dans la calibration.

    Returns
    -------
    (scores_source, scores_target) transformés, prêts pour
    :func:`shift_detection_auroc`.
    """
    src = np.asarray(score_source, dtype=np.float64)
    tgt = np.asarray(score_target, dtype=np.float64)
    if src.ndim != 1 or tgt.ndim != 1 or len(src) == 0 or len(tgt) == 0:
        raise ValueError("score_source et score_target doivent être des vecteurs non vides")
    center = float(np.median(src))
    return np.abs(src - center), np.abs(tgt - center)


def bivariate_shift_score(f_source: np.ndarray, h_source: np.ndarray,
                          f_target: np.ndarray, h_target: np.ndarray,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Score de décalage bivarié : distance de Mahalanobis au nuage SOURCE (F, H).

    Généralise le détecteur bilatéral : le décalage est mesuré comme un départ
    du couple (F, H) hors de la distribution source, dans n'importe quelle
    direction du plan. La moyenne et la covariance sont estimées sur la source
    seule : aucune information cible n'entre dans la calibration.

    Returns
    -------
    (scores_source, scores_target) prêts pour :func:`shift_detection_auroc`.
    """
    fs, hs = np.asarray(f_source, np.float64), np.asarray(h_source, np.float64)
    ft, ht = np.asarray(f_target, np.float64), np.asarray(h_target, np.float64)
    if any(a.ndim != 1 or len(a) == 0 for a in (fs, hs, ft, ht)):
        raise ValueError("les quatre scores doivent être des vecteurs non vides")
    if len(fs) != len(hs) or len(ft) != len(ht):
        raise ValueError("F et H doivent être appariés au sein de chaque domaine")
    if len(fs) < 3:
        raise ValueError("au moins 3 exemples source sont requis pour la covariance")
    src = np.stack([fs, hs], axis=1)
    tgt = np.stack([ft, ht], axis=1)
    mu = src.mean(axis=0)
    cov = np.cov(src, rowvar=False)
    # régularisation ridge : F et H sont corrélés via F = <E> - T·H
    cov += 1e-9 * max(float(np.trace(cov)), 1e-12) * np.eye(2)
    inv = np.linalg.inv(cov)

    def dist(m: np.ndarray) -> np.ndarray:
        c = m - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", c, inv, c))

    return dist(src), dist(tgt)


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n: int = 10_000,
                        alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Intervalle de confiance apparié sur la différence moyenne a - b.

    Returns
    -------
    (différence observée, borne basse, borne haute)
    """
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("a et b doivent être des vecteurs non vides de même forme")
    if n <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("n doit être positif et alpha appartenir à ]0, 1[")
    rng = np.random.default_rng(seed)
    d = a - b
    boot = np.empty(n, dtype=np.float64)
    chunk_size = min(1024, n)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        idx = rng.integers(0, len(d), size=(stop - start, len(d)))
        boot[start:stop] = d[idx].mean(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def _auroc_by_ranks(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC de Mann-Whitney par rangs moyens, robuste aux ex aequo."""
    ranks = rankdata(np.concatenate([pos, neg]), method="average")
    m, n = len(pos), len(neg)
    return float((ranks[:m].sum() - m * (m + 1) / 2) / (m * n))


def _split_by_label(y_true: np.ndarray, score_a: np.ndarray,
                    score_b: np.ndarray) -> tuple[np.ndarray, ...]:
    y = np.asarray(y_true).astype(bool)
    a, b = np.asarray(score_a, dtype=np.float64), np.asarray(score_b, dtype=np.float64)
    if not (y.shape == a.shape == b.shape) or y.ndim != 1:
        raise ValueError("y_true, score_a et score_b doivent être des vecteurs de même forme")
    if not (y.any() and (~y).any()):
        raise ValueError("les deux classes doivent être présentes")
    return a[y], a[~y], b[y], b[~y]


def delong_test(y_true: np.ndarray, score_a: np.ndarray,
                score_b: np.ndarray) -> dict[str, float]:
    """Test de DeLong pour deux AUROC appariées sur les mêmes échantillons.

    Returns
    -------
    dict : ``auroc_a``, ``auroc_b``, ``delta`` (a - b), ``z``, ``p_value``
        (bilatéral, approximation normale de DeLong et al., 1988).
    """
    pos_a, neg_a, pos_b, neg_b = _split_by_label(y_true, score_a, score_b)
    m, n = len(pos_a), len(neg_a)
    v10 = np.empty((2, m))
    v01 = np.empty((2, n))
    aucs = np.empty(2)
    for i, (pos, neg) in enumerate(((pos_a, neg_a), (pos_b, neg_b))):
        tz = rankdata(np.concatenate([pos, neg]), method="average")
        aucs[i] = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10[i] = (tz[:m] - rankdata(pos, method="average")) / n
        v01[i] = 1.0 - (tz[m:] - rankdata(neg, method="average")) / m
    cov = np.cov(v10) / m + np.cov(v01) / n
    delta = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 0.0:
        z = 0.0 if delta == 0.0 else float("inf") * float(np.sign(delta))
        p = 1.0 if delta == 0.0 else 0.0
    else:
        z = delta / float(np.sqrt(var))
        p = float(2.0 * norm.sf(abs(z)))
    return {"auroc_a": float(aucs[0]), "auroc_b": float(aucs[1]),
            "delta": delta, "z": float(z), "p_value": p}


def bootstrap_auroc_delta(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                          n: int = 10_000, alpha: float = 0.05,
                          seed: int = 0) -> tuple[float, float, float]:
    """IC bootstrap stratifié et apparié de AUROC(a) - AUROC(b).

    Le rééchantillonnage tire les mêmes indices pour les deux scores, classe par
    classe, préservant l'appariement échantillon par échantillon.
    """
    if n <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("n doit être positif et alpha appartenir à ]0, 1[")
    pos_a, neg_a, pos_b, neg_b = _split_by_label(y_true, score_a, score_b)
    rng = np.random.default_rng(seed)
    m, k = len(pos_a), len(neg_a)
    deltas = np.empty(n, dtype=np.float64)
    for i in range(n):
        ip = rng.integers(0, m, size=m)
        ineg = rng.integers(0, k, size=k)
        deltas[i] = (_auroc_by_ranks(pos_a[ip], neg_a[ineg])
                     - _auroc_by_ranks(pos_b[ip], neg_b[ineg]))
    observed = _auroc_by_ranks(pos_a, neg_a) - _auroc_by_ranks(pos_b, neg_b)
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return float(observed), float(lo), float(hi)
