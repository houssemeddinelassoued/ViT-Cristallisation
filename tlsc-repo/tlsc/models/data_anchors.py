"""Ancres construites depuis les données : centroïdes de classe (régimes R2/R3).

Le protocole v3 (§2.3) définit trois régimes d'ancrage :

R1  invites textuelles pures (zero-shot) — implémenté dans ``clip_anchors.py`` ;
R2  raffinement few-shot : centroïdes calculés sur quelques exemples étiquetés
    du domaine SOURCE ;
R3  centroïdes calculés sur le jeu source complet — plafond de référence.

Ces centroïdes correspondent aux « mu_0, mu_1 » du croquis fondateur, avec la
distance inter-classes D_inter comme diagnostic de géométrie. Règle anti-fuite :
les centroïdes ne voient que le split d'entraînement SOURCE (sévérité 0) —
jamais d'image cible, jamais d'étiquette de test.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def class_centroids(z: Tensor, y: Tensor, k_shot: int | None = None,
                    seed: int = 0) -> Tensor:
    """Centroïdes de classe L2-normalisés à partir d'embeddings étiquetés.

    Parameters
    ----------
    z : Tensor, shape (N, d)
        Embeddings L2-normalisés du split source.
    y : Tensor, shape (N,)
        Étiquettes entières 0..K-1 ; chaque classe doit être présente.
    k_shot : int | None
        Si donné, ne retient que ``k_shot`` exemples par classe, tirés sans
        remise avec la graine ``seed`` (régime R2). ``None`` = tout le split (R3).
    seed : int
        Graine du tirage few-shot, sans effet si ``k_shot`` est None.

    Returns
    -------
    Tensor, shape (K, d), L2-normalisé — comparable aux ancres textuelles.
    """
    if z.ndim != 2 or y.ndim != 1 or z.shape[0] != y.shape[0]:
        raise ValueError("z doit être (N, d) et y (N,) de même longueur")
    if k_shot is not None and k_shot <= 0:
        raise ValueError("k_shot doit être strictement positif ou None")
    classes = torch.unique(y)
    if classes.numel() < 2:
        raise ValueError("au moins deux classes sont requises")
    gen = torch.Generator().manual_seed(seed)
    centroids = []
    for c in classes.tolist():
        idx = torch.nonzero(y == c, as_tuple=True)[0]
        if k_shot is not None:
            if len(idx) < k_shot:
                raise ValueError(f"classe {c}: {len(idx)} exemples < k_shot={k_shot}")
            idx = idx[torch.randperm(len(idx), generator=gen)[:k_shot]]
        centroids.append(F.normalize(z[idx].mean(0), dim=-1))
    return torch.stack(centroids)


def inter_class_distance(mu: Tensor) -> float:
    """Distance inter-classes du croquis fondateur : ||mu_0 - mu_1||^2.

    Pour K > 2, retourne la moyenne des distances carrées entre paires.
    """
    if mu.ndim != 2 or mu.shape[0] < 2:
        raise ValueError("mu doit être (K, d) avec K >= 2")
    d2 = torch.cdist(mu, mu).pow(2)
    k = mu.shape[0]
    return float(d2.sum() / (k * (k - 1)))
