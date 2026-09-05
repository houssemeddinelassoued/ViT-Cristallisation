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


def anchor_cloud_distance(z: Tensor, mu: Tensor) -> float:
    """Distance quadratique moyenne du nuage à son ancre la plus proche.

    Mesure le « gap » entre les ancres et les données. Pour des ancres
    textuelles CLIP, cette quantité capture le gap de modalité : les
    plongements de texte et d'image occupent deux cônes disjoints de
    l'hypersphère, si bien que les ancres peuvent être très écartées entre
    elles tout en étant loin de toute image.

    Parameters
    ----------
    z : Tensor, shape (N, d) — embeddings L2-normalisés.
    mu : Tensor, shape (K, d) — ancres L2-normalisées.

    Returns
    -------
    float — moyenne sur les N embeddings de ``min_k ||z - mu_k||^2``.
    """
    if z.ndim != 2 or mu.ndim != 2 or z.shape[1] != mu.shape[1]:
        raise ValueError("z (N, d) et mu (K, d) doivent partager la dimension d")
    d2 = torch.cdist(z, mu).pow(2)
    return float(d2.min(dim=1).values.mean())


def anchor_quality(z: Tensor, mu: Tensor, y: Tensor | None = None) -> dict[str, float]:
    """Critère de qualité d'ancrage : séparation rapportée à la dispersion.

    Motivation (exp03) : ``D_inter`` seule est trompeuse — elle est maximale
    pour les ancres textuelles, qui sont pourtant les moins utiles, parce que le
    gap de modalité écarte les ancres entre elles sans les rapprocher des
    données. Un critère exploitable doit donc rapporter la séparation
    inter-classes à une échelle de dispersion mesurée sur le nuage.

    Deux rapports sans dimension sont retournés, tous deux « plus grand =
    mieux » :

    ``Q_gap = D_inter / d_cloud``
        Séparation rapportée au gap ancres-nuage. Ne demande aucune étiquette,
        donc calculable sur un domaine cible non annoté.
    ``Q_fisher = D_inter / d_within``
        Séparation rapportée à la dispersion intra-classe autour de l'ancre de
        sa propre classe — transposition du rapport de Fisher à des ancres
        imposées. Demande les étiquettes, donc réservé à la source.

    Parameters
    ----------
    z : Tensor, shape (N, d) — embeddings L2-normalisés.
    mu : Tensor, shape (K, d) — ancres L2-normalisées.
    y : Tensor | None, shape (N,)
        Étiquettes 0..K-1. Si ``None``, ``d_within`` et ``Q_fisher`` sont
        absents du résultat.

    Returns
    -------
    dict : ``D_inter``, ``d_cloud``, ``Q_gap``, et si ``y`` est fourni
    ``d_within`` et ``Q_fisher``.
    """
    d_inter = inter_class_distance(mu)
    d_cloud = anchor_cloud_distance(z, mu)
    out = {
        "D_inter": d_inter,
        "d_cloud": d_cloud,
        "Q_gap": float(d_inter / d_cloud) if d_cloud > 0.0 else float("inf"),
    }
    if y is not None:
        if y.ndim != 1 or y.shape[0] != z.shape[0]:
            raise ValueError("y doit être (N,) et de même longueur que z")
        if int(y.max()) >= mu.shape[0] or int(y.min()) < 0:
            raise ValueError("les étiquettes doivent indexer les ancres (0..K-1)")
        d2 = torch.cdist(z, mu).pow(2)
        d_within = float(d2.gather(1, y.view(-1, 1).long()).mean())
        out["d_within"] = d_within
        out["Q_fisher"] = (float(d_inter / d_within) if d_within > 0.0
                           else float("inf"))
    return out
