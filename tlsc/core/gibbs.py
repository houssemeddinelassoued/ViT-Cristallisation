"""Observables thermodynamiques de l'espace latent.

Convention : embeddings et ancres L2-normalisés, donc l'énergie de configuration
``E_k(z) = ||z - mu_k||^2 = 2 - 2<z, mu_k>`` appartient à [0, 4].

Toutes les fonctions sont différentiables et opèrent par lots. La stabilité
numérique repose exclusivement sur ``log_softmax`` et ``logsumexp``.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "squared_distances", "gibbs", "entropy", "free_energy", "mean_energy",
    "observables", "crystallinity", "temperature_from_logit_scale",
]


def squared_distances(z: Tensor, mu: Tensor, normalized: bool = True) -> Tensor:
    """Distances euclidiennes au carré entre embeddings et ancres.

    Parameters
    ----------
    z : Tensor, shape (B, d)
        Embeddings image, L2-normalisés si ``normalized``.
    mu : Tensor, shape (K, d)
        Ancres de classe, L2-normalisées si ``normalized``.
    normalized : bool, default True
        Si vrai, utilise l'identité sphérique ``2 - 2<z, mu>`` : exacte,
        moins coûteuse et numériquement plus stable que ``cdist``.

    Returns
    -------
    Tensor, shape (B, K)
    """
    if normalized:
        return (2.0 - 2.0 * (z @ mu.t())).clamp_min(0.0)
    return torch.cdist(z, mu, p=2).pow(2)


def gibbs(z: Tensor, mu: Tensor, T: float, normalized: bool = True) -> tuple[Tensor, Tensor]:
    """Log-postérieur de Gibbs et énergies de configuration.

    Returns
    -------
    log_p : Tensor, shape (B, K)
    d2 : Tensor, shape (B, K)
    """
    d2 = squared_distances(z, mu, normalized)
    return torch.log_softmax(-d2 / T, dim=-1), d2


def entropy(log_p: Tensor) -> Tensor:
    """Entropie de Gibbs, en nats. Shape (B,)."""
    return -(log_p.exp() * log_p).sum(-1)


def free_energy(d2: Tensor, T: float) -> Tensor:
    """Énergie libre ``F = -T ln Z``. Shape (B,).

    Contrairement à l'entropie, elle dépend du niveau *absolu* des énergies :
    c'est ce qui lui permet de signaler un échantillon éloigné de toutes les
    ancres, même lorsque le modèle s'y montre confiant (Proposition 2).
    """
    return -T * torch.logsumexp(-d2 / T, dim=-1)


def mean_energy(log_p: Tensor, d2: Tensor) -> Tensor:
    """Énergie moyenne ``<E>``. Shape (B,)."""
    return (log_p.exp() * d2).sum(-1)


def crystallinity(H: Tensor, K: int) -> Tensor:
    """Paramètre d'ordre ``chi = 1 - H / ln K``, dans [0, 1]."""
    return 1.0 - H / math.log(K)


def temperature_from_logit_scale(logit_scale: Tensor | float) -> float:
    """Température de Gibbs correspondant à l'échelle de logits de CLIP.

    CLIP stocke ``logit_scale`` tel que les logits valent
    ``logit_scale.exp() * <z, t_k>``, donc ``tau = 1 / logit_scale.exp()``.
    L'équivalence établie dans le cadre théorique donne ``T = 2 * tau``.
    """
    ls = (float(logit_scale.detach().cpu()) if isinstance(logit_scale, Tensor)
          else float(logit_scale))
    return 2.0 / math.exp(ls)


def observables(z: Tensor, mu: Tensor, T: float, normalized: bool = True) -> dict[str, Tensor]:
    """Les trois observables, le postérieur et la cristallinité, en une passe.

    Returns
    -------
    dict avec les clés ``log_p``, ``p``, ``H``, ``F``, ``E``, ``chi``.
    """
    log_p, d2 = gibbs(z, mu, T, normalized)
    H = entropy(log_p)
    return {
        "log_p": log_p,
        "p": log_p.exp(),
        "H": H,
        "F": free_energy(d2, T),
        "E": mean_energy(log_p, d2),
        "chi": crystallinity(H, mu.shape[0]),
    }
