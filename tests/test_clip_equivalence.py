"""L'inférence zero-shot de CLIP est exactement une mesure de Gibbs.

Ce test vérifie numériquement l'observation centrale de l'axe :
sur la sphère unité, ||z - mu||^2 = 2 - 2<z, mu>, donc

    softmax_k( <z, mu_k> / tau )  ==  softmax_k( -||z - mu_k||^2 / T )   avec T = 2*tau

Il ne requiert aucun modèle : seule l'identité algébrique est en jeu.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tlsc.core.gibbs import gibbs


def _clip_zero_shot(z: torch.Tensor, mu: torch.Tensor, tau: float) -> torch.Tensor:
    """Postérieur zero-shot tel que CLIP le calcule : softmax des cosinus / tau."""
    return torch.softmax((z @ mu.t()) / tau, dim=-1)


def test_equivalence_exacte_pour_toute_temperature() -> None:
    torch.manual_seed(0)
    z = F.normalize(torch.randn(128, 64, dtype=torch.float64), dim=-1)
    mu = F.normalize(torch.randn(7, 64, dtype=torch.float64), dim=-1)
    for tau in (0.005, 0.01, 0.02, 0.07, 0.5):
        p_clip = _clip_zero_shot(z, mu, tau)
        p_gibbs = gibbs(z, mu, T=2.0 * tau)[0].exp()
        assert torch.allclose(p_clip, p_gibbs, atol=1e-12), f"écart à tau={tau}"


def test_equivalence_sur_les_predictions() -> None:
    """Les deux voies donnent la même classe prédite, y compris en simple précision."""
    torch.manual_seed(1)
    z = F.normalize(torch.randn(512, 128), dim=-1)
    mu = F.normalize(torch.randn(4, 128), dim=-1)
    tau = 0.01
    a = _clip_zero_shot(z, mu, tau).argmax(-1)
    b = gibbs(z, mu, T=2.0 * tau)[0].argmax(-1)
    assert torch.equal(a, b)


def test_le_facteur_deux_est_necessaire() -> None:
    """Poser T = tau au lieu de T = 2*tau change le résultat : le facteur n'est
    pas une convention d'écriture."""
    torch.manual_seed(2)
    z = F.normalize(torch.randn(64, 32, dtype=torch.float64), dim=-1)
    mu = F.normalize(torch.randn(3, 32, dtype=torch.float64), dim=-1)
    tau = 0.05
    p_clip = _clip_zero_shot(z, mu, tau)
    p_faux = gibbs(z, mu, T=tau)[0].exp()
    assert not torch.allclose(p_clip, p_faux, atol=1e-6)
