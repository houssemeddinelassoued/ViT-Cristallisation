"""Invariants du noyau formel. Ces tests doivent être verts avant toute expérience."""
from __future__ import annotations

import math

import torch

from tlsc.core.gibbs import (
    crystallinity,
    entropy,
    free_energy,
    observables,
    temperature_from_logit_scale,
)


def test_uniforme_donne_ln_K() -> None:
    """Un point équidistant de toutes les ancres atteint l'entropie maximale."""
    K, d = 4, 8
    mu = torch.eye(K, d)
    z = mu.mean(0, keepdim=True)
    z = z / z.norm()
    obs = observables(z, mu, T=1.0)
    assert torch.allclose(obs["H"], torch.tensor([math.log(K)]), atol=1e-5)
    assert torch.allclose(obs["chi"], torch.zeros(1), atol=1e-5)


def test_degenere_donne_zero() -> None:
    """Une température très basse rend l'assignation déterministe."""
    mu = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z = torch.tensor([[1.0, 0.0]])
    obs = observables(z, mu, T=1e-3)
    assert obs["H"].item() < 1e-6
    assert obs["chi"].item() > 1.0 - 1e-6


def test_identite_thermodynamique() -> None:
    """Lemme 1 : F = <E> - T*H, à la précision machine, pour toute température."""
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(64, 16, dtype=torch.float64), dim=-1)
    mu = torch.nn.functional.normalize(torch.randn(3, 16, dtype=torch.float64), dim=-1)
    for T in (0.02, 0.05, 0.2, 1.0, 5.0):
        o = observables(z, mu, T)
        assert torch.allclose(o["F"], o["E"] - T * o["H"], atol=1e-10)


def test_invariance_de_H_et_covariance_de_F() -> None:
    """Proposition 2 : une translation uniforme des énergies laisse H inchangée
    et décale F d'exactement c. C'est le fondement du découplage."""
    torch.manual_seed(0)
    d2 = torch.rand(32, 5, dtype=torch.float64) * 4
    T, c = 0.3, 1.7
    H1 = entropy(torch.log_softmax(-d2 / T, -1))
    H2 = entropy(torch.log_softmax(-(d2 + c) / T, -1))
    assert torch.allclose(H1, H2, atol=1e-12)
    assert torch.allclose(free_energy(d2 + c, T), free_energy(d2, T) + c, atol=1e-9)


def test_corollaire_loin_de_toutes_les_ancres() -> None:
    """Corollaire clinique : un point éloigné de TOUTES les ancres a une entropie
    basse (confiance apparente) mais une énergie libre élevée. C'est ce que
    l'entropie seule ne peut pas détecter."""
    mu = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    proche = torch.tensor([[0.98, 0.20]])          # près de l'ancre 0
    loin = torch.tensor([[6.0, 1.2]])              # loin des deux ancres
    T = 0.5
    o_p = observables(proche, mu, T, normalized=False)
    o_l = observables(loin, mu, T, normalized=False)
    assert o_l["H"].item() <= o_p["H"].item()      # pas plus incertain
    assert o_l["F"].item() > o_p["F"].item() + 1.0  # mais bien plus haut en énergie


def test_temperature_depuis_logit_scale() -> None:
    """T = 2*tau, avec tau = 1 / exp(logit_scale).

    Exact en float64 ; en float32 — le dtype réel du paramètre de CLIP — la
    tolérance est celle de la simple précision, pas une approximation du calcul.
    """
    ls64 = torch.tensor(math.log(100.0), dtype=torch.float64)
    assert abs(temperature_from_logit_scale(ls64) - 0.02) < 1e-15
    ls32 = torch.tensor(math.log(100.0), dtype=torch.float32)
    assert abs(temperature_from_logit_scale(ls32) - 0.02) < 1e-7


def test_crystallinite_bornee() -> None:
    torch.manual_seed(1)
    z = torch.nn.functional.normalize(torch.randn(256, 32), dim=-1)
    mu = torch.nn.functional.normalize(torch.randn(5, 32), dim=-1)
    chi = crystallinity(observables(z, mu, 0.4)["H"], 5)
    assert float(chi.min()) >= -1e-6 and float(chi.max()) <= 1.0 + 1e-6
