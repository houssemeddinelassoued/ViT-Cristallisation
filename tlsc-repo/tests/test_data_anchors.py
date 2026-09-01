"""Tests des ancres de données (centroïdes R2/R3 et distance inter-classes)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tlsc.models.data_anchors import class_centroids, inter_class_distance


def _toy(n_per_class: int = 30, d: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    a = F.normalize(torch.randn(d, generator=g), dim=-1)
    b = F.normalize(torch.randn(d, generator=g), dim=-1)
    z0 = F.normalize(a + 0.1 * torch.randn(n_per_class, d, generator=g), dim=-1)
    z1 = F.normalize(b + 0.1 * torch.randn(n_per_class, d, generator=g), dim=-1)
    z = torch.cat([z0, z1])
    y = torch.cat([torch.zeros(n_per_class, dtype=torch.long),
                   torch.ones(n_per_class, dtype=torch.long)])
    return z, y


def test_centroides_normalises_et_proches_des_directions() -> None:
    z, y = _toy()
    mu = class_centroids(z, y)
    assert mu.shape == (2, 16)
    norms = mu.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    # chaque centroïde est plus proche de sa classe que de l'autre
    for c in (0, 1):
        own = (z[y == c] @ mu[c]).mean()
        other = (z[y == c] @ mu[1 - c]).mean()
        assert own > other


def test_few_shot_reproductible_et_different_du_complet() -> None:
    z, y = _toy(n_per_class=50)
    r2a = class_centroids(z, y, k_shot=8, seed=3)
    r2b = class_centroids(z, y, k_shot=8, seed=3)
    r3 = class_centroids(z, y)
    assert torch.equal(r2a, r2b)
    assert not torch.allclose(r2a, r3, atol=1e-6)


def test_k_shot_trop_grand_refuse() -> None:
    z, y = _toy(n_per_class=5)
    with pytest.raises(ValueError):
        class_centroids(z, y, k_shot=6)


def test_validations_entrees() -> None:
    z, y = _toy()
    with pytest.raises(ValueError):
        class_centroids(z, y[:-1])
    with pytest.raises(ValueError):
        class_centroids(z, torch.zeros(len(y), dtype=torch.long))
    with pytest.raises(ValueError):
        class_centroids(z, y, k_shot=0)


def test_distance_inter_classes() -> None:
    mu = torch.eye(2)  # orthogonaux sur la sphère : ||mu0 - mu1||^2 = 2
    assert inter_class_distance(mu) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        inter_class_distance(mu[:1])
