"""Tests des ancres de données (centroïdes R2/R3 et distance inter-classes)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tlsc.models.data_anchors import (
    anchor_cloud_distance,
    anchor_quality,
    class_centroids,
    inter_class_distance,
)


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


def _nuage_et_ancres_textuelles(d: int = 16, n: int = 40, seed: int = 1):
    """Nuage à deux classes voisines, et des ancres « textuelles » hors du cône.

    Reproduit en miniature la géométrie observée à l'exp03 : les ancres
    textuelles sont maximalement écartées entre elles (antipodales) mais
    orthogonales au nuage d'images — c'est le gap de modalité.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.zeros(d)
    base[0] = 1.0
    tilt = torch.zeros(d)
    tilt[1] = 1.0
    z0 = F.normalize(base + 0.05 * torch.randn(n, d, generator=g), dim=-1)
    z1 = F.normalize(base + 0.30 * tilt + 0.05 * torch.randn(n, d, generator=g), dim=-1)
    z = torch.cat([z0, z1])
    y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    texte = torch.zeros(2, d)
    texte[0, 8], texte[1, 8] = 1.0, -1.0  # antipodales : D_inter maximal (= 4)
    return z, y, texte


def test_distance_ancres_nuage_croit_avec_l_eloignement() -> None:
    z, y, texte = _nuage_et_ancres_textuelles()
    centroides = class_centroids(z, y)
    assert anchor_cloud_distance(z, centroides) < anchor_cloud_distance(z, texte)
    assert anchor_cloud_distance(z, centroides) >= 0.0


def test_qualite_ancrage_penalise_le_gap_de_modalite() -> None:
    """Le critère doit renverser le classement trompeur de D_inter seule."""
    z, y, texte = _nuage_et_ancres_textuelles()
    centroides = class_centroids(z, y)
    q_texte = anchor_quality(z, texte, y)
    q_centro = anchor_quality(z, centroides, y)
    # D_inter seule désigne (à tort) les ancres textuelles comme les meilleures
    assert q_texte["D_inter"] > q_centro["D_inter"]
    # les deux rapports rétablissent le classement utile
    assert q_centro["Q_gap"] > q_texte["Q_gap"]
    assert q_centro["Q_fisher"] > q_texte["Q_fisher"]


def test_qualite_sans_etiquettes_omet_le_rapport_de_fisher() -> None:
    z, y, _ = _nuage_et_ancres_textuelles()
    mu = class_centroids(z, y)
    q = anchor_quality(z, mu)
    assert set(q) == {"D_inter", "d_cloud", "Q_gap"}
    assert q["Q_gap"] > 0.0


def test_qualite_validations() -> None:
    z, y, _ = _nuage_et_ancres_textuelles()
    mu = class_centroids(z, y)
    with pytest.raises(ValueError):
        anchor_cloud_distance(z, mu[:, :-1])
    with pytest.raises(ValueError):
        anchor_quality(z, mu, y[:-1])
    with pytest.raises(ValueError):
        anchor_quality(z, mu, torch.full_like(y, 5))
