"""Tests de la sonde par couche (logit lens) sur un ViT open_clip synthétique.

Aucun poids téléchargé : un VisionTransformer miniature initialisé
aléatoirement suffit à vérifier les propriétés structurelles.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tlsc.models.layer_probe import probe_layers


@pytest.fixture(scope="module")
def tiny_vit():
    from open_clip.transformer import VisionTransformer

    torch.manual_seed(0)
    vt = VisionTransformer(image_size=32, patch_size=16, width=64, layers=3,
                           heads=2, mlp_ratio=2.0, output_dim=16)
    vt.eval()
    return vt


def test_derniere_couche_egale_encode_image(tiny_vit) -> None:
    torch.manual_seed(1)
    px = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        layers = probe_layers(tiny_vit, px)
        reference = F.normalize(tiny_vit(px).float(), dim=-1)
    assert layers.shape == (3, 4, 16)
    assert torch.allclose(layers[-1], reference, atol=1e-6)


def test_embeddings_normalises_a_chaque_couche(tiny_vit) -> None:
    torch.manual_seed(2)
    px = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        layers = probe_layers(tiny_vit, px)
    norms = layers.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_couches_intermediaires_distinctes(tiny_vit) -> None:
    """La sonde doit refléter la progression : deux couches ne coïncident pas."""
    torch.manual_seed(3)
    px = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        layers = probe_layers(tiny_vit, px)
    assert not torch.allclose(layers[0], layers[-1], atol=1e-3)


def test_pooling_attentionnel_refuse() -> None:
    from open_clip.transformer import VisionTransformer

    vt = VisionTransformer(image_size=32, patch_size=16, width=64, layers=1,
                           heads=2, mlp_ratio=2.0, output_dim=16,
                           attentional_pool=True)
    vt.eval()
    with pytest.raises(ValueError):
        probe_layers(vt, torch.randn(1, 3, 32, 32))
