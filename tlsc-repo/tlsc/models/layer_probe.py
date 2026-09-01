"""Sonde par couche : embeddings intermédiaires d'un ViT open_clip (« logit lens »).

Motivation — croquis fondateur du projet : la profondeur N est une variable
d'optimisation (`argmin_N H`). Pour mesurer la trajectoire des observables de
Gibbs en fonction de la profondeur, il faut extraire à chaque bloc du
transformeur visuel un embedding comparable aux ancres textuelles. La seule
voie fidèle est la boucle explicite sur les blocs (les forward hooks ne
peuvent pas interrompre le calcul — piège documenté dans CLAUDE.md), suivie à
chaque profondeur du même post-traitement que la sortie finale : ``ln_post``
puis projection ``proj`` sur le jeton CLS.

Avertissement d'interprétation : seule la dernière couche a été alignée à
l'espace texte par l'entraînement contrastif. Les embeddings intermédiaires
passés par la projection finale sont une lecture « logit lens » — leur
géométrie vis-à-vis des ancres est une question empirique, pas une garantie.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def probe_layers(visual: torch.nn.Module, px: Tensor) -> Tensor:
    """Embeddings L2-normalisés du jeton CLS après chaque bloc du ViT visuel.

    Parameters
    ----------
    visual : open_clip.transformer.VisionTransformer
        Encodeur visuel gelé. Doit utiliser le pooling par jeton CLS
        (``pool_type='tok'``) sans pooling attentionnel.
    px : Tensor, shape (B, 3, H, W)
        Lot d'images prétraitées.

    Returns
    -------
    Tensor, shape (N, B, d)
        Pour chaque profondeur n = 1..N, l'embedding projeté et L2-normalisé.
        La couche N coïncide avec ``visual(px)`` normalisé, à la précision
        machine (testé).
    """
    if getattr(visual, "attn_pool", None) is not None:
        raise ValueError("sonde non définie pour les modèles à pooling attentionnel")
    if getattr(visual, "pool_type", None) != "tok":
        raise ValueError(f"pool_type non pris en charge: {getattr(visual, 'pool_type', None)!r}")
    if visual.proj is None:
        raise ValueError("l'encodeur visuel doit posséder une projection finale")

    x = visual._embeds(px)
    transformer = visual.transformer
    batch_first = bool(getattr(transformer, "batch_first", True))
    if not batch_first:
        x = x.transpose(0, 1).contiguous()  # NLD -> LND

    outs: list[Tensor] = []
    for block in transformer.resblocks:
        x = block(x)
        seq = x.transpose(0, 1) if not batch_first else x  # (B, L, width)
        if visual.final_ln_after_pool:
            pooled = visual.ln_post(seq[:, 0])
        else:
            pooled = visual.ln_post(seq)[:, 0]
        outs.append(F.normalize((pooled @ visual.proj).float(), dim=-1))
    return torch.stack(outs)
