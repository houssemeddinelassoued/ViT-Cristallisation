"""Encodeur vision-langage gelé et ancres textuelles.

Les ancres de classe ne sont pas apprises : ce sont les plongements textuels des
invites décrivant chaque classe. C'est ce qui rend la Phase 1 optionnelle en
régime zero-shot.

Les invites sont figées sur la validation SOURCE et déclarées explicitement ;
les choisir au vu des résultats cibles constituerait une fuite.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from tlsc.core.gibbs import temperature_from_logit_scale


@dataclass
class PromptSet:
    """Jeu d'invites pour une tâche : une liste de formulations par classe."""

    name: str
    classes: list[str]
    templates: list[str] = field(default_factory=lambda: ["a photo of {}."])

    def phrases(self) -> list[list[str]]:
        """Retourne, pour chaque classe, la liste de ses formulations."""
        return [[t.format(c) for t in self.templates] for c in self.classes]

    @property
    def n_classes(self) -> int:
        return len(self.classes)


# Invites de référence, fixées a priori. Toute modification doit être datée et
# justifiée sur la validation source, jamais sur la cible.
PROMPTS: dict[str, PromptSet] = {
    "breastmnist": PromptSet(
        name="breastmnist",
        classes=["a malignant breast tumor", "a normal or benign breast tissue"],
        templates=[
            "a breast ultrasound image of {}.",
            "an ultrasound scan showing {}.",
            "grayscale sonography of {}.",
        ],
    ),
    "pneumoniamnist": PromptSet(
        name="pneumoniamnist",
        classes=["a normal chest", "a chest with pneumonia"],
        templates=[
            "a chest x-ray image of {}.",
            "a radiograph showing {}.",
        ],
    ),
    "pathmnist": PromptSet(
        name="pathmnist",
        classes=[
            "adipose tissue", "background", "debris", "lymphocytes", "mucus",
            "smooth muscle", "normal colon mucosa", "cancer-associated stroma",
            "colorectal adenocarcinoma epithelium",
        ],
        templates=[
            "a histopathology image of {}.",
            "a hematoxylin and eosin stained slide of {}.",
        ],
    ),
}


class FrozenCLIP:
    """Encapsule un encodeur ``open_clip`` gelé et ses ancres textuelles.

    Parameters
    ----------
    model_name : str
        Architecture ``open_clip`` (par exemple ``"ViT-B-16-quickgelu"``).
    pretrained : str
        Jeu de poids (``"openai"``, ``"laion2b_s34b_b88k"``, …).
    device : str
        Périphérique de calcul.
    """

    def __init__(self, model_name: str = "ViT-B-16-quickgelu", pretrained: str = "openai",
                 device: str = "cuda") -> None:
        import open_clip  # import tardif : la dépendance n'est utile qu'ici

        self.device = device
        self.model_name, self.pretrained = model_name, pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @property
    def temperature(self) -> float:
        """Température de Gibbs ``T = 2*tau`` déduite du modèle."""
        return temperature_from_logit_scale(self.model.logit_scale)

    @torch.no_grad()
    def anchors(self, prompts: PromptSet, ensemble: bool = True) -> Tensor:
        """Construit les ancres de classe à partir des invites.

        Parameters
        ----------
        prompts : PromptSet
        ensemble : bool, default True
            Si vrai, moyenne les plongements des formulations d'une même classe
            puis renormalise — baseline forte et gratuite. Sinon, n'utilise que
            la première formulation.

        Returns
        -------
        Tensor, shape (K, d), L2-normalisé.
        """
        out = []
        for phrases in prompts.phrases():
            chosen = phrases if ensemble else phrases[:1]
            tok = self.tokenizer(chosen).to(self.device)
            emb = F.normalize(self.model.encode_text(tok).float(), dim=-1)
            out.append(F.normalize(emb.mean(0), dim=-1))
        return torch.stack(out)

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        """Encode un lot d'images prétraitées. Retourne (B, d) L2-normalisé."""
        return F.normalize(self.model.encode_image(images.to(self.device)).float(), dim=-1)

    @torch.no_grad()
    def encode_layers(self, images: Tensor) -> Tensor:
        """Embeddings CLS après chaque bloc du ViT visuel (« logit lens »).

        Retourne (N, B, d) L2-normalisé ; la couche N égale ``encode`` à la
        précision machine. Voir ``tlsc/models/layer_probe.py``.
        """
        from tlsc.models.layer_probe import probe_layers

        return probe_layers(self.model.visual, images.to(self.device))
