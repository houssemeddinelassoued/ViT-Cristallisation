"""Cohorte P0 : MedMNIST avec corruptions paramétrées par sévérité.

Deux sources de corruption sont possibles, et le choix est enregistré dans les
métriques du run :

``medmnistc``
    Corruptions réalistes par modalité du paquet ``medmnistc`` (ADSMI @ MICCAI
    2024). C'est la source à utiliser pour toute figure publiée.
``local``
    Repli minimal implémenté ici — bruit de speckle, flou, gain, gamma — pour
    permettre de faire tourner la chaîne sans dépendance supplémentaire. À
    déclarer explicitement : ce ne sont pas les corruptions de référence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SEVERITIES = (0, 1, 2, 3, 4, 5)


@dataclass
class Cohort:
    """Un jeu d'images PIL et leurs étiquettes."""

    images: list[Image.Image]
    labels: np.ndarray
    name: str
    severity: int
    corruption: str


def _to_rgb(arr: np.ndarray) -> Image.Image:
    """Convertit un tableau MedMNIST (H, W) ou (H, W, 3) en image RGB."""
    if arr.ndim == 2:
        a = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        a = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        a = arr
    else:
        raise ValueError(f"forme d'image non prise en charge: {arr.shape}")
    return Image.fromarray(a.astype(np.uint8), mode="RGB")


# ── repli local : corruptions plausibles en échographie ──────────────────────
def _speckle(a: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Bruit multiplicatif — l'artefact caractéristique de l'échographie."""
    sigma = (0.0, 0.08, 0.16, 0.26, 0.38, 0.52)[s]
    return a * (1.0 + rng.normal(0.0, sigma, a.shape))


def _blur(a: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Flou séparable — défaut de focalisation ou de résolution de sonde."""
    k = (0, 1, 2, 3, 4, 5)[s]
    if k == 0:
        return a
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    out = np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 0, a)
    return np.apply_along_axis(lambda m: np.convolve(m, ker, mode="same"), 1, out)


def _gain(a: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Décalage de gain et de contraste — réglage d'appareil."""
    g = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)[s]
    b = (0.0, 8.0, 16.0, 24.0, 32.0, 40.0)[s]
    return (a - 128.0) * g + 128.0 + b


def _gamma(a: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Compression dynamique — préréglage constructeur."""
    gm = (1.0, 1.2, 1.45, 1.75, 2.1, 2.5)[s]
    return 255.0 * np.power(np.clip(a, 0, 255) / 255.0, gm)


_LOCAL = {"speckle": _speckle, "blur": _blur, "gain": _gain, "gamma": _gamma}


def corrupt_local(arr: np.ndarray, kind: str, severity: int, seed: int) -> np.ndarray:
    """Applique une corruption de repli, de sévérité 0 (aucune) à 5."""
    if severity not in SEVERITIES:
        raise ValueError(f"sévérité invalide: {severity}; valeurs admises: {SEVERITIES}")
    if kind not in _LOCAL:
        raise ValueError(f"corruption locale inconnue: {kind}; choix: {tuple(_LOCAL)}")
    if severity == 0:
        return arr
    rng = np.random.default_rng(seed)
    out = _LOCAL[kind](arr.astype(np.float64), severity, rng)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_official(arr: np.ndarray, corruptor, severity: int) -> Image.Image:
    """Applique une instance MedMNIST-C (indices API 0–4, protocole 1–5)."""
    corrupted = corruptor.apply(_to_rgb(arr), severity - 1)
    return _to_rgb(np.asarray(corrupted))


def load(dataset: str, split: str = "test", size: int = 224,
         root: str = "data/raw") -> tuple[np.ndarray, np.ndarray]:
    """Charge un jeu MedMNIST. Retourne (images, étiquettes)."""
    import medmnist
    from medmnist import INFO

    Path(root).mkdir(parents=True, exist_ok=True)
    info = INFO[dataset]
    cls = getattr(medmnist, info["python_class"])
    try:
        ds = cls(split=split, download=True, root=root, size=size)
    except TypeError:  # versions antérieures de medmnist, sans MedMNIST+
        ds = cls(split=split, download=True, root=root)
    return ds.imgs, ds.labels.squeeze()


def build(dataset: str, severity: int, corruption: str = "speckle",
          source: str = "local", split: str = "test", size: int = 224,
          root: str = "data/raw", seed: int = 0) -> Cohort:
    """Construit une cohorte corrompue à sévérité donnée.

    Parameters
    ----------
    source : {"local", "medmnistc"}
        ``medmnistc`` utilise les corruptions officielles ; ``local`` le repli
        décrit en tête de module. Le choix est propagé dans les métriques.
    """
    if severity not in SEVERITIES:
        raise ValueError(f"sévérité invalide: {severity}; valeurs admises: {SEVERITIES}")
    if source not in {"local", "medmnistc"}:
        raise ValueError("source doit valoir 'local' ou 'medmnistc'")
    if source == "local" and corruption not in _LOCAL:
        raise ValueError(f"corruption locale inconnue: {corruption}; choix: {tuple(_LOCAL)}")

    imgs, labels = load(dataset, split=split, size=size, root=root)

    if severity == 0:
        out = [_to_rgb(a) for a in imgs]
    elif source == "medmnistc":
        from medmnistc.corruptions.registry import CORRUPTIONS_DS

        if dataset not in CORRUPTIONS_DS:
            raise ValueError(f"jeu absent du registre MedMNIST-C: {dataset}")
        if corruption not in CORRUPTIONS_DS[dataset]:
            choices = tuple(CORRUPTIONS_DS[dataset])
            raise ValueError(f"corruption MedMNIST-C inconnue: {corruption}; choix: {choices}")
        corruptor = CORRUPTIONS_DS[dataset][corruption]
        out = [_apply_official(a, corruptor, severity) for a in imgs]
    else:
        out = [_to_rgb(corrupt_local(a, corruption, severity, seed + i))
               for i, a in enumerate(imgs)]

    return Cohort(images=out, labels=np.asarray(labels), name=dataset,
                  severity=severity, corruption=f"{source}:{corruption}")
