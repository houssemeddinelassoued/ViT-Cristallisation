"""Tests du chargement et des corruptions P0 sans téléchargement de données."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tlsc.data.medmnistc import _apply_official, _to_rgb, corrupt_local, load


@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 1), (8, 8, 3)])
def test_conversion_rgb_accepte_les_formes_medmnist(shape: tuple[int, ...]) -> None:
    image = _to_rgb(np.zeros(shape, dtype=np.uint8))
    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_conversion_rgb_refuse_une_forme_ambigue() -> None:
    with pytest.raises(ValueError):
        _to_rgb(np.zeros((8, 8, 4), dtype=np.uint8))


def test_corruption_locale_reproductible_et_bornee() -> None:
    source = np.full((16, 16), 128, dtype=np.uint8)
    first = corrupt_local(source, "speckle", severity=3, seed=11)
    second = corrupt_local(source, "speckle", severity=3, seed=11)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8
    assert int(first.min()) >= 0 and int(first.max()) <= 255


@pytest.mark.parametrize(("kind", "severity"), [("inconnue", 1), ("speckle", 6)])
def test_corruption_locale_valide_ses_parametres(kind: str, severity: int) -> None:
    with pytest.raises(ValueError):
        corrupt_local(np.zeros((4, 4), dtype=np.uint8), kind, severity, seed=0)


def test_corruption_officielle_convertit_la_severite_en_index_api() -> None:
    class FakeCorruptor:
        def apply(self, image, severity: int) -> np.ndarray:
            assert image.mode == "RGB"
            assert severity == 0
            return np.asarray(image)

    image = _apply_official(np.zeros((4, 4), dtype=np.uint8), FakeCorruptor(), severity=1)
    assert image.mode == "RGB"


def test_load_cree_le_repertoire_racine(tmp_path, monkeypatch) -> None:
    root = tmp_path / "donnees" / "brutes"

    class FakeDataset:
        def __init__(self, **kwargs) -> None:
            assert kwargs["root"] == str(root)
            assert root.is_dir()
            self.imgs = np.zeros((1, 4, 4), dtype=np.uint8)
            self.labels = np.array([[1]])

    fake = SimpleNamespace(INFO={"fake": {"python_class": "FakeDataset"}},
                           FakeDataset=FakeDataset)
    monkeypatch.setitem(sys.modules, "medmnist", fake)
    images, labels = load("fake", root=str(root))
    assert images.shape == (1, 4, 4)
    assert labels.tolist() == 1