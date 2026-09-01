"""Validation des paramètres et du contrat de l'expérience 1."""
from __future__ import annotations

from argparse import Namespace

import pytest

from experiments.exp01_zero_training import validate_args


def _args(severities: list[int], batch_size: int = 8, limit: int = 0) -> Namespace:
    return Namespace(severities=severities, batch_size=batch_size, limit=limit)


def test_balaye_source_puis_severites_croissantes() -> None:
    validate_args(_args([0, 1, 3, 5]))


@pytest.mark.parametrize(
    "severities",
    ([1, 3, 5], [0, 3, 1], [0, 1, 1], [0, 6]),
)
def test_refuse_un_balayage_non_interpretable(severities: list[int]) -> None:
    with pytest.raises(ValueError):
        validate_args(_args(severities))


@pytest.mark.parametrize(("batch_size", "limit"), [(0, 0), (-1, 0), (8, -1)])
def test_refuse_les_tailles_invalides(batch_size: int, limit: int) -> None:
    with pytest.raises(ValueError):
        validate_args(_args([0, 1], batch_size=batch_size, limit=limit))