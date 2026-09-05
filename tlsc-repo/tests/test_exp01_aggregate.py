"""Tests du filtre de citabilité de l'agrégation.

Règle du dépôt : un run n'est citable que s'il est complet, réel et exécuté sur
un arbre git propre. L'agrégation ne contrôlait que les deux premiers points,
si bien qu'un run sali pouvait être retenu comme canonique et son chiffre
atteindre le site ou le papier sans qu'aucun garde-fou ne le signale.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.exp01_aggregate import lire_metrics, raison_rejet


def _run(tmp_path, nom: str, *, dirty: bool = False, dry: bool = False,
         scores: bool = True, metrics: bool = True):
    """Fabrique un dossier de run minimal sur disque."""
    d = tmp_path / nom
    d.mkdir()
    if metrics:
        (d / "metrics.json").write_text(json.dumps({
            "run_id": nom,
            "config": {"experiment": "exp01_zero_training", "dry_run": dry,
                       "severities": [0, 5]},
            "environment": {"git_dirty": dirty, "git_sha": "abc1234"},
        }), encoding="utf-8")
    if scores:
        np.savez(d / "scores.npz", F_0=np.zeros(3), H_0=np.zeros(3))
    return d


def test_run_propre_accepte(tmp_path) -> None:
    assert raison_rejet(_run(tmp_path, "propre")) is None


def test_run_sali_rejete(tmp_path) -> None:
    """Le cœur du garde-fou : un arbre sali disqualifie le run."""
    assert raison_rejet(_run(tmp_path, "sali", dirty=True)) == "git_dirty"


def test_run_factice_rejete(tmp_path) -> None:
    assert raison_rejet(_run(tmp_path, "factice", dry=True)) == "dry_run"


def test_run_incomplet_rejete(tmp_path) -> None:
    assert raison_rejet(_run(tmp_path, "sans_scores", scores=False)) == "incomplet"
    assert raison_rejet(_run(tmp_path, "sans_metrics", metrics=False)) == "incomplet"


def test_priorite_des_motifs(tmp_path) -> None:
    """Un run incomplet est signale comme tel avant tout autre motif."""
    d = _run(tmp_path, "cumul", dirty=True, dry=True, scores=False)
    assert raison_rejet(d) == "incomplet"


def test_lecture_metrics_repli_cp1252(tmp_path) -> None:
    """Les runs anterieurs au passage UTF-8 restent lisibles."""
    d = tmp_path / "ancien"
    d.mkdir()
    contenu = json.dumps({"note": "sévérité"}, ensure_ascii=False)
    (d / "metrics.json").write_bytes(contenu.encode("cp1252"))
    assert lire_metrics(d / "metrics.json")["note"] == "sévérité"


def test_collect_run_ecarte_un_run_sali(tmp_path) -> None:
    from experiments.exp01_aggregate import collect_run
    assert collect_run(_run(tmp_path, "sali2", dirty=True)) is None


def test_validation_du_fabricant(tmp_path) -> None:
    """Garde-fou du test lui-meme : le run propre doit bien etre complet."""
    d = _run(tmp_path, "controle")
    assert (d / "metrics.json").is_file() and (d / "scores.npz").is_file()
    with pytest.raises(KeyError):
        lire_metrics(d / "metrics.json")["absent"]
