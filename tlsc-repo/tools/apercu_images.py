"""Aperçu visuel des cohortes P0 — exporte en PNG ce que l'encodeur voit.

Outil de **visualisation seule** : il ne produit aucune mesure, n'écrit rien dans
`outputs/` et ne modifie aucun module de `tlsc/` ni de `experiments/`. Les pixels
exportés traversent exactement les mêmes fonctions que le pipeline
(`tlsc.data.medmnistc`), si bien qu'une image d'aperçu à la sévérité *s* est
l'image encodée par CLIP à cette sévérité — au pixel près.

Les images sources sont déjà présentes dans `data/raw/*.npz` (MedMNIST, taille
224). Aucun téléchargement n'est nécessaire ; `medmnist` retombe sur le cache
local.

Exemples
--------
Six images de BreastMNIST sous chaque corruption officielle, sévérités 0 à 5 ::

    .venv\\Scripts\\python.exe -m tools.apercu_images --dataset breastmnist \\
        --all-corruptions --limit 6

Toutes les images du split test sous une seule corruption ::

    .venv\\Scripts\\python.exe -m tools.apercu_images --dataset pneumoniamnist \\
        --corruption gaussian_blur --limit 0 --force
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tlsc.data import medmnistc as mc

#: Au-delà de ce nombre de PNG, l'export exige `--force` (garde-fou disque).
SEUIL_ALERTE_FICHIERS = 5000

#: Nombre de colonnes maximal d'une planche de contact.
MAX_COLONNES_PLANCHE = 8


def noms_de_classes(dataset: str) -> dict[int, str]:
    """Retourne la correspondance étiquette entière → nom de classe MedMNIST.

    Parameters
    ----------
    dataset : str
        Identifiant MedMNIST, par exemple ``"breastmnist"``.

    Returns
    -------
    dict of {int: str}
        Noms de classes tels que publiés par MedMNIST.
    """
    from medmnist import INFO

    if dataset not in INFO:
        raise ValueError(f"jeu MedMNIST inconnu: {dataset}")
    return {int(k): v for k, v in INFO[dataset]["label"].items()}


def corruptions_officielles(dataset: str) -> tuple[str, ...]:
    """Liste les corruptions MedMNIST-C déclarées pour un jeu.

    Parameters
    ----------
    dataset : str
        Identifiant MedMNIST.

    Returns
    -------
    tuple of str
        Noms de corruptions du registre officiel.
    """
    from medmnistc.corruptions.registry import CORRUPTIONS_DS

    if dataset not in CORRUPTIONS_DS:
        raise ValueError(f"jeu absent du registre MedMNIST-C: {dataset}")
    return tuple(CORRUPTIONS_DS[dataset])


def indices_stratifies(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Tire un échantillon d'indices en servant les classes à tour de rôle.

    Le tirage est déterministe à graine fixée et garantit la présence de chaque
    classe tant que ``limit`` est au moins égal au nombre de classes.

    Parameters
    ----------
    labels : ndarray of shape (n,)
        Étiquettes entières de la cohorte.
    limit : int
        Nombre d'images voulues ; ``0`` ou une valeur supérieure à ``n`` prend tout.
    seed : int
        Graine du tirage.

    Returns
    -------
    ndarray of int
        Indices triés par ordre croissant.
    """
    n = len(labels)
    if limit < 0:
        raise ValueError(f"limit doit etre positif ou nul, reçu {limit}")
    if limit == 0 or limit >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    piles = [list(rng.permutation(np.flatnonzero(labels == c))) for c in np.unique(labels)]
    choisis: list[int] = []
    while len(choisis) < limit and any(piles):
        for pile in piles:
            if pile and len(choisis) < limit:
                choisis.append(int(pile.pop()))
    return np.sort(np.asarray(choisis, dtype=int))


def image_a_severite(
    imgs: np.ndarray,
    index: int,
    corruption: str,
    source: str,
    severity: int,
    seed: int,
    corruptor: Any | None,
) -> Image.Image:
    """Reproduit à l'identique la corruption appliquée par `medmnistc.build`.

    Parameters
    ----------
    imgs : ndarray of shape (n, H, W)
        Images brutes du split.
    index : int
        Position de l'image dans ``imgs`` — c'est aussi le décalage de graine
        utilisé par le repli local dans le pipeline.
    corruption : str
        Nom de la corruption.
    source : {"local", "medmnistc"}
        Source de corruption, propagée telle quelle depuis la ligne de commande.
    severity : int
        Sévérité de 0 (aucune) à 5.
    seed : int
        Graine de base du repli local.
    corruptor : object or None
        Instance MedMNIST-C, requise si ``source`` vaut ``"medmnistc"``.

    Returns
    -------
    PIL.Image.Image
        Image RGB, telle que présentée au préprocesseur CLIP.
    """
    arr = imgs[index]
    if severity == 0:
        return mc._to_rgb(arr)
    if source == "medmnistc":
        if corruptor is None:
            raise ValueError("corruptor manquant pour la source 'medmnistc'")
        return mc._apply_official(arr, corruptor, severity)
    return mc._to_rgb(mc.corrupt_local(arr, corruption, severity, seed + index))


def planche_de_contact(
    par_severite: dict[int, list[Image.Image]],
    etiquettes: list[int],
    noms: dict[int, str],
    titre: str,
    chemin: Path,
) -> None:
    """Écrit une planche de contact : une ligne par sévérité, une colonne par image.

    Parameters
    ----------
    par_severite : dict of {int: list of PIL.Image.Image}
        Images déjà corrompues, indexées par sévérité.
    etiquettes : list of int
        Étiquettes des colonnes, dans l'ordre des images.
    noms : dict of {int: str}
        Noms de classes pour l'entête de colonne.
    titre : str
        Titre de la planche.
    chemin : Path
        Fichier PNG de destination.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    severites = sorted(par_severite)
    n_col = min(len(etiquettes), MAX_COLONNES_PLANCHE)
    fig, axes = plt.subplots(
        len(severites), n_col, figsize=(1.45 * n_col, 1.62 * len(severites)), squeeze=False
    )
    for ligne, s in enumerate(severites):
        for col in range(n_col):
            ax = axes[ligne][col]
            ax.imshow(par_severite[s][col], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if ligne == 0:
                ax.set_title(noms.get(etiquettes[col], str(etiquettes[col])), fontsize=7)
            if col == 0:
                ax.set_ylabel(f"sév. {s}", fontsize=8)
    fig.suptitle(titre, fontsize=10)
    fig.tight_layout()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=120)
    plt.close(fig)


def grille_corruptions(
    dataset: str,
    split: str,
    source: str,
    corruptions: list[str],
    severites: list[int],
    index: int,
    seed: int,
    size: int,
    root: str,
    chemin: Path,
    transparent: bool,
) -> dict[str, Any]:
    """Écrit une grille corruption × sévérité pour une seule image.

    Figure pédagogique : chaque ligne est une corruption, chaque colonne une
    sévérité, sur la même image de départ. À fond transparent et libellés gris
    moyen, elle reste lisible sur une page à thème clair comme sombre.

    Parameters
    ----------
    index : int
        Indice de l'image dans le split ; ``-1`` prend la première image de la
        première classe rencontrée.
    transparent : bool
        Fond transparent et libellés gris moyen plutôt que le style matplotlib.
    chemin : Path
        Fichier PNG de destination.

    Returns
    -------
    dict
        Manifeste de provenance, écrit à côté de la figure en ``.json``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs, labels = mc.load(dataset, split=split, size=size, root=root)
    if index < 0:
        index = int(indices_stratifies(np.asarray(labels), 1, seed)[0])
    if not 0 <= index < len(imgs):
        raise ValueError(f"indice hors bornes: {index}; le split compte {len(imgs)} images")
    noms = noms_de_classes(dataset)

    couleur = "#8b96ad" if transparent else "black"
    fig, axes = plt.subplots(
        len(corruptions),
        len(severites),
        figsize=(1.22 * len(severites), 1.34 * len(corruptions)),
        squeeze=False,
    )
    for ligne, corruption in enumerate(corruptions):
        corruptor = None
        if source == "medmnistc":
            from medmnistc.corruptions.registry import CORRUPTIONS_DS

            if corruption not in CORRUPTIONS_DS[dataset]:
                choix = corruptions_officielles(dataset)
                raise ValueError(f"corruption MedMNIST-C inconnue: {corruption}; choix: {choix}")
            corruptor = CORRUPTIONS_DS[dataset][corruption]
        for col, s in enumerate(severites):
            ax = axes[ligne][col]
            ax.imshow(
                image_a_severite(imgs, index, corruption, source, s, seed, corruptor),
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for cote in ax.spines.values():
                cote.set_color(couleur)
                cote.set_linewidth(0.5)
            if ligne == 0:
                ax.set_title(f"sév. {s}", fontsize=8, color=couleur)
            if col == 0:
                ax.set_ylabel(corruption, fontsize=7.5, color=couleur)
    fig.tight_layout()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chemin, dpi=110, transparent=transparent,
                facecolor="none" if transparent else "white")
    plt.close(fig)

    manifeste = {
        "avertissement": (
            "Illustration pedagogique — aucune mesure. Les valeurs publiables viennent "
            "de outputs/<run_id>/metrics.json."
        ),
        "figure": chemin.name,
        "dataset": dataset,
        "split": split,
        "source": source,
        "corruptions": corruptions,
        "severites": severites,
        "index_image": index,
        "classe": noms.get(int(labels[index]), str(int(labels[index]))),
        "image_size": size,
        "seed": seed,
        "genere_utc": datetime.now(UTC).isoformat(),
        "commande": (
            f"python -m tools.apercu_images --dataset {dataset} --split {split} "
            f"--source {source} --grid --grid-index {index} "
            f"--corruption {' '.join(corruptions)} "
            f"--severities {' '.join(str(s) for s in severites)}"
        ),
    }
    chemin.with_suffix(".json").write_text(
        json.dumps(manifeste, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[apercu] grille -> {chemin}")
    return manifeste


def exporter(
    dataset: str,
    split: str,
    source: str,
    corruptions: list[str],
    severites: list[int],
    limit: int,
    seed: int,
    size: int,
    root: str,
    sortie: Path,
    ecrire_png: bool,
    ecrire_planche: bool,
    force: bool,
) -> dict[str, Any]:
    """Exporte l'aperçu et retourne le manifeste écrit à côté des images.

    Returns
    -------
    dict
        Manifeste de provenance, également sérialisé dans ``apercu.json``.
    """
    for s in severites:
        if s not in mc.SEVERITIES:
            raise ValueError(f"sévérité invalide: {s}; valeurs admises: {mc.SEVERITIES}")
    if source not in {"local", "medmnistc"}:
        raise ValueError("source doit valoir 'local' ou 'medmnistc'")

    imgs, labels = mc.load(dataset, split=split, size=size, root=root)
    indices = indices_stratifies(np.asarray(labels), limit, seed)
    noms = noms_de_classes(dataset)

    total = len(indices) * len(severites) * len(corruptions) if ecrire_png else 0
    if total > SEUIL_ALERTE_FICHIERS and not force:
        raise ValueError(
            f"export de {total} PNG demande --force (seuil {SEUIL_ALERTE_FICHIERS}); "
            "reduisez --limit ou passez --no-png pour ne garder que les planches"
        )

    racine = sortie / dataset / split
    manifeste: dict[str, Any] = {
        "avertissement": (
            "Apercu visuel — aucune mesure. Ne jamais citer ces fichiers comme resultat ; "
            "les valeurs publiables viennent de outputs/<run_id>/metrics.json."
        ),
        "dataset": dataset,
        "split": split,
        "source": source,
        "corruptions": corruptions,
        "severites": severites,
        "image_size": size,
        "limit": limit,
        "seed": seed,
        "indices": [int(i) for i in indices],
        "labels": {int(i): int(labels[i]) for i in indices},
        "noms_de_classes": {str(k): v for k, v in noms.items()},
        "genere_utc": datetime.now(UTC).isoformat(),
        "source_donnees": f"{root}/{dataset}_{size}.npz",
    }

    for corruption in corruptions:
        corruptor = None
        if source == "medmnistc":
            from medmnistc.corruptions.registry import CORRUPTIONS_DS

            if corruption not in CORRUPTIONS_DS[dataset]:
                choix = corruptions_officielles(dataset)
                raise ValueError(f"corruption MedMNIST-C inconnue: {corruption}; choix: {choix}")
            corruptor = CORRUPTIONS_DS[dataset][corruption]

        par_severite: dict[int, list[Image.Image]] = {}
        for s in severites:
            lot = [
                image_a_severite(imgs, int(i), corruption, source, s, seed, corruptor)
                for i in indices
            ]
            par_severite[s] = lot
            if ecrire_png:
                dossier = racine / f"{source}_{corruption}" / f"sev{s}"
                dossier.mkdir(parents=True, exist_ok=True)
                for i, img in zip(indices, lot, strict=True):
                    img.save(dossier / f"idx{int(i):04d}_classe{int(labels[i])}.png")
            print(f"[apercu] {dataset}/{split} {source}:{corruption} sev{s} -> {len(lot)} images")

        if ecrire_planche:
            chemin = racine / f"planche_{source}_{corruption}.png"
            planche_de_contact(
                par_severite,
                [int(labels[i]) for i in indices],
                noms,
                f"{dataset} · {split} · {source}:{corruption}",
                chemin,
            )
            print(f"[apercu] planche -> {chemin}")

    racine.mkdir(parents=True, exist_ok=True)
    (racine / "apercu.json").write_text(
        json.dumps(manifeste, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifeste


def construire_parseur() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments de l'outil."""
    p = argparse.ArgumentParser(
        prog="tools.apercu_images",
        description="Exporte en PNG les images des cohortes P0, telles que vues par l'encodeur.",
    )
    p.add_argument("--dataset", default="breastmnist", help="jeu MedMNIST (defaut: breastmnist)")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--source", default="medmnistc", choices=("local", "medmnistc"))
    p.add_argument("--corruption", nargs="*", default=["speckle_noise"],
                   help="une ou plusieurs corruptions (defaut: speckle_noise)")
    p.add_argument("--all-corruptions", action="store_true",
                   help="balaye toutes les corruptions officielles du jeu")
    p.add_argument("--severities", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--limit", type=int, default=6,
                   help="nombre d'images echantillonnees ; 0 = toutes (defaut: 6)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--root", default="data/raw", help="cache MedMNIST (defaut: data/raw)")
    p.add_argument("--out", default="data/apercu", help="dossier de sortie (defaut: data/apercu)")
    p.add_argument("--grid", action="store_true",
                   help="grille corruption x severite sur une seule image (figure pedagogique)")
    p.add_argument("--grid-index", type=int, default=-1,
                   help="indice de l'image de la grille ; -1 = tirage stratifie (defaut)")
    p.add_argument("--grid-name", default=None,
                   help="nom du PNG de la grille (defaut: grille_<jeu>_<source>.png)")
    p.add_argument("--transparent", action="store_true",
                   help="fond transparent et libelles gris moyen, pour page claire ou sombre")
    p.add_argument("--no-png", action="store_true", help="n'ecrire que les planches de contact")
    p.add_argument("--no-sheet", action="store_true", help="n'ecrire que les PNG individuels")
    p.add_argument("--force", action="store_true",
                   help=f"autorise un export de plus de {SEUIL_ALERTE_FICHIERS} fichiers")
    return p


def main(argv: list[str] | None = None) -> int:
    """Point d'entree ligne de commande."""
    args = construire_parseur().parse_args(argv)
    if args.all_corruptions:
        corruptions = (
            list(mc._LOCAL)
            if args.source == "local"
            else list(corruptions_officielles(args.dataset))
        )
    else:
        corruptions = list(args.corruption)
    if not corruptions:
        raise ValueError("aucune corruption demandee")

    if args.grid:
        nom = args.grid_name or f"grille_{args.dataset}_{args.source}.png"
        grille_corruptions(
            dataset=args.dataset,
            split=args.split,
            source=args.source,
            corruptions=corruptions,
            severites=list(args.severities),
            index=args.grid_index,
            seed=args.seed,
            size=args.image_size,
            root=args.root,
            chemin=Path(args.out) / nom,
            transparent=args.transparent,
        )
        return 0

    manifeste = exporter(
        dataset=args.dataset,
        split=args.split,
        source=args.source,
        corruptions=corruptions,
        severites=list(args.severities),
        limit=args.limit,
        seed=args.seed,
        size=args.image_size,
        root=args.root,
        sortie=Path(args.out),
        ecrire_png=not args.no_png,
        ecrire_planche=not args.no_sheet,
        force=args.force,
    )
    print(f"[apercu] manifeste : {len(manifeste['indices'])} images, "
          f"{len(manifeste['corruptions'])} corruption(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
