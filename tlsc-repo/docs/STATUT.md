# Statut des sources du workspace

## Sources autoritatives

1. `../../02protocoleexperimentalv3.md` (racine du dépôt): protocole expérimental v3.
2. `docs/cadre-theorique.md`: notations et prédiction falsifiable implémentées.
3. `CLAUDE.md`: contraintes de développement et de reproductibilité.
4. `outputs/<run_id>/metrics.json`: seule source admissible pour une valeur mesurée.

## Documents antérieurs ou de présentation

- Les documents de cadrage historiques sont regroupés dans `../archive/`
  (2026-09-01) : `01  Feuille de Route Cristallisation.html`,
  `02  Chantier Cristallisation.html`, `03  Cristallisation Latente.html`,
  plans, guides et notes aux encadrants. Ils décrivent des versions antérieures
  du protocole ; en cas de divergence sur les cohortes, baselines ou
  identifiants d'ablation, la v3 prévaut.
- `plateforme-explicative.html` n'existe plus qu'en **une seule copie**,
  `../docs/plateforme-explicative.html` (site public) ; la copie racine a été
  retirée (2026-09-01). C'est un support pédagogique, pas une source de mesures.
- `thermodynamic_vit_infographic.html` a été retiré du projet (2026-09-01) : sa maquette
  décrivait une architecture différente (adaptation au moment du test, sortie anticipée)
  contredisant le cadre théorique canonique, avec des chiffres entièrement fabriqués.

## État d'implémentation

Le dépôt implémente le noyau Gibbs/CLIP, l'expérience P0 zero-shot
(`experiments/exp01_zero_training.py`), l'analyse statistique appariée
(`experiments/exp01_analysis.py` : DeLong + bootstrap stratifié),
l'agrégation multi-runs (`experiments/exp01_aggregate.py`), les détecteurs
bilatéral et bivarié de `tlsc/eval/metrics.py`, la sonde par couche
(`tlsc/models/layer_probe.py` + `experiments/exp02_layer_trajectories.py`,
exploratoire déclarée — profondeur N comme variable) et les régimes
d'ancrage R1/R2/R3 (`tlsc/models/data_anchors.py` +
`experiments/exp03_anchor_regimes.py`, centroïdes source et D_inter,
exploratoire déclarée). Les cohortes P0
couvertes sont BreastMNIST-C (7 corruptions officielles) et PneumoniaMNIST-C
(4 corruptions). La règle d'arrêt par échantillon, TPT,
C-TPT, l'adaptation visuelle, les cohortes P1/P2 et la calibration conforme
restent à implémenter et à valider avant toute revendication correspondante.

Le papier issu de l'expérience 1 se trouve dans `../../papers/contribution/` (main.tex + refs.bib) ;
chaque valeur y est rattachée à un run de `outputs/`.
