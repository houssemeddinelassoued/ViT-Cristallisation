# Statut des sources du workspace

## Sources autoritatives

1. `../02protocoleexperimentalv3.md` dans le workspace: protocole expérimental v3.
2. `docs/cadre-theorique.md`: notations et prédiction falsifiable implémentées.
3. `CLAUDE.md`: contraintes de développement et de reproductibilité.
4. `outputs/<run_id>/metrics.json`: seule source admissible pour une valeur mesurée.

## Documents antérieurs ou de présentation

- `01  Feuille de Route Cristallisation.html`, `02  Chantier Cristallisation.html` et
  `03  Cristallisation Latente.html` décrivent une version antérieure du protocole.
  En cas de divergence sur les cohortes, baselines ou identifiants d'ablation, la v3
  prévaut.
- `Cristallisation Latente.html` et `Cristallisation Latente_files/` sont un export de
  navigateur d'un artefact Claude, pas une source scientifique maintenable.
- `thermodynamic_vit_infographic.html` a été retiré du projet (2026-09-01) : sa maquette
  décrivait une architecture différente (adaptation au moment du test, sortie anticipée)
  contredisant le cadre théorique canonique, avec des chiffres entièrement fabriqués.

## État d'implémentation

Le dépôt implémente le noyau Gibbs/CLIP, l'expérience P0 zero-shot
(`experiments/exp01_zero_training.py`), l'analyse statistique appariée
(`experiments/exp01_analysis.py` : DeLong + bootstrap stratifié) et
l'agrégation multi-runs (`experiments/exp01_aggregate.py`), y compris le
détecteur bilatéral `two_sided_shift_score` de `tlsc/eval/metrics.py`
(médiane calibrée sur la source seule). Les cohortes P0 couvertes sont
BreastMNIST-C (7 corruptions officielles) et PneumoniaMNIST-C (4 corruptions).
TPT, C-TPT, l'adaptation visuelle, la sortie anticipée, les cohortes P1/P2 et
la calibration conforme restent à implémenter et à valider avant toute
revendication correspondante.

Le papier issu de l'expérience 1 se trouve dans `paper/` (main.tex + refs.bib) ;
chaque valeur y est rattachée à un run de `outputs/`.
