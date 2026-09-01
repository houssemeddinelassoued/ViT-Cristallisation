# TLSC — Cristallisation de l'espace latent

Adaptation au moment du test et arrêt anticipé à risque contrôlé, sur encodeur
vision-langage gelé, pour l'imagerie médicale sous décalage d'acquisition.

## Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e .
```

Vérification obligatoire — ne pas continuer sans ces quatre lignes :

```bash
python -c "import torch;print(torch.__version__);print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));print('bf16:',torch.cuda.is_bf16_supported())"
```

## Tests

```bash
pytest            # doit être vert avant toute expérience
```

Ils vérifient l'identité thermodynamique F = <E> - T*H à la précision machine,
l'invariance de H et la covariance de F (Proposition 2), et l'équivalence exacte entre
l'inférence zero-shot de CLIP et la mesure de Gibbs.

## Expérience 1 — détection du décalage, sans entraînement

```bash
python -m experiments.exp01_zero_training --dataset breastmnist --model ViT-B-16
```

Aucun entraînement, aucune annotation cible. Produit `outputs/<run_id>/metrics.json`,
les figures, et un enregistrement complet de la configuration.

**Prédiction à falsifier** : sous décalage croissant, l'énergie libre F sépare le domaine
source du domaine corrompu (AUROC élevée) tandis que l'entropie H échoue (AUROC proche de
0,5). C'est le corollaire de la Proposition 2. Si la prédiction ne tient pas, le résultat
se consigne tel quel — c'est une réfutation, pas un bug.

**État (2026-09-01)** : prédiction forte **réfutée** sur les 11 configurations
canoniques (H jamais au hasard). Résultat exploitable : F est un détecteur **signé**
— dominant sur les corruptions structurelles, sous le hasard sur les photométriques —
et la variante bilatérale `two_sided_shift_score` récupère 5 inversions sur 6.
Détails : `docs/rapport-exp01.md`, `docs/JOURNAL.md`, papier dans `../papers/contribution/`.

Analyse statistique appariée et agrégation multi-runs :

```bash
python -m experiments.exp01_analysis outputs     # DeLong + bootstrap → analysis.json
python -m experiments.exp01_aggregate outputs    # tableau + figures → outputs/aggregate/
```

## Expérience 2 — trajectoires par couche (profondeur N comme variable)

```bash
python -m experiments.exp02_layer_trajectories --dataset breastmnist --source medmnistc --corruption speckle_noise --severities 0 1 2 3 4 5
```

Sonde « logit lens » (`tlsc/models/layer_probe.py`) : embedding CLS après chacun des
12 blocs du ViT visuel, puis H_n, F_n, chi_n et détection par couche. Expérience
**exploratoire déclarée** (pas de seuil pré-enregistré), issue du croquis fondateur
`argmin_N H` de l'encadrant.

**État (2026-09-01)** : le signal de décalage est plus fort en profondeur intermédiaire
qu'en sortie (speckle : couches 3–7 ; brightness_down : couche 1, où l'inversion de
signe de la couche 12 disparaît), mais la meilleure couche dépend de la corruption ;
H_n n'est pas monotone sur la source (minimum en couche 4). Détails :
`docs/rapport-exp02.md`.

## Expérience 3 — régimes d'ancrage (invites vs centroïdes de données)

```bash
python -m experiments.exp03_anchor_regimes --dataset breastmnist --source medmnistc --corruption speckle_noise --severities 0 1 2 3 4 5 --k-shot 16
```

Compare R1 (invites textuelles), R2 (centroïdes few-shot) et R3 (centroïdes sur le
train source complet) : D_inter, classification et détection F/H par sévérité
(`tlsc/models/data_anchors.py`). Anti-fuite : centroïdes calculés sur le split train
SOURCE à sévérité 0 uniquement. Exploratoire déclarée.

**État (2026-09-01)** : l'inversion photométrique de F était une propriété des ancres
textuelles — elle disparaît dès k = 16 (brightness_down : AUROC(F) 0,359 → 0,845) ;
la balanced accuracy source passe de 0,517 (R1) à 0,692 (R3). Détails :
`docs/rapport-exp03.md`.

## Expérience 4 — arrêt anticipé calibré (« sortir quand H < ε »)

```bash
python -m experiments.exp04_early_exit --dataset breastmnist --source medmnistc --corruption speckle_noise --severities 0 1 2 3 4 5
```

Règle d'arrêt par échantillon du croquis fondateur (`tlsc/eval/early_exit.py`) :
seuils H/F et couche fixe calibrés sur une moitié stratifiée du split test source,
évaluation sur l'autre moitié, sévérités 0–5. Exploratoire déclarée.

**État (2026-09-01)** : sortie en couche ≈ 1 à exactitude préservée (≈ 92 % de calcul
économisé) — mais trivialement, les ancres R1 étant quasi aveugles ; à refaire sous
ancres R2/R3. Résultat exploitable : la profondeur de sortie N* croît avec la
sévérité (1,00 → 2,79 sur brightness_down) — signal de dérive gratuit. Détails :
`docs/rapport-exp04.md`.

Le cadre et les seuils pré-enregistrés sont définis dans `docs/cadre-theorique.md`.
`docs/STATUT.md` indique quels documents du workspace sont autoritatifs.
