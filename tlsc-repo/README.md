# TLSC — Cristallisation de l'espace latent

Adaptation au moment du test et arrêt anticipé à risque contrôlé, sur encodeur
vision-langage gelé, pour l'imagerie médicale sous décalage d'acquisition.

## Installation

Deux environnements coexistent. `.venv-cuda` est celui à utiliser ; `.venv` n'est conservé
que pour reproduire les runs antérieurs au 2026-09-05.

| Environnement | torch | Usage |
|---|---|---|
| `.venv-cuda` | 2.4.1+cu124 | **Par défaut** — isolé, tout nouveau run |
| `.venv` | 2.4.1 (CPU) | Hérite du conda `medrag` ; ne jamais y toucher à torch |

Il n'y a **pas de CUDA Toolkit à installer** : le pilote NVIDIA suffit, les bibliothèques
d'exécution CUDA sont embarquées dans les paquets PyTorch. La version de torch est
**épinglée** à celle de `.venv` pour que le passage à la carte ne change que le matériel.

```powershell
Set-Location tlsc-repo
C:\Users\<vous>\.conda\envs\medrag\python.exe -m venv .venv-cuda   # 3.11, sans heritage
.venv-cuda\Scripts\python.exe -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
.venv-cuda\Scripts\python.exe -m pip install -r requirements.txt
.venv-cuda\Scripts\python.exe -m pip install -e .
```

Vérification obligatoire — ne pas continuer sans ces quatre lignes :

```bash
.venv-cuda/Scripts/python.exe -c "import torch;print(torch.__version__);print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));print('bf16:',torch.cuda.is_bf16_supported())"
```

**Si le téléchargement de torch se fige** (2,5 Go sur connexion lente ; pip reste bloqué
sur une socket morte, sans erreur ni progression), le récupérer d'abord avec un
téléchargement reprenable, puis installer le fichier local :

```bash
curl -L -C - --retry 30 --speed-limit 10240 --speed-time 60 \
  -o torch-2.4.1+cu124-cp311-cp311-win_amd64.whl \
  "https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-cp311-cp311-win_amd64.whl"
.venv-cuda/Scripts/python.exe -m pip install torch-2.4.1+cu124-cp311-cp311-win_amd64.whl
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

**État (runs rejoués le 2026-09-05)** : l'inversion photométrique de F était une
propriété des ancres textuelles — elle disparaît dès k = 16 (brightness_down :
AUROC(F) 0,360 → 0,845) ; la balanced accuracy source passe de 0,517 (R1) à 0,687 (R3).
Détails : `docs/rapport-exp03.md`.

## Expérience 4 — arrêt anticipé calibré (« sortir quand H < ε »)

```bash
python -m experiments.exp04_early_exit --dataset breastmnist --source medmnistc --corruption speckle_noise --severities 0 1 2 3 4 5
```

Règle d'arrêt par échantillon du croquis fondateur (`tlsc/eval/early_exit.py`) :
seuils H/F et couche fixe calibrés sur une moitié stratifiée du split test source,
évaluation sur l'autre moitié, sévérités 0–5. Exploratoire déclarée.

**État (2026-09-01, invalidé le 2026-09-05)** : cette expérience calibre sur
l'exactitude *brute*. L'exp05 a montré que celle-ci vaut 0,731 à onze couches sur douze
sous ancres R1 et 0,718 à la douzième, soit le taux de la classe majoritaire ou moins : l'« exactitude préservée » et
les ≈ 92 % de calcul économisé ne mesurent donc rien de diagnostique. Les conclusions
d'efficience sont reprises sur critère **équilibré** dans l'exp05. Détails et
avertissement : `docs/rapport-exp04.md`.

## Expérience 5 — ancres × arrêt calibré (test non trivial de « argmin_N H »)

```bash
python -m experiments.exp05_anchored_early_exit --dataset breastmnist --source medmnistc --corruption speckle_noise --severities 0 1 2 3 4 5 --k-shot 16
```

Croisement des expériences 3 et 4 : l'arrêt calibré est rejoué sous les trois régimes
d'ancrage, avec des centroïdes R2/R3 **recalculés couche par couche** sur le split train
source. L'exp04 avait trouvé la règle dégénérée (N* = 1) sous ancres textuelles quasi
aveugles ; le test n'est non trivial que là où l'exactitude varie réellement avec la
profondeur. Trois mesures : profondeur utile par régime, AUROC de la profondeur de
sortie N* comme détecteur de décalage (`exit_depth_auroc`), et critère de qualité
d'ancrage (`anchor_quality` : `Q_gap` = D_inter / distance ancres-nuage, `Q_fisher` =
D_inter / dispersion intra-classe). Exploratoire déclarée.

Anti-fuite en trois barrières : centroïdes sur le train source à sévérité 0 ; seuils ε
et couche fixe sur la moitié calibration du test source ; évaluations sur l'autre
moitié seulement. `--train-limit` borne le coût d'encodage du split train.

**État (consolidé sur 5 graines le 2026-09-05)** : la dégénérescence N* = 1 de l'exp04
est levée, et elle avait deux causes — des ancres textuelles au hasard (exactitude
équilibrée dans [0,474 ; 0,521] quelle que soit la couche) *et* un critère de calibration
brut que le prédicteur majoritaire maximise. Ce qui **survit à cinq graines** : une couche
intermédiaire bat la couche de sortie sous ancres de données, dans les quatre
configurations (+0,045 à +0,134). Ce qui **ne survit pas** : la localisation de la couche
optimale (±3 couches) ; le gain sous décalage, dont le signe dépend du type de corruption,
positif sur le flou (+0,151 ± 0,111) et négatif sur les photométriques (−0,038 ± 0,042) ;
et la supériorité de la profondeur fixe sur la règle par échantillon, qui suit la même
dichotomie (20 comparaisons gagnées sur 40). Agrégation multi-graines :
`python -m experiments.exp05_aggregate outputs`. Détails : `docs/rapport-exp05.md`.

## Voir les images — aperçu visuel des cohortes

```bash
python -m tools.apercu_images --dataset breastmnist --all-corruptions --limit 6
python -m tools.apercu_images --dataset pneumoniamnist --corruption gaussian_blur --limit 0 --force
```

Exporte en PNG ce que l'encodeur voit : une planche de contact par corruption (lignes =
sévérités 0–5) et les images individuelles, dans `data/apercu/<jeu>/<split>/` — dossier
**ignoré par git**, régénérable en une commande. Les images sources sont déjà dans
`data/raw/*.npz` : aucun téléchargement.

Les pixels passent par les mêmes fonctions que le pipeline (`tlsc.data.medmnistc`), donc
une image d'aperçu à sévérité *s* est exactement l'image encodée par CLIP à cette
sévérité. `--limit 0` exporte tout le split (`--force` au-delà de 5 000 fichiers) ;
`--no-png` ne garde que les planches. Le mode `--grid` produit la grille corruption x
sévérité qui illustre la section 6 du site — ces illustrations vivent dans
`docs/assets/apercu/`, avec leur manifeste `.json` :

```bash
python -m tools.apercu_images --dataset breastmnist --all-corruptions --grid --transparent \
    --out ../docs/assets/apercu --grid-name apercu_corruptions_breastmnist.png
```

**Aucune mesure** n'est produite : chaque sortie dépose son manifeste (`apercu.json` pour
les planches, un `.json` homonyme pour les grilles) et ces fichiers ne se citent jamais
comme résultat.

Le cadre et les seuils pré-enregistrés sont définis dans `docs/cadre-theorique.md`.
`docs/STATUT.md` indique quels documents du workspace sont autoritatifs.
