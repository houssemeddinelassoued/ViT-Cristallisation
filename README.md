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

Le cadre et les seuils pré-enregistrés sont définis dans `docs/cadre-theorique.md`.
`docs/STATUT.md` indique quels documents du workspace sont autoritatifs.
