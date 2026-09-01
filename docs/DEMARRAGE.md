# Démarrage — de zéro à la première mesure

## 1. Environnement (30 min)

```bash
nvidia-smi                          # relever le GPU et la mémoire réellement disponibles
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1       # Windows PowerShell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt && pip install -e .
python -c "import torch;print(torch.cuda.is_available());print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

La chaîne fonctionne sur CPU pour les contrôles et les petits runs P0. Les campagnes
complètes doivent utiliser CUDA et adapter la taille de lot à la mémoire réellement
disponible; ne pas supposer une RTX 4070 12 Go.

## 2. Vérifier le noyau AVANT toute expérience (1 min)

```bash
pytest -q
```

Les tests vérifient l'identité thermodynamique en double précision, l'invariance
de H et la covariance de F (Proposition 2), le corollaire clinique, et l'équivalence
exacte entre l'inférence zero-shot de CLIP et la mesure de Gibbs. **Aucune expérience
ne se lance sur un test rouge.**

## 3. Valider la chaîne sans rien télécharger (2 min)

```bash
python -m experiments.exp01_zero_training --dry-run --limit 500
```

Embeddings synthétiques. Produit un `outputs/<run_id>/` complet, marqué `dry_run: true`.
Cela ne prouve que la tuyauterie.

## 4. La première mesure réelle (quelques heures)

```bash
python -m experiments.exp01_zero_training \
  --dataset breastmnist --model ViT-B-16-quickgelu --pretrained openai \
    --corruption speckle --source local --severities 0 1 2 3 4 5
```

Puis, pour la figure publiable, avec les corruptions officielles :

Sous Windows, `medmnistc` requiert aussi le runtime DLL ImageMagick et les variables
`MAGICK_HOME` et `PATH`. La version Python est fixée dans `requirements.txt`; utiliser
`speckle_noise`, nom exact du registre officiel :

```powershell
$env:MAGICK_HOME = "C:\chemin\vers\ImageMagick"
$env:PATH = "$env:MAGICK_HOME;$env:PATH"
python -m experiments.exp01_zero_training --dataset breastmnist `
  --model ViT-B-16-quickgelu --pretrained openai `
  --source medmnistc --corruption speckle_noise --severities 0 1 3 5
```

## 5. Lire le résultat

Ouvrir `outputs/<run_id>/figures/02_detection.png` et le champ `verdict` de
`metrics.json`.

**La prédiction du cadre théorique** : `auroc_F` doit croître nettement avec la
sévérité tandis que `auroc_H` reste proche de 0,5.

- Si elle tient : reporter les valeurs dans `docs/JOURNAL.md` et dans le tableau 8.1
  du protocole, puis enchaîner sur les sondes linéaires par couche.
- Si elle ne tient pas : c'est une réfutation. La consigner telle quelle, vérifier
  d'abord qu'il ne s'agit pas d'un problème d'invites (essayer `--no-ensemble` et
  d'autres formulations sur la validation source), puis en tirer les conséquences
  sur la narration de l'axe.

## 6. Ensuite

`--dataset pathmnist` pour l'histologie, puis les sondes linéaires par couche
(trajectoire de l'entropie en profondeur, jalon J1).
