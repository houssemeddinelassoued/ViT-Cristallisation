# Rapport de résultats — Expérience 2 (trajectoires par couche, cohorte P0)

**Date du rapport** : 2026-09-01
**Expérience** : `exp02_layer_trajectories` — observables de Gibbs H_n, F_n, χ_n après
chacun des 12 blocs du ViT visuel (sonde « logit lens », `tlsc/models/layer_probe.py`).
**Modèle** : ViT-B-16-quickgelu · poids `openai` (CLIP gelé), T = 2τ ≈ 0,0200
**Cohorte** : BreastMNIST 224×224, split test, n = 156, corruptions MedMNIST-C, sévérités 0–5
**Statut épistémique** : **exploratoire déclaré** — aucun seuil pré-enregistré ; attente
qualitative non contraignante consignée a priori dans le docstring du script : « H_n décroît
avec la profondeur sur la source ».

Origine : croquis fondateur de l'encadrant — la profondeur N comme variable d'optimisation
(`argmin_N H`), cf. `note-meeting.md` à la racine du workspace.

Règle du dépôt : seul un chiffre présent dans un `outputs/<run_id>/metrics.json` existe.

---

## 1. Runs

| run_id | Corruption | Statut |
|---|---|---|
| 20260901T165926Z_breastmnist_beee8e35 | speckle_noise | complet, git 9abab3b propre ✅ |
| 20260901T170336Z_breastmnist_19a29d7c | motion_blur | complet, git 9abab3b propre ✅ |
| 20260901T170957Z_breastmnist_1dfe8046 | brightness_down | complet, git 9abab3b propre ✅ |

## 2. Détection du décalage par couche — AUROC(F) et AUROC(H), sévérité 5

| Couche | speckle F | speckle H | motion F | motion H | bright_down F | bright_down H |
|---|---|---|---|---|---|---|
| 1 | 0,043 | **1,000** | 0,114 | 0,341 | **0,997** | 0,979 |
| 2 | 0,966 | 0,999 | 0,145 | 0,631 | 0,914 | 0,928 |
| 3 | **1,000** | 0,969 | 0,023 | 0,392 | 0,461 | 0,961 |
| 4 | 1,000 | 0,753 | 0,087 | 0,898 | 0,967 | **0,989** |
| 5 | 1,000 | 0,094 | 0,675 | 0,426 | 0,840 | 0,942 |
| 6 | 1,000 | 0,325 | 0,742 | 0,092 | 0,897 | 0,686 |
| 7 | 1,000 | 0,016 | 0,236 | 0,166 | 0,751 | 0,813 |
| 8 | 0,972 | 0,628 | 0,426 | 0,808 | 0,516 | 0,801 |
| 9 | 0,895 | 0,478 | 0,612 | 0,148 | 0,334 | 0,158 |
| 10 | 0,987 | 0,243 | 0,957 | 0,162 | 0,495 | 0,228 |
| 11 | 0,976 | 0,530 | **0,982** | 0,481 | 0,469 | 0,476 |
| 12 (sortie) | 0,814 | 0,734 | 0,885 | 0,760 | 0,359 | 0,563 |

## 3. Lecture

1. **Le signal de décalage est nettement plus fort en profondeur intermédiaire qu'en
   sortie.** speckle : AUROC(F) = 1,000 aux couches 3–7 (contre 0,814 en couche 12) ;
   brightness_down : 0,997 en couche 1 — **l'inversion de signe de la couche 12 (0,359)
   n'existe pas dans les premières couches**. L'intuition du croquis fondateur (la
   profondeur est une variable qui compte) est confirmée empiriquement.
2. **Aucune profondeur n'est optimale partout** : meilleure couche pour F = 3 (speckle),
   11 (motion_blur), 1 (brightness_down). motion_blur est même *sous le hasard* aux
   couches 1–4. Un N fixe choisi au vu de ces résultats serait une fuite ; la suite
   naturelle est une règle calibrée sur la source, par échantillon.
3. **Pas de « cristallisation en profondeur » monotone sur la source** : H_n vaut
   0,475 → 0,459 → 0,430 → **0,323** (min, couche 4) → 0,466 → … → 0,517 (couche 12).
   L'attente qualitative consignée a priori (« H_n décroît avec n ») est **réfutée** sur
   ce substrat : le minimum d'entropie est atteint au tiers de la profondeur.
4. **Avertissement d'interprétation** (consigné dans chaque metrics.json) : les couches
   intermédiaires sont lues par la projection finale (« logit lens ») alors que seule la
   couche 12 a été alignée à l'espace texte par l'entraînement contrastif. Le pouvoir de
   détection observé est un fait empirique sur cette lecture, pas une propriété garantie
   des représentations intermédiaires.

## 4. Décision

- La profondeur N entre officiellement dans le tableau de bord expérimental
  (figures `01_trajectoires_couches.png`, `02_detection_par_couche.png` par run).
- Prochaine étape (chantier 3) : règle d'arrêt par échantillon calibrée sur la source
  — comparer arrêt sur H (idée initiale) vs arrêt sur F vs profondeur fixe, en
  simulation depuis les `scores_layers.npz` existants, sans nouveau calcul.
- À élargir ensuite : PneumoniaMNIST (réplication seconde modalité) et corruptions
  restantes, si le motif se confirme.
