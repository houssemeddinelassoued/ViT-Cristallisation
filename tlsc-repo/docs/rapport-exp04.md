# Rapport de résultats — Expérience 4 (arrêt anticipé calibré, cohorte P0)

**Date du rapport** : 2026-09-01
**Expérience** : `exp04_early_exit` — règle d'arrêt par échantillon issue du croquis
fondateur (« sortir quand H < ε ») : sortie à la première couche où l'observable
(H ou F) passe sous un seuil calibré, comparée à la couche fixe et à la pleine
profondeur (`tlsc/eval/early_exit.py`).
**Modèle** : ViT-B-16-quickgelu · poids `openai` (CLIP gelé), ancres textuelles R1,
T = 2τ ≈ 0,0200
**Cohorte** : BreastMNIST 224×224, test n = 156 — split stratifié : 77 calibration /
79 évaluation (seed 0). Corruptions MedMNIST-C, sévérités 0–5.
**Statut épistémique** : **exploratoire déclaré** — pas de seuil de verdict
pré-enregistré.
**Anti-fuite** : seuils ε et couche fixe choisis sur la moitié CALIBRATION de la
source ; toutes les mesures rapportées portent sur la moitié ÉVALUATION.

> **Avertissement ajouté le 2026-09-05 — lire avant les chiffres d'efficience.**
> Ce rapport calibre sur l'exactitude *brute*. L'exp05 a établi que cette exactitude vaut
> 0,731 à chacune des douze couches sous ancres textuelles, soit exactement la proportion
> de la classe majoritaire de BreastMNIST : le critère est maximisé par le prédicteur
> dégénéré, et l'« exactitude préservée » de la section 4 est la préservation de ce taux,
> non d'un pouvoir diagnostique. En exactitude **équilibrée**, ces ancres valent 0,500,
> le hasard. Les mesures ci-dessous sont conservées comme trace du chantier ; les
> conclusions d'efficience sont reprises, sur critère équilibré, dans
> `rapport-exp05.md`.

---

## 1. Runs (git 3ef3ebe propre)

| run_id | Corruption |
|---|---|
| 20260901T182032Z_breastmnist_a8b68f84 | speckle_noise |
| 20260901T182436Z_breastmnist_a1a557b5 | motion_blur |
| 20260901T183051Z_breastmnist_898385f7 | brightness_down |

## 2. Calibration (identique dans les trois runs — même source)

| Règle | ε | Profondeur moyenne (calib) | Accuracy (calib) | Accuracy pleine profondeur |
|---|---|---|---|---|
| H-stop | 0,4991 | 1,00 | 0,731 | 0,705 |
| F-stop | 1,7261 | 1,03 | 0,731 | 0,705 |
| Couche fixe | — | 1 | 0,731 | 0,705 |

## 3. Évaluation sous décalage (split évaluation, n = 79)

Exactitude : les trois stratégies calibrées restent à 0,731 à toutes les sévérités
(elles sortent presque toujours en couche 1) ; la pleine profondeur varie de 0,577
(motion_blur sév. 5) à 0,833 (speckle sév. 4).

Profondeur moyenne de sortie N̄* :

| Sévérité | speckle H-stop | speckle F-stop | motion H/F-stop | bright_down H-stop | bright_down F-stop |
|---|---|---|---|---|---|
| 0 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| 1 | 1,00 | 1,08 | 1,00 | 1,00 | 1,59 |
| 3 | 2,03 | 1,03 | 1,00 | 1,00 | 1,97 |
| 5 | 2,08 | 1,00 | 1,00 | 1,00 | 2,79 |

## 4. Lecture — honnête

1. **L'économie de calcul est réelle mais triviale sur ce substrat.** La règle
   calibrée sort en couche ~1 avec exactitude préservée (0,731 ≥ 0,705), soit ~92 %
   de blocs économisés — mais uniquement parce que les ancres textuelles R1 sont
   quasi aveugles (bal. acc. 0,517, rapport exp01) : la profondeur n'apporte aucun
   gain diagnostique à préserver. Sur ce couple substrat/tâche, « argmin_N H » est
   dégénéré : N* = 1. La démonstration devra être refaite sous ancres R2/R3
   (exp03), où la classification porte un signal réel.
2. **Résultat exploitable : N̄* est lui-même un signal de décalage.** La profondeur
   de sortie croît avec la sévérité (F-stop brightness_down : 1,00 → 2,79 ;
   H-stop speckle : 1,00 → 2,08) : les images corrompues « retardent » leur
   cristallisation. Le compteur de couches, disponible gratuitement en production,
   devient un moniteur de dérive — à quantifier en AUROC au chantier suivant.
3. **Cohérence avec exp02** : la couche 1 concentre déjà l'essentiel de
   l'exactitude R1 (0,731), conformément aux trajectoires plates de `rapport-exp02.md`.

## 5. Décision

- Croiser exp03 × exp04 : rejouer l'arrêt calibré sous ancres R2 (few-shot k=16),
  où l'exactitude varie réellement avec la profondeur — c'est le test non trivial
  de la règle « H < ε ».
- Quantifier N̄* comme détecteur de décalage (AUROC des profondeurs de sortie
  source vs cible).
- Étendre à PneumoniaMNIST une fois le protocole croisé stabilisé.
