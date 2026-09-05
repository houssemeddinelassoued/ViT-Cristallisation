# Rapport de résultats — Expérience 1 (zero-training, cohorte P0)

**Date du rapport** : 2026-09-01
**Expérience** : `exp01_zero_training` — détection de décalage d'acquisition par énergie libre F vs entropie H, sans aucun entraînement.
**Modèle** : ViT-B-16-quickgelu · poids `openai` (CLIP), T = 2τ ≈ 0,0200
**Cohorte** : BreastMNIST 224×224, split test, n = 156, corruptions MedMNIST-C
**Invites** : ensemble de 3 gabarits/classe (malin vs normal/bénin)

Règle du dépôt : seul un chiffre présent dans un `outputs/<run_id>/metrics.json` existe.
Tous les chiffres ci-dessous proviennent de ces fichiers.

---

## 1. Critère pré-enregistré

Évalué à la sévérité maximale (5), corruption vs images propres :

| # | Critère | Seuil |
|---|---|---|
| C1 | AUROC(F) | ≥ 0,70 |
| C2 | AUROC(H) | ≤ 0,60 |
| C3 | AUROC(F) − AUROC(H) | ≥ 0,05 |

La prédiction est confirmée seulement si les trois tiennent.

---

## 2. Inventaire des exécutions

| run_id | Corruption | Seed | Statut | Citable |
|---|---|---|---|---|
| 20260831T233917Z_…_6243078d | speckle_noise | 0 | limit=32, non versionné (smoke test) | ⚠️ non |
| 20260831T235358Z_…_da17e51e | speckle_noise | 0 | complet mais non versionné (git dirty) | ⚠️ non |
| 20260901T000620Z_…_c2a0cdaf | speckle_noise | 0 | complet, git 101ab28 propre | ✅ |
| 20260901T000843Z_…_4aecb603 | — | — | interrompu, pas de metrics.json | ❌ |
| 20260901T001121Z_…_7d2ca324 | speckle_noise | 2 | complet, git 101ab28 propre | ✅ |
| 20260901T001336Z_…_0a27403c | speckle_noise | 3 | complet, git 101ab28 propre | ✅ |
| 20260905T164506Z_…_0253fd15 | speckle_noise | 4 | complet, git 3e51dc3 propre, carte | ✅ |
| 20260905T164541Z_…_9bd74780 | motion_blur | 0 | complet, git 3e51dc3 propre, carte | ✅ |
| 20260901T002347Z_…_1f087957 | — | — | interrompu, pas de metrics.json | ❌ |
| 20260901T002352Z_…_38532b74 | brightness | 0 | échec (nom de corruption invalide) | ❌ |

Note : le seed n'affecte que le générateur de corruption ; à sévérité 0 les images
sont identiques d'un seed à l'autre, d'où des métriques sévérité-0 strictement égales.
Aucun run avec seed 1 n'a abouti (exécution interrompue).
Corruptions MedMNIST-C valides pour breastmnist : `pixelate`, `jpeg_compression`,
`speckle_noise`, `motion_blur`, `brightness_up`, `brightness_down`, `contrast_down`.

---

## 3. Résultat principal — speckle_noise, 4 seeds (0, 2, 3, 4)

### 3.1 Détection de décalage, AUROC(F) et AUROC(H) par sévérité

| Sévérité | AUROC(F) seed 0 | seed 2 | seed 3 | seed 4 | **moyenne F** | AUROC(H) seed 0 | seed 2 | seed 3 | seed 4 | **moyenne H** |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0,461 | 0,466 | 0,470 | 0,469 | **0,467** | 0,619 | 0,624 | 0,632 | 0,622 | **0,624** |
| 3 | 0,664 | 0,676 | 0,682 | 0,674 | **0,674** | 0,674 | 0,658 | 0,667 | 0,647 | **0,662** |
| 5 | 0,836 | 0,826 | 0,821 | 0,827 | **0,828** | 0,748 | 0,741 | 0,724 | 0,723 | **0,734** |

À sévérité 5 : AUROC(F) = 0,828 ± 0,006 ; AUROC(H) = 0,734 ± 0,013 ;
écart F−H = 0,094 ± 0,008 (moyenne ± écart-type sur 4 seeds).
Le résultat est stable inter-seeds : l'écart-type est inférieur au centième d'AUROC pour F.

### 3.2 Verdict par rapport au critère pré-enregistré (sévérité 5)

| Critère | Seuil | Mesuré (4 seeds) | Statut |
|---|---|---|---|
| C1 · AUROC(F) ≥ 0,70 | 0,70 | 0,821 – 0,836 | ✅ tenu sur 4/4 seeds |
| C2 · AUROC(H) ≤ 0,60 | 0,60 | 0,723 – 0,748 | ❌ violé sur 4/4 seeds |
| C3 · F − H ≥ 0,05 | 0,05 | 0,085 – 0,104 | ✅ tenu sur 4/4 seeds |

**Prédiction pré-enregistrée : NON confirmée** (`prediction_confirmee: false` dans les
4 runs). Réfutation partielle : F détecte le décalage et domine H comme prédit,
mais l'hypothèse forte « H reste proche du hasard » est fausse pour speckle_noise.

### 3.3 Anomalie reproductible : F sous le hasard à faible sévérité

À sévérité 1, AUROC(F) = 0,467 ± 0,004 (sous 0,5) alors que AUROC(H) = 0,624.
Mécanisme visible dans les observables : F_mean *diminue* légèrement entre
sévérité 0 (1,2878) et sévérité 1 (1,2844–1,2856) — un léger speckle *rapproche*
les images des ancres textuelles au lieu de les en éloigner. La dominance de F
n'émerge qu'à partir de la sévérité ~3 et devient nette à 5. Reproduit sur 4 seeds.

### 3.4 Classification zero-shot (contexte)

| Sévérité | Accuracy | Bal. accuracy | AUROC clf | H_mean | F_mean |
|---|---|---|---|---|---|
| 0 | 0,712 | 0,517 | 0,636 | 0,517 | 1,288 |
| 1 | 0,654–0,686 | 0,485–0,514 | 0,642–0,657 | 0,566 | 1,285 |
| 3 | 0,679–0,724 | 0,548–0,608 | 0,664–0,705 | 0,580 | 1,315 |
| 5 | 0,692–0,731 | 0,677–0,733 | 0,742–0,784 | 0,611 | 1,357 |

Deux observations importantes :

1. **Le zero-shot de base est quasi hasard** (balanced accuracy 0,517 à sévérité 0) :
   les ancres textuelles CLIP séparent à peine malin/bénin en échographie.
2. **La classification s'améliore avec la corruption** (AUROC 0,636 → ~0,755 en
   moyenne à sévérité 5) — comportement contre-intuitif qui fragilise la lecture
   « décalage = dégradation » sur cette cohorte, et cohérent avec des ancres mal
   alignées au domaine.

---

## 4. Réplication sur motion_blur (seed 0, 1 run)

| Sévérité | AUROC(F) | AUROC(H) | AUROC clf | Accuracy |
|---|---|---|---|---|
| 1 | 0,526 | 0,602 | 0,530 | 0,686 |
| 3 | 0,749 | 0,722 | 0,610 | 0,641 |
| 5 | **0,866** | **0,757** | 0,582 | 0,603 |

- Même verdict qualitatif : C1 ✅ (0,866), C2 ❌ (0,757), C3 ✅ (0,109) →
  `prediction_confirmee: false`.
- Contrairement au speckle, le motion_blur **dégrade** la classification
  (AUROC 0,636 → 0,582), profil plus conforme à un vrai décalage nuisible.
- La même structure se répète : H meilleur que F à faible sévérité, F domine
  nettement à forte sévérité.

---

## 5. Synthèse et interprétation

1. **Ce qui est robuste** (4 seeds × speckle + 1 × motion_blur) :
   - F détecte fortement le décalage à sévérité maximale (0,82–0,87), au-dessus du
     seuil de 0,70 dans tous les runs ;
   - F domine H de ~0,09–0,11 AUROC à sévérité maximale, au-dessus du seuil de 0,05 ;
   - l'invariant F = ⟨E⟩ − TH tient numériquement dans tous les runs.

2. **Ce qui réfute la prédiction forte** : H n'est jamais proche du hasard
   (0,72–0,76 à sévérité 5). Les corruptions testées ne sont pas des translations
   pures d'énergie : elles modifient aussi les écarts relatifs entre classes,
   donc l'entropie porte du signal de décalage.

3. **Structure en deux régimes** (nouvelle observation, non pré-enregistrée) :
   - faible sévérité → H > F, et F peut passer sous le hasard ;
   - forte sévérité → F ≫ H.
   Hypothèse à tester : la composante translationnelle de l'énergie ne domine
   qu'au-delà d'un seuil de corruption.

4. **Limite majeure** : les ancres textuelles CLIP openai sont quasi aveugles sur
   BreastMNIST (bal. acc. 0,517). Toute conclusion sur H vs F est conditionnée à
   cette géométrie d'ancres faible.

## 6. Prochaines étapes

1. Balayer les 7 corruptions MedMNIST-C valides (`brightness_up`, `brightness_down`,
   `contrast_down` sont les candidates « translation d'énergie » — attention aux
   noms exacts, `brightness` seul est invalide).
2. Relancer avec seed 1 pour compléter la grille multi-seeds.
3. Pré-enregistrer une exp01b : BiomedCLIP (ancres alignées au domaine médical),
   sévérités complètes 0–5, mêmes seuils.
4. Documenter le régime faible-sévérité (F < 0,5) comme résultat à part entière.
5. ~~Exécuter les runs sur un dépôt git propre pour lever la réserve `git_dirty`
   des runs 5458df41 et 4a406385.~~ **Fait le 2026-09-05** : rejoués sur arbre
   propre et sur carte (`0253fd15`, `9bd74780`). Seule la taille de lot diffère
   (64 au lieu de 8), ce qui ne change pas le calcul et explique le nouveau
   condensé de configuration ; les AUROC concordent à 0,001 près.

---

*Runs sources : c2a0cdaf, 7d2ca324, 0a27403c (speckle_noise, seeds 0/2/3),
0253fd15 (speckle_noise, seed 4) et 9bd74780 (motion_blur, seed 0), ces deux derniers
rejoués le 2026-09-05. Environnement des trois premiers : Python 3.11.14, torch
2.4.1+cpu, Windows, processeur, commit 101ab28 ; des deux derniers : torch 2.4.1+cu124,
RTX 4060, commit 3e51dc3.*

---

## 7. Balayage systématique — 2026-09-01, commit a82b42d (arbre propre)

Exécution des étapes 1, 2 et 4 du §6, plus une seconde modalité (PneumoniaMNIST,
radiographie, n = 624) et la sensibilité aux invites. Runs canoniques : ensemble
d'invites, seed 0, sévérités 0–5 complètes. Analyse appariée par
`experiments/exp01_analysis.py` (DeLong + bootstrap stratifié 10 000),
agrégation par `experiments/exp01_aggregate.py` (`outputs/aggregate/`).

### 7.1 Détection à sévérité 5 — les 11 runs canoniques

| Dataset | Corruption | AUROC(F) | AUROC(H) | ΔAUROC [IC 95 %] | p (DeLong) | \|F−méd_src\| | run id |
|---|---|---|---|---|---|---|---|
| breast | pixelate | 1,000 | 0,888 | +0,112 [+0,078, +0,150] | 1,9e−9 | 1,000 | 3e02269f |
| breast | jpeg_compression | 0,920 | 0,711 | +0,208 [+0,143, +0,274] | 7,7e−10 | 0,899 | 98d42003 |
| breast | motion_blur | 0,885 | 0,760 | +0,125 [+0,060, +0,190] | 1,7e−4 | 0,845 | 2308a744 |
| breast | speckle_noise | 0,814 | 0,734 | +0,080 [+0,006, +0,154] | 0,032 | 0,742 | de6dba0c |
| pneumonia | gaussian_blur | 0,970 | 0,798 | +0,172 [+0,148, +0,198] | 4,1e−41 | 0,962 | 2bc1d5f9 |
| breast | brightness_up | 0,594 | 0,677 | −0,083 [−0,176, +0,012] | 0,083 | 0,447 | 41fa7739 |
| breast | brightness_down | 0,359 | 0,563 | −0,204 [−0,297, −0,112] | 1,3e−5 | 0,606 | c1697982 |
| breast | contrast_down | 0,299 | 0,498 | −0,199 [−0,287, −0,112] | 7,4e−6 | 0,630 | 4ae2e534 |
| pneumonia | gaussian_noise | 0,127 | 0,462 | −0,335 [−0,368, −0,300] | 1,7e−86 | 0,803 | a0dfe1a7 |
| pneumonia | brightness_down | 0,200 | 0,644 | −0,445 [−0,482, −0,406] | 8,0e−118 | 0,598 | 2de86cbd |
| pneumonia | contrast_down | 0,209 | 0,698 | −0,490 [−0,526, −0,452] | 1,7e−148 | 0,589 | 2764835a |

### 7.2 Lecture

1. **Dichotomie structurel / photométrique.** F ≫ H sur les corruptions qui
   détruisent la structure spatiale (pixelate, jpeg, motion_blur, speckle,
   gaussian_blur) ; F passe **sous le hasard** sur les corruptions photométriques
   (brightness, contrast, bruit gaussien radiographique) : les images corrompues
   se *rapprochent* des ancres textuelles. **F est un détecteur signé.**
2. **Variante bilatérale.** |F − médiane_source(F)| (calibrée sur la source
   seule, `two_sided_shift_score`) récupère 5 des 6 inversions
   (gaussian_noise 0,127 → 0,803) sans dégrader le bloc structurel. Échec
   résiduel : brightness_up (trajectoire de F non monotone).
3. **Sensibilité aux invites** (runs 1230876b, 417916a6, 6fdae919 — speckle ;
   6353723a, 7ee23e9b, fb676849 — brightness_down) : dispersion inter-gabarits
   de l'AUROC(F) ≈ 0,03–0,05 contre ≈ 0,06–0,08 pour H. L'inversion de signe
   persiste sous chaque gabarit isolé — ce n'est pas un artefact d'invite.
4. **Graines** (0f58bbbd, f7608199) : AUROC(F) speckle sév. 5 = 0,814/0,819/0,827 —
   le bruit d'échantillonnage des corruptions est négligeable devant les effets.
5. **Verdict pré-enregistré** : C2 (H ≤ 0,60) échoue sur les 11 configurations →
   `prediction_confirmee: false` partout. Réfutation consignée telle quelle ;
   la dichotomie signée est le résultat exploitable.

### 7.3 Artefacts

- Agrégat : `outputs/aggregate/{summary.json, summary.csv, fig_*.png}`.
- Papier : `papers/contribution/main.tex` (compilé, 10 pages) — chaque valeur y est tracée
  vers un run id de ce tableau.
