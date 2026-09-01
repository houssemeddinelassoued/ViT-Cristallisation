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
| 20260901T001636Z_…_5458df41 | speckle_noise | 4 | complet, git 101ab28 (dirty) | ✅ (réserve) |
| 20260901T002007Z_…_4a406385 | motion_blur | 0 | complet, git 101ab28 (dirty) | ✅ (réserve) |
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
5. Exécuter les runs sur un dépôt git propre pour lever la réserve `git_dirty`
   des runs 5458df41 et 4a406385.

---

*Runs sources : c2a0cdaf, 7d2ca324, 0a27403c, 5458df41 (speckle_noise, seeds 0/2/3/4),
4a406385 (motion_blur, seed 0). Environnement : Python 3.11.14, torch 2.4.1+cpu,
Windows, CPU. Commit 101ab28.*
