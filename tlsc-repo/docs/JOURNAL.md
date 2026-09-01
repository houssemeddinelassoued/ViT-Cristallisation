# Journal de bord

Une entrée par exécution. **Un chiffre absent d'un `outputs/<run_id>/metrics.json`
n'existe pas** — c'est la règle qui rend le protocole opposable plutôt que déclaratif.

Format : date · run_id · ce qui était testé · ce qui a été observé · décision.

---

## 2026-08-31 · socle initial

- Dépôt créé, noyau formel écrit, 10 tests verts, lint propre.
- Chaîne d'expérience validée de bout en bout en mode `--dry-run`.
- **Aucune mesure réelle.** Les runs `dry_run: true` présents dans `outputs/` sont
  des validations de tuyauterie sur embeddings synthétiques ; ils portent un champ
  `avertissement` explicite et ne doivent jamais être cités.
- Décision : passer à l'expérience réelle dès que `open_clip` et `medmnist` sont
  installés sur la machine cible.

## 2026-09-01 · première exécution réelle

| Champ | Valeur |
|---|---|
| Date | 2026-09-01 |
| run_id | 20260901T000620Z_breastmnist_c2a0cdaf |
| Modèle · poids | ViT-B-16-quickgelu · openai |
| Cohorte · corruption | breastmnist · medmnistc:speckle_noise (sév. 0, 1, 3, 5) |
| AUROC(F) à sévérité max | 0,836 |
| AUROC(H) à sévérité max | 0,748 |
| Prédiction confirmée ? | **Non** — F ≥ 0,70 ✓ et F−H = 0,088 ≥ 0,05 ✓, mais H = 0,748 > 0,60 ✗ |
| Décision | Réfutation partielle : F domine H mais H n'est pas au hasard. Observations annexes : à sévérité 1, AUROC(F) = 0,461 (sous le hasard) contre AUROC(H) = 0,619 ; la classification s'améliore avec la corruption (AUROC 0,636 → 0,784), signe d'un zero-shot de base quasi hasard (bal. acc. 0,517 à sév. 0). Prochaines étapes : tester d'autres corruptions et cohortes, renforcer les invites, envisager un modèle plus adapté au domaine médical. |

- Le run `20260901T000843Z_breastmnist_4aecb603` (figures sans `metrics.json`) est
  incomplet et ne doit pas être cité.

## 2026-09-01 · balayage systématique P0 (commit a82b42d, arbre propre)

19 runs canoniques et de contrôle, tous complets, sévérités 0–5, ensemble
d'invites sauf mention. Détail chiffré : `docs/rapport-exp01.md` §7 et
`outputs/aggregate/summary.csv`.

| Lot | Runs | Ce qui était testé | Observé |
|---|---|---|---|
| 7 corruptions breastmnist | de6dba0c, 2308a744, 41fa7739, c1697982, 4ae2e534, 98d42003, 3e02269f | prédiction pré-enregistrée corruption par corruption | F ≫ H sur structurel (pixelate 1,000 ; jpeg 0,920) ; **F sous le hasard sur photométrique** (brightness_down 0,359 ; contrast_down 0,299) |
| Sensibilité invites | 1230876b, 417916a6, 6fdae919, 6353723a, 7ee23e9b, fb676849 | chaque gabarit isolé, speckle + brightness_down | dispersion AUROC(F) ≈ 0,03–0,05 < dispersion AUROC(H) ≈ 0,06–0,08 ; l'inversion de signe persiste sous chaque gabarit |
| Graines 1–2, speckle | 0f58bbbd, f7608199 | bruit d'échantillonnage des corruptions | AUROC(F) sév. 5 : 0,814/0,819/0,827 — négligeable |
| PneumoniaMNIST ×4 | a0dfe1a7, 2bc1d5f9, 2de86cbd, 2764835a | réplication seconde modalité (n = 624) | dichotomie répliquée : gaussian_blur F = 0,970 ≫ H ; photométrique inversé (F = 0,127–0,209, p ≤ 1e−86) |

- **Verdict pré-enregistré : réfuté sur les 11 configurations canoniques**
  (C2 échoue partout ; `prediction_confirmee: false`). Consigné tel quel.
- **Résultat exploitable non pré-enregistré** : F est un détecteur *signé* ;
  la variante bilatérale |F − médiane_source| (ajoutée à `tlsc/eval/metrics.py`,
  testée) récupère 5 inversions sur 6, recalculée dans `outputs/aggregate/`.
- Analyse appariée (DeLong + bootstrap 10 000) écrite dans chaque
  `outputs/<run_id>/analysis.json` ; run fumigène 6243078d lu en cp1252 (antérieur
  au passage UTF-8), gestion de repli ajoutée aux scripts d'analyse.
- Papier rédigé et compilé : `papers/contribution/main.tex` (10 pages, pdflatex/MiKTeX propre).
- Décision : prochaine étape scientifique = score bivarié (F, H) et cohorte P1
  (BUSI → UDIAT → BUS-UCLM) ; contrôle BiomedCLIP sur la configuration finale.

## 2026-09-01 · score bivarié (F, H) — Mahalanobis au nuage source

- Ajout de `bivariate_shift_score` à `tlsc/eval/metrics.py` (distance de
  Mahalanobis au nuage source (F, H), calibration source seule, régularisation
  ridge pour la quasi-colinéarité F = ⟨E⟩ − T·H). 4 tests ajoutés, suite verte
  (41 tests). Recalcul intégré à `experiments/exp01_aggregate.py`
  (`auroc_FH_bivariate` dans summary.json/csv, 3e barre sur `fig_two_sided.png`).
- AUROC à sévérité 5 sur les 11 configurations canoniques, bivarié vs bilatéral :
  meilleur sur 5 (speckle 0,774 vs 0,742 ; pneumonia gaussian_noise 0,837 vs
  0,803 ; pneumonia contrast_down 0,645 vs 0,584 ; pneumonia brightness_down
  0,617 vs 0,593), équivalent sur 3 (pixelate, jpeg, motion_blur), moins bon
  sur 3 (breast contrast_down 0,580 vs 0,630 ; breast brightness_down 0,578 vs
  0,606 ; gaussian_blur 0,943 vs 0,962).
- **brightness_up reste le cas d'échec** (0,473, sous le hasard même en bivarié) :
  le couple (F, H) cible se rapproche du centre source — aucune statistique de
  distance calibrée source ne peut le voir ; il faudra une statistique de forme
  (densité, non pas distance).
- Lecture : gain incrémental, pas transformateur. Le bivarié domine surtout là où
  H porte un signal complémentaire (photométrique pneumonia) ; il reste sous le
  bilatéral univarié F sur les photométriques breast où H est peu informatif.
- Décision : conserver le bivarié comme variante rapportée, pas comme détecteur
  principal ; passer à la cohorte P1.
