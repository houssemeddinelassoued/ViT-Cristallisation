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

## 2026-09-01 · exp02 — trajectoires par couche (profondeur N comme variable)

Chantier d'alignement sur le croquis fondateur de l'encadrant (`argmin_N H`,
`note-meeting.md`). Sonde « logit lens » : boucle explicite sur les 12 blocs du
ViT visuel, ln_post + proj sur le CLS à chaque profondeur
(`tlsc/models/layer_probe.py`, 4 tests sur ViT synthétique — la couche 12 égale
`encode_image` à la précision machine). **Exploratoire déclaré** : pas de seuil
pré-enregistré ; attente qualitative a priori « H_n décroît avec n sur la source ».

| run_id | Corruption | Meilleure couche (F) | AUROC(F) meilleure / couche 12 |
|---|---|---|---|
| 20260901T165926Z_breastmnist_beee8e35 | speckle_noise | 3 (1,000 aux couches 3–7) | 1,000 / 0,814 |
| 20260901T170336Z_breastmnist_19a29d7c | motion_blur | 11 | 0,982 / 0,885 |
| 20260901T170957Z_breastmnist_1dfe8046 | brightness_down | 1 | 0,997 / 0,359 |

- **Observé** : signal de décalage bien plus fort en profondeur intermédiaire
  qu'en sortie ; l'inversion photométrique de la couche 12 disparaît aux
  premières couches (brightness_down 0,997 en couche 1). Mais la meilleure
  couche dépend de la corruption (3 / 11 / 1) — pas de N universel.
- **Attente a priori réfutée** : H_n non monotone sur la source (min 0,323 en
  couche 4, remontée à 0,517 en couche 12). Pas de « cristallisation en
  profondeur » sur ce substrat. Consigné tel quel.
- Avertissement logit lens dans chaque metrics.json (couches intermédiaires
  jamais alignées à l'espace texte par l'entraînement).
- Décision : chantier suivant = règle d'arrêt par échantillon calibrée sur la
  source (H-stop vs F-stop vs N fixe), en simulation depuis les
  `scores_layers.npz` — aucun nouveau calcul d'encodage requis.
- Détail : `docs/rapport-exp02.md`.

## 2026-09-01 · exp03 — régimes d'ancrage R1/R2/R3 (centroïdes du croquis fondateur)

Chantier d'alignement 2 : ancres = centroïdes de classe calculés depuis les
données source (`tlsc/models/data_anchors.py`, 5 tests) + D_inter. Anti-fuite :
centroïdes sur le split train SOURCE à sévérité 0 uniquement. **Exploratoire
déclaré.** Runs (git 00f9f37 propre) : fe6d6b35 (speckle_noise), bccf7433
(brightness_down).

| Régime | D_inter | Bal.acc (sév.0) | speckle F sév.5 | bright_down F sév.5 |
|---|---|---|---|---|
| R1 texte | 0,1032 | 0,517 | 0,814 | 0,359 |
| R2 few-shot k=16 | 0,0208 | 0,597 | 0,977 | 0,845 |
| R3 centroïdes | 0,0073 | 0,692 | 0,977 | 0,844 |

- **Observé** : l'inversion photométrique de F était une propriété des ancres
  *textuelles* — elle disparaît dès k = 16 (0,359 → 0,845). La faiblesse
  zero-shot venait des ancres (bal.acc 0,517 → 0,692 sous R3). D_inter seule
  est trompeuse : maximale pour les ancres les moins utiles (gap de modalité).
  H se dégrade en détecteur sous R2/R3 (0,25–0,54) — la dominance de F
  s'accentue.
- Décision : R2 (32 étiquettes source) = meilleur compromis observé ; croiser
  ensuite meilleure couche (exp02) × ancres R2 ; formaliser un critère de
  qualité d'ancrage combinant D_inter et distance ancres–nuage.
- Détail : `docs/rapport-exp03.md`.

## 2026-09-01 · exp04 — arrêt anticipé calibré (« sortir quand H < ε »)

Chantier d'alignement 3 : règle d'arrêt par échantillon du croquis fondateur
(`tlsc/eval/early_exit.py`, 5 tests). Seuils ε (H et F) et couche fixe calibrés
sur une moitié stratifiée du split test source (77 images) ; évaluation sur
l'autre moitié (79), sévérités 0–5. **Exploratoire déclaré.** Runs
(git 3ef3ebe propre) : a8b68f84 (speckle), a1a557b5 (motion_blur),
898385f7 (brightness_down). Ancres R1.

- **Observé (1)** : la règle calibrée sort en couche ≈ 1 avec exactitude
  préservée (0,731 contre 0,705 pleine profondeur) — ~92 % de calcul économisé,
  mais *trivialement* : sous ancres R1 quasi aveugles, la profondeur n'apporte
  aucun gain diagnostique. Sur ce couple substrat/tâche, argmin_N H est
  dégénéré (N* = 1). Consigné tel quel.
- **Observé (2), exploitable** : la profondeur de sortie N̄* croît avec la
  sévérité (F-stop brightness_down 1,00 → 2,79 ; H-stop speckle 1,00 → 2,08) —
  l'arrêt anticipé est lui-même un moniteur de dérive, gratuit en production.
- Décision : test non trivial de la règle = croisement ancres R2 (exp03) ×
  arrêt calibré (exp04) ; quantifier N̄* en AUROC comme détecteur.
- Détail : `docs/rapport-exp04.md`.
