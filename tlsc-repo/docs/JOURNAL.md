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
| PneumoniaMNIST ×4 | a0dfe1a7, puis 2bc1d5f9, 2de86cbd, 2764835a rejoués le 2026-09-05 | réplication seconde modalité (n = 624) | dichotomie répliquée : gaussian_blur F = 0,970 ≫ H ; photométrique inversé (F = 0,127–0,209, p ≤ 1e−86) |

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
  0,803 ; pneumonia contrast_down 0,646 vs 0,589 ; pneumonia brightness_down
  0,618 vs 0,598), équivalent sur 3 (pixelate, jpeg, motion_blur), moins bon
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
déclaré.** Runs rejoués le 2026-09-05 sur arbre propre et sur carte
(git 3e51dc3) : `20260905T164120Z_…_fe6d6b35` (speckle_noise),
`20260905T163735Z_…_bccf7433` (brightness_down). Les exécutions du 2026-09-01
étaient annoncées ici « git 00f9f37 propre » ; c'était faux pour bccf7433, qui
portait `git_dirty: true` — corrigé et rejoué (voir 2026-09-05, assainissement).

| Régime | D_inter | Bal.acc (sév.0) | speckle F sév.5 | bright_down F sév.5 |
|---|---|---|---|---|
| R1 texte | 0,1032 | 0,517 | 0,815 | 0,360 |
| R2 few-shot k=16 | 0,0208 | 0,597 | 0,977 | 0,845 |
| R3 centroïdes | 0,0073 | 0,687 | 0,977 | 0,843 |

- **Observé** : l'inversion photométrique de F était une propriété des ancres
  *textuelles* — elle disparaît dès k = 16 (0,360 → 0,845). La faiblesse
  zero-shot venait des ancres (bal.acc 0,517 → 0,687 sous R3). D_inter seule
  est trompeuse : maximale pour les ancres les moins utiles (gap de modalité).
  H se dégrade en détecteur sous R2/R3 (0,25–0,56) — la dominance de F
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

## 2026-09-05 · exp05 — croisement ancres × arrêt calibré, N* détecteur, qualité d'ancrage

> **Entrée à une seule graine, corrigée le jour même par la consolidation à cinq graines
> (entrée suivante).** Les chiffres ci-dessous restent tels qu'ils ont été mesurés, mais
> plusieurs ne sont pas reproductibles : la couche 9, le gain de +0,241 et l'AUROC de
> 1,000 étaient des tirages particuliers. Se référer à l'entrée de consolidation.

Chantier décidé en fin d'exp03 et d'exp04 : rejouer l'arrêt calibré sous les trois
régimes d'ancrage, où l'exactitude varie réellement avec la profondeur
(`experiments/exp05_anchored_early_exit.py`, centroïdes recalculés couche par couche ;
`exit_depth_auroc` et `anchor_quality` ajoutés, 10 tests). **Exploratoire déclaré.**
Runs (git 973b931, arbre propre) : 7d1b0ed1 et 6f91783e (BreastMNIST, speckle_noise et
brightness_down), 9b5f183e et 6044eb14 (PneumoniaMNIST, gaussian_blur et
brightness_down).

- **Correction de méthode, préalable à toute lecture.** Le premier run a montré que
  l'exactitude *brute* vaut 0,731 aux douze couches sous ancres R1 — exactement la
  proportion de la classe majoritaire de BreastMNIST. Le critère de calibration de
  l'exp04 était donc maximisé par le prédicteur dégénéré, et validait une sortie en
  couche 1 sans contenu diagnostique. L'exp05 calibre sur l'exactitude **équilibrée**
  (`calibrate_epsilon(..., y=...)`, comportement par défaut inchangé pour l'exp04).
  Les profondeurs des deux expériences ne sont pas comparables.
- **Q1 — la dégénérescence N* = 1 est levée, et elle avait deux causes.** Sous R1
  l'équilibrée vaut 0,500 à onze couches sur douze : les ancres textuelles sont au
  hasard, pas seulement faibles. Sous ancres de données un optimum intérieur apparaît,
  jamais en couche 1 ni en couche 12 : couches 2 et 3 sur BreastMNIST (0,719 et 0,684),
  couches 9 et 6 sur PneumoniaMNIST (0,841 et 0,825).
- **Résultat le plus exploitable, non visé.** Les dernières couches dégradent le
  diagnostic sous décalage : PneumoniaMNIST gaussian_blur sous R2, la couche fixe 9
  tient 0,780 à sévérité 5 quand la pleine profondeur tombe à 0,539, pour 75 % du calcul.
  L'écart croît avec la sévérité (+0,136 à sév. 0, +0,241 à sév. 5).
- **Mais la règle par échantillon du croquis fondateur perd contre la couche fixe.**
  H-stop et F-stop sortent en couche ≈ 1,1 et plafonnent à 0,622 et 0,652 à sévérité 5,
  loin des 0,780 de la couche fixe calibrée. Consigné tel quel : ce qui survit du
  croquis est « toutes les couches ne se valent pas », pas « chaque image choisit la
  sienne ».
- **Q2 — N* est un détecteur réel mais ni universel ni de signe constant.** AUROC 1,000
  (règle F, R2) sur les deux corruptions BreastMNIST, 0,835 et 0,825 sur PneumoniaMNIST
  brightness_down, mais 0,503 sur gaussian_blur. Sous R3 sur BreastMNIST la règle H donne
  0,059 et 0,167 : détecteur fortement **inversé**, les images corrompues sortant plus
  tôt — même signature signée que F à l'exp01.
- **Q3 — critère de qualité d'ancrage : succès partiel.** Q_gap et Q_fisher corrigent
  l'inversion due au gap de modalité (distance ancres-nuage ≈ 1,28 pour le texte contre
  ≈ 0,04–0,09 pour les données) et placent partout les ancres de données devant les
  ancres textuelles. Mais ils classent R2 devant R3, alors que l'exactitude mesurée place
  R3 devant : les centroïdes complets se rapprochent de la moyenne globale et écrasent
  D_inter. Critère utilisable pour *rejeter*, pas pour départager.
- Corrigé au passage : `exp01_aggregate` référençait `os` sans l'importer et plantait au
  lancement (bug présent sur `main`) ; agrégation rejouée, valeurs du journal reproduites
  à l'identique.
- Décision : appliquer la transformation bilatérale aux profondeurs de sortie ; traiter
  « couche fixe calibrée » et « règle par échantillon » comme deux lignes distinctes de la
  matrice d'ablation ; refaire le croisement à plusieurs graines avec traitement
  statistique apparié avant toute revendication. Priorité inchangée : cohorte P1.
- Détail : `docs/rapport-exp05.md`.

## 2026-09-05 · assainissement — sept runs rejoués sur arbre propre, filtre de citabilité

Audit complet du dépôt selon la grille de `.claude/agents/superviseur-tlsc.md`.
Constat bloquant : sur 45 runs, 14 n'étaient pas citables, et plusieurs l'étaient
pourtant — dont un sur le **site public**. Corrigé.

- **Cause racine, dans l'outil.** `exp01_aggregate` ne contrôlait que la complétude
  et le caractère réel d'un run, jamais la propreté de l'arbre : les runs `git_dirty`
  entraient donc dans `summary.csv`. Ajout de `raison_rejet()` appliquant la règle
  complète, affichage nommé de tout run écarté, 8 tests dédiés
  (`tests/test_exp01_aggregate.py`). Correction d'un constat d'audit erroné de ma
  part : ces runs n'atteignaient pas le rang **canonique**, qui exige six sévérités.
- **Runs rejoués** (git 3e51dc3, arbre propre, carte RTX 4060) :
  `20260905T163407Z_…_2bc1d5f9`, `20260905T163525Z_…_2de86cbd`,
  `20260905T163629Z_…_2764835a` (exp01 PneumoniaMNIST, cités sans réserve dans le
  rapport) ; `20260905T163735Z_…_bccf7433` et `20260905T164120Z_…_fe6d6b35`
  (exp03, le journal les déclarait à tort « git 00f9f37 propre » alors que le second
  portait `git_dirty: true`) ; `20260905T164506Z_…_0253fd15` et
  `20260905T164541Z_…_9bd74780` (exp01, levant la réserve notée au rapport exp01 §6.5).
  Le run exp03 speckle était propre mais a dû suivre : les deux runs exp03 partagent
  une colonne d'exactitude à sévérité 0, laisser l'un sur processeur et l'autre sur
  carte l'aurait rendue fausse pour l'un des deux.
- **Écarts mesurés** entre processeur et carte : AUROC à 3,3e−4 près sur exp01. Sur
  exp03, l'exactitude équilibrée R3 à sévérité 0 passe de 0,692 à 0,687 — une seule
  prédiction basculée sur 156, près de la frontière de décision. Valeurs publiées
  ajustées d'une unité sur la troisième décimale : exp03 R1 speckle 0,814→0,815,
  R1 brightness 0,359→0,360, R3 brightness F 0,844→0,843 et H 0,539→0,540 ;
  exp01 pneumonia ΔAUROC −0,444→−0,445 et −0,489→−0,490, bilatéral 0,593→0,598 et
  0,584→0,589. Plage AUROC(H) sous R2/R3 corrigée de 0,25–0,54 à 0,25–0,56, qui
  était déjà imprécise avant le rejeu.
- **Propagation** : `docs/results.html`, `rapport-exp01.md`, `rapport-exp03.md`,
  `README.md`, ce journal, et `papers/contribution/main.tex` recompilé (zéro
  référence indéfinie ; le PDF ne contient plus aucune valeur périmée). Les trois
  figures de `outputs/aggregate/` sont resynchronisées vers le site et le papier —
  `fig_two_sided.png` du papier divergeait de sa source depuis le 2026-09-01.
- Restent non citables et **non cités** : deux runs interrompus sans `metrics.json`,
  un run fumigène, et quelques runs sales de mise au point antérieurs au 2026-09-01.
- Décision : consolidation multi-graines de l'exp05, puis porte de décision P1.

## 2026-09-05 · exp05 consolidée — cinq graines, et ce qui ne survit pas

20 runs = 5 graines × 4 configurations, tous propres (git a9b5f12, carte RTX 4060),
agrégés par `experiments/exp05_aggregate.py` (9 tests) vers
`outputs/aggregate/exp05_summary.{json,csv}`. Ajout au passage de `auroc_two_sided`
dans `exit_depth_auroc` (2 tests) : la profondeur de sortie hérite du caractère
signé de son observable, la transformation bilatérale lui est donc applicable.

- **Ce qui survit.** Les ancres textuelles sont au hasard (équilibrée 0,500–0,509 à
  toute couche, deux cohortes, cinq graines). Une couche intermédiaire bat la couche
  de sortie sous ancres de données dans les **quatre** configurations, de +0,045 à
  +0,134, écart supérieur à la dispersion inter-graines. La règle d'arrêt par
  échantillon reste battue par une profondeur fixe calibrée.
- **Ce qui ne survit pas, et que l'entrée précédente affirmait.** La couche optimale
  n'est pas localisée : écart-type de 2,7 à 3,4 couches sur une plage de 12 (seule
  exception, R3 sur PneumoniaMNIST à 8,4 ± 1,3). La « couche 9 » était un tirage.
- **Le gain sous décalage change de signe selon la corruption.** Flou gaussien :
  +0,134 ± 0,153 (R2) et +0,151 ± 0,111 (R3), cinq graines positives sous R3.
  Photométrique : −0,073 ± 0,069 et −0,038 ± 0,042, cinq graines négatives sous R3.
  Le +0,241 annoncé sur une graine vaut +0,134 ± 0,153. C'est la dichotomie
  structurel/photométrique de l'exp01, réapparue sur une grandeur toute autre.
- **N* détecteur : réel mais instable.** Règle F : 0,86–0,90 sur trois configurations,
  mais 0,900 ± 0,224 sur BreastMNIST, où la même configuration donne 0,500 ou 1,000
  selon la graine. Échec sur le flou (0,514–0,520), là même où le gain d'exactitude
  est maximal : détection et efficience ne vont pas de pair.
- **Bilatéral sur N*** : récupère partiellement les détecteurs inversés sur
  BreastMNIST (0,24–0,40 → 0,43–0,56, soit du franchement inversé au hasard), sans
  rien changer sur PneumoniaMNIST où les profondeurs sont quasi constantes. Gain
  réel, mais aucun détecteur fondé sur H et la profondeur n'est exploitable.
- **Q3** : Q_gap égalise R1 et R3 sur BreastMNIST (0,0799) alors que leurs exactitudes
  diffèrent de 0,15 ; sur PneumoniaMNIST R2 et R3 ne sont pas séparables
  (0,396 ± 0,102 contre 0,355 ± 0,009). Critère utile pour **rejeter**, pas pour
  départager.
- **Correction de cohérence** : l'exp04, calibrée sur l'exactitude brute, annonçait
  encore une économie de calcul « à exactitude préservée ». Avertissement ajouté au
  site et à `rapport-exp04.md` : cette préservation était celle du taux de la classe
  majoritaire.
- Décision : traitement statistique apparié avant toute revendication ; ne pas
  revendiquer de gain d'efficience sans distinguer corruptions structurelles et
  photométriques ; priorité à la porte de décision P1.
- Détail : `docs/rapport-exp05.md`.
