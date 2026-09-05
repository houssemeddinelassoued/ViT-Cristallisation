# Rapport de résultats — Expérience 5 (ancres × arrêt calibré, cohorte P0)

**Date du rapport** : 2026-09-05, **consolidé sur cinq graines le même jour**
**Expérience** : `exp05_anchored_early_exit` — croisement des chantiers exp03 (régimes
d'ancrage) et exp04 (arrêt anticipé). L'arrêt calibré est rejoué sous les trois régimes,
avec des centroïdes R2/R3 **recalculés couche par couche** sur le split train source
(`tlsc/models/data_anchors.py`, `tlsc/eval/early_exit.py`).
**Modèle** : ViT-B-16-quickgelu · poids `openai` (CLIP gelé), T = 2τ ≈ 0,0200, 12 blocs.
**Cohortes** : BreastMNIST 224×224 (test n = 156, split 78/78) et PneumoniaMNIST 224×224
(test n = 624, split 312/312). Corruptions MedMNIST-C, sévérités 0–5.
**Statut épistémique** : **exploratoire déclaré** — aucun seuil de verdict pré-enregistré.
**Anti-fuite, trois barrières** : (1) centroïdes R2/R3 sur le split TRAIN source à
sévérité 0 seulement ; (2) seuils ε et couche fixe sur la moitié CALIBRATION du split test
source ; (3) toutes les valeurs rapportées, source comme cible, sur la moitié ÉVALUATION.

> **Ce rapport corrige une version antérieure fondée sur une seule graine.** Les
> conclusions de tête y étaient nettement surestimées : un gain annoncé à +0,241 vaut
> +0,134 ± 0,153 sur cinq graines, une couche optimale annoncée à 9 vaut 4,8 ± 2,7, une
> AUROC annoncée à 1,000 vaut 0,900 ± 0,224. La section 7 dit ce qui survit et ce qui ne
> survit pas. C'est l'objet même d'une consolidation multi-graines.

> **Changement de critère par rapport à l'exp04.** L'exp04 calibrait sur l'exactitude
> *brute*. Le premier run de l'exp05 a montré que cette exactitude vaut 0,731 à chacune
> des douze couches sous ancres textuelles, soit exactement la proportion de la classe
> majoritaire de BreastMNIST : le critère était maximisé par le prédicteur dégénéré qui
> répond toujours la même classe. L'exp05 calibre sur l'exactitude **équilibrée**
> (`calibrate_epsilon(..., y=...)`, deux tests dédiés). Les profondeurs des deux
> expériences ne sont pas comparables, et les conclusions d'efficience de l'exp04 sont
> à relire à cette lumière.

---

## 1. Runs

20 runs = 5 graines (0 à 4) × 4 configurations, tous **complets, propres**
(`git_dirty: false`, commit a9b5f12) et exécutés sur carte RTX 4060.
Agrégation : `python -m experiments.exp05_aggregate outputs` →
`outputs/aggregate/exp05_summary.{json,csv}`.

| Cohorte | Corruption | n train source | n éval | durée d'un run |
|---|---|---|---|---|
| BreastMNIST | speckle_noise | 546 | 78 | 30 s |
| BreastMNIST | brightness_down | 546 | 78 | 22 s |
| PneumoniaMNIST | gaussian_blur | 1200 | 312 | 67 s |
| PneumoniaMNIST | brightness_down | 1200 | 312 | 58 s |

Les runs de graine 0 sont `20260905T172741Z_…_7d1b0ed1`, `20260905T172821Z_…_6f91783e`,
`20260905T172853Z_…_9b5f183e` et `20260905T173011Z_…_6044eb14` ; les graines 1 à 4
suivent immédiatement. Le split train de PneumoniaMNIST est sous-échantillonné de façon
stratifiée à 1200 images (`--train-limit`) : R3 y est un plafond approché.

Ce que la graine fait varier : le tirage few-shot des centroïdes R2, et la coupe
calibration/évaluation. Elle ne change ni les images ni les corruptions.

## 2. Q1 — la profondeur utile cesse-t-elle d'être N* = 1 ?

Exactitude **équilibrée**, domaine source, split évaluation, moyenne ± écart-type sur
cinq graines.

| Cohorte | Régime | Couche oracle | Équil. à l'oracle | Équil. couche 12 | Écart |
|---|---|---|---|---|---|
| BreastMNIST | R1 texte | 7,6 ± 6,0 | 0,509 ± 0,009 | — | — |
| BreastMNIST | R2 few-shot | 5,2 ± 3,1 | 0,748 ± 0,061 | 0,614 ± 0,065 | +0,134 |
| BreastMNIST | R3 centroïdes | 4,0 ± 3,4 | 0,703 ± 0,041 | 0,658 ± 0,035 | +0,045 |
| PneumoniaMNIST | R1 texte | 1,0 ± 0,0 | 0,500 ± 0,000 | — | — |
| PneumoniaMNIST | R2 few-shot | 4,8 ± 2,7 | 0,821 ± 0,016 | 0,728 ± 0,031 | +0,093 |
| PneumoniaMNIST | R3 centroïdes | 8,4 ± 1,3 | 0,835 ± 0,023 | 0,730 ± 0,014 | +0,105 |

1. **Les ancres textuelles sont au hasard, pas seulement faibles.** L'exactitude
   équilibrée vaut 0,500 à 0,509 quelle que soit la couche, sur les deux cohortes et
   les cinq graines. La « préservation d'exactitude » de l'exp04 était la préservation
   du taux de la classe majoritaire. La dégénérescence N* = 1 avait donc **deux causes
   cumulées** : des ancres aveugles et un critère de calibration que le prédicteur
   majoritaire maximise.
2. **Une couche intermédiaire bat la couche de sortie, et ce constat est robuste.**
   L'écart est positif dans les quatre configurations sous ancres de données, de +0,045
   à +0,134, et il dépasse l'écart-type inter-graines dans les quatre cas. Lues avec des
   ancres de données, les dernières couches coûtent du calcul *et* de la performance sur
   données propres.
3. **Mais *quelle* couche est optimale n'est pas stable.** L'écart-type de la couche
   oracle vaut 2,7 à 3,4 couches sur une plage de 12, sauf pour R3 sur PneumoniaMNIST
   (8,4 ± 1,3). La valeur « couche 9 » du rapport à une graine était un tirage
   particulier, pas une propriété du substrat.

> **Réserve de méthode sur l'oracle.** La couche oracle est choisie *sur le split
> d'évaluation lui-même*. C'est donc une borne supérieure optimiste, pas une performance
> atteignable : elle mesure « il existe une couche meilleure que la sortie », non « on
> sait la trouver ». La couche fixe calibrée de la section 3 est, elle, choisie sur la
> moitié calibration et évaluée sur l'autre — c'est la version honnête.

## 3. Q1 (suite) — comportement sous décalage

Écart d'exactitude équilibrée entre la **couche fixe calibrée** et la pleine profondeur,
à sévérité 5, moyenne ± écart-type sur cinq graines. Positif = la sortie anticipée est
meilleure.

| Cohorte · corruption | R2 few-shot | R3 centroïdes |
|---|---|---|
| Pneumonia · gaussian_blur (structurel) | **+0,134 ± 0,153** | **+0,151 ± 0,111** |
| Breast · speckle_noise (structurel) | +0,010 ± 0,104 | +0,029 ± 0,074 |
| Breast · brightness_down (photométrique) | −0,017 ± 0,042 | −0,035 ± 0,057 |
| Pneumonia · brightness_down (photométrique) | −0,073 ± 0,069 | **−0,038 ± 0,042** |

**Le signe de l'effet dépend du type de corruption.** Sur le flou gaussien, la couche
fixe calibrée l'emporte nettement, et les cinq graines sont positives sous R3 (minimum
+0,022). Sur les corruptions photométriques, elle *perd* : sous R3 sur PneumoniaMNIST,
les cinq graines sont négatives (maximum −0,013). Sur speckle l'écart est
indiscernable de zéro.

C'est la même dichotomie structurel/photométrique que l'exp01 avait trouvée pour
l'énergie libre comme détecteur. Elle réapparaît ici sur une grandeur toute autre, la
profondeur utile — piste à creuser, pas conclusion.

**La règle par échantillon du croquis fondateur reste battue par la couche fixe.**
Les règles calibrées H-stop et F-stop sortent en couche ≈ 1,1 et n'atteignent pas le
niveau de la couche fixe. Ce qui survit du croquis est « toutes les couches ne se valent
pas », pas « chaque image doit choisir la sienne ».

## 4. Q2 — la profondeur de sortie comme détecteur de décalage

AUROC des profondeurs N* source contre cible, sévérité 5, split évaluation, cinq graines
(`exit_depth_auroc`). La colonne bilatérale applique à N* la transformation
|N* − médiane_source(N*)| de `two_sided_shift_score`.

| Cohorte · corruption | Régime | AUROC (règle F) | AUROC (règle H) | H bilatérale |
|---|---|---|---|---|
| Breast · speckle_noise | R2 | 0,900 ± 0,224 | 0,397 ± 0,130 | 0,526 ± 0,168 |
| Breast · speckle_noise | R3 | 0,617 ± 0,160 | 0,267 ± 0,087 | 0,433 ± 0,148 |
| Breast · brightness_down | R2 | 0,900 ± 0,224 | 0,376 ± 0,191 | 0,560 ± 0,232 |
| Breast · brightness_down | R3 | 0,617 ± 0,160 | 0,240 ± 0,096 | 0,446 ± 0,192 |
| Pneumonia · brightness_down | R2 | 0,857 ± 0,081 | 0,493 ± 0,026 | 0,493 ± 0,026 |
| Pneumonia · brightness_down | R3 | 0,886 ± 0,080 | 0,500 ± 0,007 | 0,500 ± 0,007 |
| Pneumonia · gaussian_blur | R2 | 0,520 ± 0,025 | 0,474 ± 0,009 | 0,474 ± 0,009 |
| Pneumonia · gaussian_blur | R3 | 0,514 ± 0,020 | 0,482 ± 0,006 | 0,482 ± 0,006 |

1. **Le compteur de couches est un détecteur réel mais instable.** Sous la règle F il
   atteint 0,86 à 0,90 sur trois configurations, mais avec un écart-type de 0,224 sur
   BreastMNIST : selon la graine, la même configuration donne 0,500 ou 1,000. La valeur
   « 1,000 » du rapport à une graine n'était pas reproductible.
2. **Il échoue sur le flou gaussien** (0,514 à 0,520), là même où le gain d'exactitude
   de la section 3 est maximal. Détection et efficience ne vont pas de pair.
3. **La transformation bilatérale récupère partiellement les détecteurs inversés, sans
   les rendre utilisables.** Sur BreastMNIST la règle H passe de 0,24–0,40 à 0,43–0,56,
   c'est-à-dire du franchement inversé au voisinage du hasard. Sur PneumoniaMNIST elle
   ne change rien, parce que les profondeurs y sont quasi constantes (sortie en couche
   ≈ 1) : il n'y a pas de dispersion sur laquelle la transformation puisse opérer.
   Le gain existe donc, mais aucun détecteur fondé sur H et la profondeur n'atteint un
   niveau exploitable.

## 5. Q3 — le critère de qualité d'ancrage

Q_gap = D_inter / distance ancres-nuage, mesuré sur le nuage TRAIN source à la couche 12.
R1 et R3 ne dépendent pas de la graine ; R2 en dépend par son tirage few-shot.

| Cohorte | Régime | Q_gap | Équil. mesurée (couche 12) |
|---|---|---|---|
| BreastMNIST | R1 texte | 0,0799 ± 0,0000 | 0,509 |
| BreastMNIST | R2 few-shot | 0,1637 ± 0,0440 | 0,614 |
| BreastMNIST | R3 centroïdes | 0,0799 ± 0,0000 | 0,658 |
| PneumoniaMNIST | R1 texte | 0,0892 ± 0,0000 | 0,500 |
| PneumoniaMNIST | R2 few-shot | 0,3960 ± 0,1023 | 0,728 |
| PneumoniaMNIST | R3 centroïdes | 0,3546 ± 0,0094 | 0,730 |

1. **Le constat d'exp03 tient** : D_inter seule désigne les ancres textuelles comme les
   meilleures alors qu'elles classent au hasard, parce que le gap de modalité y vaut
   ≈ 1,28 contre ≈ 0,04 à 0,09 pour les ancres de données.
2. **Succès partiel du critère.** Q_gap place les ancres de données devant les ancres
   textuelles sur PneumoniaMNIST, et R2 devant R1 sur BreastMNIST : l'inversion due au
   gap de modalité est corrigée.
3. **Échec confirmé sur le classement fin.** Sur BreastMNIST, Q_gap égalise R1 et R3
   (0,0799 tous deux) alors que leurs exactitudes diffèrent de 0,15, et place R2 devant
   R3 à rebours de la mesure. Sur PneumoniaMNIST l'écart entre R2 et R3 (0,396 ± 0,102
   contre 0,355 ± 0,009) n'est pas séparable compte tenu de la dispersion. Le critère
   sert donc à **rejeter** des ancres détachées du nuage, pas à départager deux jeux
   d'ancres tous deux ancrés dans les données.

## 6. Lecture — honnête

- **Ce qui survit à cinq graines** : les ancres textuelles sont au hasard sur ce
  substrat ; une couche intermédiaire bat la couche de sortie sous ancres de données,
  dans les quatre configurations ; la règle d'arrêt par échantillon est battue par une
  profondeur fixe calibrée ; le critère de qualité d'ancrage rejette les ancres
  détachées du nuage mais ne départage pas les autres.
- **Ce qui ne survit pas** : la localisation de la couche optimale, instable de ±3
  couches ; le gain sous décalage, dont le signe dépend du type de corruption et qui
  est négatif sur les corruptions photométriques ; l'AUROC de 1,000 du détecteur N*,
  qui vaut 0,900 ± 0,224.
- **Ce qui reste à faire avant toute revendication** : un traitement statistique
  apparié (le protocole demande bootstrap apparié et correction de Holm-Bonferroni) ;
  la couche oracle est de plus sélectionnée sur le split d'évaluation, donc optimiste.
- Limites inchangées : substrat P0 uniquement, deux corruptions par cohorte, ancres
  R2/R3 déclarées exploratoires, couches intermédiaires lues en logit lens, R3 approché
  sur PneumoniaMNIST.

## 7. Décision

- Traiter séparément, dans la matrice d'ablation, « couche fixe calibrée » et « règle
  par échantillon » : sur P0 la première domine, et confondre les deux attribuerait à
  l'adaptativité un gain qui vient du choix de profondeur.
- Ne pas revendiquer de gain d'efficience sous décalage sans distinguer corruptions
  structurelles et photométriques : le signe s'inverse entre les deux.
- Appliquer le traitement statistique apparié aux quatre configurations avant de porter
  quoi que ce soit dans le papier.
- Priorité inchangée : la porte de décision sur la cohorte P1, puisque le régime
  principal du protocole classe au hasard sur les deux cohortes P0 disponibles.
