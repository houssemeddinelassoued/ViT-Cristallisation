# Rapport de résultats — Expérience 5 (ancres × arrêt calibré, cohorte P0)

**Date du rapport** : 2026-09-05
**Expérience** : `exp05_anchored_early_exit` — croisement des chantiers exp03 (régimes
d'ancrage) et exp04 (arrêt anticipé). L'arrêt calibré est rejoué sous les trois régimes,
avec des centroïdes R2/R3 **recalculés couche par couche** sur le split train source
(`tlsc/models/data_anchors.py`, `tlsc/eval/early_exit.py`).
**Modèle** : ViT-B-16-quickgelu · poids `openai` (CLIP gelé), T = 2τ ≈ 0,0200, 12 blocs.
**Cohortes** : BreastMNIST 224×224 (test n = 156, split 78 calibration / 78 évaluation) et
PneumoniaMNIST 224×224 (test n = 624, split 312 / 312). Corruptions MedMNIST-C,
sévérités 0–5.
**Statut épistémique** : **exploratoire déclaré** — aucun seuil de verdict pré-enregistré.
**Anti-fuite, trois barrières** : (1) centroïdes R2/R3 sur le split TRAIN source à
sévérité 0 seulement ; (2) seuils ε et couche fixe sur la moitié CALIBRATION du split test
source ; (3) toutes les valeurs rapportées, source comme cible, sur la moitié ÉVALUATION.

> **Changement de critère par rapport à l'exp04 — à lire avant toute comparaison.**
> L'exp04 calibrait sur l'exactitude *brute*. Le premier run de l'exp05 a montré que
> cette exactitude vaut 0,731 à chacune des douze couches sous ancres textuelles, soit
> exactement la proportion de la classe majoritaire de BreastMNIST : le critère était
> donc maximisé par le prédicteur dégénéré qui répond toujours la même classe, et
> validait une sortie en couche 1 sans contenu diagnostique. L'exp05 calibre sur
> l'exactitude **équilibrée** (`calibrate_epsilon(..., y=...)`, deux tests dédiés). Les
> profondeurs de l'exp04 et celles de l'exp05 ne sont donc pas directement comparables.

---

## 1. Runs (arbre propre pour les quatre)

| run_id | Cohorte | Corruption | git | n train source | n éval | s |
|---|---|---|---|---|---|---|
| 20260904T232709Z_breastmnist_7d1b0ed1 | BreastMNIST | speckle_noise | 973b931 | 546 | 78 | 275 |
| 20260904T233155Z_breastmnist_6f91783e | BreastMNIST | brightness_down | 973b931 | 546 | 78 | 283 |
| 20260904T233649Z_pneumoniamnist_9b5f183e | PneumoniaMNIST | gaussian_blur | 973b931 | 1200 | 312 | 1100 |
| 20260905T001255Z_pneumoniamnist_6044eb14 | PneumoniaMNIST | brightness_down | 2abe4c0 | 1200 | 312 | 958 |

Les quatre runs portent `environment.git_dirty: false`. Le quatrième a été relancé : sa
première exécution avait démarré sur un arbre sali par un fichier sans effet sur le code
scientifique, ce qui la rendait non citable ; la relance reproduit ses valeurs à
l'identique, champ pour champ.

Le split train de PneumoniaMNIST est sous-échantillonné de façon stratifiée à 1200 images
(`--train-limit`) pour borner le coût d'encodage par couche ; R3 y est donc un plafond
approché, non le plafond exact du split complet.

## 2. Q1 — la profondeur utile cesse-t-elle d'être N* = 1 ?

Exactitude **équilibrée** par couche, domaine source, split évaluation.

| Cohorte | Régime | Couche oracle | Équil. à l'oracle | Équil. couche 1 | Équil. couche 12 |
|---|---|---|---|---|---|
| BreastMNIST | R1 texte | 12 | 0,521 | 0,500 | 0,521 |
| BreastMNIST | R2 few-shot k=16 | 2 | 0,719 | 0,610 | 0,604 |
| BreastMNIST | R3 centroïdes | 3 | 0,684 | 0,610 | 0,640 |
| PneumoniaMNIST | R1 texte | 1 | 0,500 | 0,500 | 0,492 |
| PneumoniaMNIST | R2 few-shot k=16 | 9 | 0,841 | 0,773 | 0,705 |
| PneumoniaMNIST | R3 centroïdes | 6 | 0,825 | 0,791 | 0,740 |

1. **Les ancres textuelles sont au hasard, pas seulement faibles.** L'exactitude
   équilibrée vaut 0,500 à onze couches sur douze sur les deux cohortes. La « préservation
   d'exactitude » de l'exp04 était la préservation du taux de la classe majoritaire.
   La dégénérescence N* = 1 avait donc **deux causes cumulées** : des ancres aveugles et
   un critère de calibration que le prédicteur majoritaire maximise.
2. **Sous ancres de données, un optimum intérieur apparaît, loin de la sortie.** La couche
   optimale vaut 2 ou 3 sur BreastMNIST, 6 ou 9 sur PneumoniaMNIST — jamais 1, jamais 12.
   « argmin_N » cesse d'être dégénéré : le test non trivial demandé en fin d'exp04 est
   exécuté, et la profondeur est bien une variable.
3. **Les derniers blocs détruisent le signal.** Sur PneumoniaMNIST sous R2, l'équilibrée
   passe de 0,841 en couche 9 à 0,705 en couche 12 ; sous R3, de 0,825 en couche 6 à
   0,740. Lues avec des ancres de données, les trois dernières couches coûtent du calcul
   *et* de la performance.

## 3. Q1 (suite) — comportement sous décalage

PneumoniaMNIST · gaussian_blur, équilibrée sur le split évaluation.

| Sévérité | R2 couche fixe 9 | R2 pleine profondeur | R3 couche fixe 9 | R3 pleine profondeur |
|---|---|---|---|---|
| 0 | 0,841 | 0,705 | 0,824 | 0,740 |
| 3 | 0,813 | 0,617 | 0,788 | 0,608 |
| 5 | 0,780 | 0,539 | 0,674 | 0,535 |

**La sortie anticipée n'est pas seulement moins chère, elle est nettement plus robuste
au décalage.** À sévérité 5, la couche fixe calibrée conserve 0,780 là où la pleine
profondeur tombe à 0,539, pour 75 % du calcul. L'écart se creuse avec la sévérité
(+0,136 à sévérité 0, +0,241 à sévérité 5).

**Mais la règle par échantillon du croquis fondateur perd contre une couche fixe.**
Les règles calibrées H-stop et F-stop sortent en couche ≈ 1,1 et n'atteignent que
0,622 et 0,652 à sévérité 5, très en deçà des 0,780 de la couche fixe 9. Sur ce
substrat, « sortir quand H < ε », échantillon par échantillon, est battu par
« choisir une profondeur sur la calibration ». Consigné tel quel.

## 4. Q2 — la profondeur de sortie comme détecteur de décalage

AUROC des profondeurs N* source contre cible, sévérité 5, split évaluation
(`exit_depth_auroc`).

| Cohorte · corruption | Régime | AUROC (règle H) | AUROC (règle F) |
|---|---|---|---|
| Breast · speckle_noise | R2 | 0,442 | **1,000** |
| Breast · speckle_noise | R3 | **0,059** | 0,808 |
| Breast · brightness_down | R2 | 0,436 | **1,000** |
| Breast · brightness_down | R3 | **0,167** | 0,808 |
| Pneumonia · gaussian_blur | R2 | 0,478 | 0,503 |
| Pneumonia · gaussian_blur | R3 | 0,482 | 0,502 |
| Pneumonia · brightness_down | R2 | 0,511 | 0,835 |
| Pneumonia · brightness_down | R3 | 0,499 | 0,825 |

1. **Le compteur de couches est un détecteur réel, gratuit en production**, mais pas
   universel : parfait sur BreastMNIST sous R2 (1,000 aux deux corruptions), fort sur
   PneumoniaMNIST photométrique (0,835 et 0,825), au hasard sur PneumoniaMNIST
   gaussian_blur (0,503).
2. **Il hérite du caractère *signé* de son observable.** Sous R3 sur BreastMNIST, la
   règle H donne 0,059 et 0,167 : loin d'être aveugle, le détecteur est fortement
   **inversé** — les images corrompues cristallisent *plus tôt*, pas plus tard. C'est la
   signature déjà rencontrée pour F à l'exp01. La transformation bilatérale
   `two_sided_shift_score` devrait s'y appliquer et récupérer ces cas ; non testé ici.
3. L'observation qualitative de l'exp04 (« N̄* croît avec la sévérité ») est donc
   quantifiée et **restreinte** : elle ne vaut ni pour tous les régimes, ni pour toutes
   les corruptions, et son sens peut s'inverser.

## 5. Q3 — le critère de qualité d'ancrage

Mesuré sur le nuage TRAIN source, couche 12 (`anchor_quality`).

| Cohorte | Régime | D_inter | d_cloud | Q_gap | Q_fisher | Équil. mesurée (couche 12) |
|---|---|---|---|---|---|---|
| BreastMNIST | R1 texte | 0,1032 | 1,2921 | 0,0799 | 0,0795 | 0,521 |
| BreastMNIST | R2 few-shot | 0,0208 | 0,0932 | 0,2235 | 0,2072 | 0,604 |
| BreastMNIST | R3 centroïdes | 0,0073 | 0,0908 | 0,0799 | 0,0778 | 0,640 |
| PneumoniaMNIST | R1 texte | 0,1137 | 1,2750 | 0,0892 | 0,0889 | 0,492 |
| PneumoniaMNIST | R2 few-shot | 0,0250 | 0,0438 | 0,5701 | 0,5522 | 0,705 |
| PneumoniaMNIST | R3 centroïdes | 0,0144 | 0,0420 | 0,3416 | 0,3344 | 0,740 |

1. **Le constat d'exp03 est reproduit sur les deux cohortes** : D_inter seule désigne les
   ancres textuelles comme les meilleures (0,1032 et 0,1137, un ordre de grandeur
   au-dessus), alors qu'elles classent au hasard. La distance ancres-nuage montre
   pourquoi : le gap de modalité y vaut ≈ 1,28 contre ≈ 0,04 à 0,09 pour les ancres de
   données.
2. **Succès partiel du critère.** Q_gap et Q_fisher placent les ancres de données devant
   les ancres textuelles partout : l'inversion due au gap de modalité est corrigée.
3. **Échec sur le classement fin.** Les deux rapports placent R2 devant R3 sur les deux
   cohortes, alors que l'exactitude mesurée place R3 devant (0,640 contre 0,604 ;
   0,740 contre 0,705). Mécanisme identifié : les centroïdes complets moyennent beaucoup
   d'exemples et se rapprochent de la moyenne globale, ce qui écrase D_inter, tandis que
   le bruit d'échantillonnage à k = 16 maintient les centroïdes R2 écartés. Le critère
   est donc utilisable pour **rejeter** des ancres détachées du nuage, pas pour départager
   deux jeux d'ancres tous deux ancrés dans les données.

## 6. Lecture — honnête

- Le chantier demandé en fin d'exp03 et exp04 est exécuté. Sa conclusion principale n'est
  pas celle qu'on attendait : sous ancres utiles, la profondeur est bien une variable qui
  compte, mais la forme *adaptative par échantillon* du croquis fondateur est battue par
  une simple couche fixe calibrée. Ce qui survit du croquis, c'est « toutes les couches ne
  se valent pas », pas « chaque image doit choisir la sienne ».
- Le résultat le plus exploitable est ailleurs, et n'était pas visé : **les dernières
  couches de l'encodeur visuel dégradent le diagnostic sous décalage**, et une profondeur
  intermédiaire calibrée sur la source seule récupère jusqu'à +0,241 d'exactitude
  équilibrée à sévérité maximale, pour un quart de calcul en moins.
- Limites. Substrat P0 uniquement, une graine, deux corruptions par cohorte ; ancres R2/R3
  déclarées exploratoires (le protocole v3 garde R1 comme régime principal) ; couches
  intermédiaires lues en logit lens, jamais alignées à l'espace texte par l'entraînement ;
  R3 approché sur PneumoniaMNIST (`--train-limit 1200`). Aucun intervalle de confiance
  n'accompagne ces chiffres : le traitement statistique apparié (§6 du protocole) reste à
  appliquer avant toute revendication.

## 7. Décision

- Appliquer la transformation bilatérale aux profondeurs de sortie et vérifier si elle
  récupère les détecteurs inversés (0,059 et 0,167 sous R3).
- Comparer explicitement « couche fixe calibrée » et « règle par échantillon » comme deux
  lignes distinctes de la matrice d'ablation : sur P0 la première domine, ce qui doit être
  dit avant d'écrire quoi que ce soit sur l'arrêt anticipé.
- Refaire ce croisement avec plusieurs graines et un traitement statistique apparié avant
  d'en tirer une revendication.
- Le passage à la cohorte P1 (BUSI → UDIAT → BUS-UCLM) reste la priorité du protocole ;
  ces résultats P0 fixent la profondeur et le régime d'ancrage à y tester en premier.
