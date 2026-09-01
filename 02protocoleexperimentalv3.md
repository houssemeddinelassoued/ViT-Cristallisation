---
title: "Protocole expérimental"
version: 3
sous_titre: "Encodeur vision-langage gelé — axe de thèse, imagerie médicale et syndrome de Sjögren"
date: 2026-08-31
auteur: "Houssem Eddine Lassoued"
remplace: "claude/02-protocole-experimental.md (v2)"
math: latex
---

# Protocole expérimental — version 3

*Cette version remplace la v2. Elle tire les conséquences du recadrage de l'axe autour d'un encodeur vision-langage gelé de type CLIP, et du terrain échographique de la thèse.*

---

## 0. Ce qui change par rapport à la v2, et pourquoi

La v2 supposait un ViT initialisé sur ImageNet, dont il fallait entraîner les têtes de projection et ancrer les centroïdes par perte contrastive. Le substrat est désormais un **encodeur vision-langage gelé**, ce qui déplace six éléments du protocole.

| Élément | v2 | v3 |
|---|---|---|
| Centroïdes | Appris hors ligne par SupCon | **Plongements textuels des classes** — disponibles sans entraînement |
| Phase 1 | Un sprint d'entraînement complet | Optionnelle en régime zero-shot ; sondes linéaires par couche pour l'arrêt anticipé |
| Baseline de référence | TENT | **TPT et C-TPT** — l'adaptation au moment du test propre aux modèles vision-langage |
| Contrôle décisif | TENT + arrêt calibré | **TPT + arrêt calibré** |
| Seconde modalité | RSNA → ChestX-ray14 | **BUSI → UDIAT → BUS-UCLM** : décalage échographique inter-appareils réel |
| Variable expérimentale nouvelle | — | **La formulation des invites**, qui définit littéralement les ancres |

> Mesurer contre TENT alors que le substrat est CLIP reviendrait à se comparer à la mauvaise référence. C'est la correction la plus importante de cette version.

---

## 1. Règle de tenue des résultats

**Aucune valeur de performance n'est écrite avant d'avoir été produite par un script versionné**, et chaque cellule des tableaux porte l'identifiant du run qui l'a produite. Les tableaux du §8 sont volontairement vides.

---

## 2. Le substrat : encodeur et ancres

### 2.1 Modèles candidats

| Réf | Modèle | Rôle |
|---|---|---|
| M1 | CLIP ViT-B/16 généraliste (`open_clip`, poids OpenAI ou LAION-2B) | Référence principale — c'est l'acquis de la thèse |
| M2 | BiomedCLIP | Vérifie si un pré-entraînement biomédical déplace les conclusions |
| M3 | Encodeur échographique spécialisé, s'il en existe un adapté | Optionnel, subordonné à la disponibilité |

L'ensemble du protocole se déroule sur M1. M2 sert de contrôle de robustesse sur la configuration finale seulement. **Le backbone n'est jamais entraîné.**

### 2.2 Les ancres sont des invites textuelles

$$\mu_k \;=\; \frac{\mathrm{TextEnc}(\text{invite}_k)}{\lVert \mathrm{TextEnc}(\text{invite}_k) \rVert}$$

Trois conséquences structurent le protocole.

**(a) La Phase 1 devient optionnelle.** En régime zero-shot, les ancres existent sans le moindre entraînement. La première mesure du projet ne demande donc aucun GPU sérieux ni aucune annotation.

**(b) La formulation des invites devient une variable expérimentale de premier ordre**, et une source de fuite majeure. Une invite choisie parce qu'elle donne de bons résultats sur le jeu de test invalide toute la démonstration. Les invites sont donc figées sur un split de validation du **domaine source**, avant toute évaluation, et reproduites *in extenso* dans le manuscrit.

**(c) L'ensemble d'invites est une baseline forte et gratuite.** Moyenner les plongements de plusieurs reformulations puis renormaliser améliore souvent nettement le zero-shot. Ne pas l'inclure dans la matrice donnerait une baseline artificiellement faible — et un relecteur le verrait.

### 2.3 Régimes d'ancrage

| Réf | Régime | Étiquettes requises | Statut |
|---|---|---|---|
| R1 | Invites textuelles pures | aucune | Régime principal |
| R2 | Invites + raffinement few-shot des prototypes sur la source | quelques dizaines | Réaliste en clinique |
| R3 | Prototypes entièrement appris (sonde linéaire) | jeu source complet | Plafond de référence |

### 2.4 Température

L'équivalence établie dans le cadre théorique donne $T = 2\tau$, où $\tau$ est la température apprise du modèle. C'est la valeur par défaut. Toute recalibration de $T$ se fait **sur le split de validation source**, jamais sur la cible, et l'écart à $2\tau$ doit être rapporté.

### 2.5 Trois familles de paramètres adaptables

| Réf | $\theta_{\text{adapt}}$ | Voie |
|---|---|---|
| V | Paramètres affines des LayerNorm de l'encodeur visuel | Voie TENT |
| P | Jetons d'invite du côté texte | Voie TPT |
| Vp | Invites visuelles insérées dans l'encodeur d'image | Voie prompt visuel |

La méthode proposée adapte **V + Vp** par défaut, et laisse les ancres textuelles fixes.

> **Une subtilité théorique à signaler dans le manuscrit.** Si $\theta_{\text{adapt}}$ inclut les jetons d'invite (voie P), alors les ancres deviennent $\mu_k(\theta)$ et le gradient du terme de répulsion cesse d'être nul même sans prototypes souples. Autrement dit, la dégénérescence décrite par la Proposition 3 est propre à une adaptation purement visuelle à ancres figées. En contrepartie, des ancres mobiles peuvent dériver l'une vers l'autre : le terme d'ancrage $\beta$ devient alors indispensable, et non plus seulement prudentiel. L'ablation D4 mesure ce compromis.

---

## 3. Cohortes

| Palier | Cohorte | Modalité | Nature du décalage | Rôle |
|---|---|---|---|---|
| **P0** | BreastMNIST-C · PathMNIST-C | Échographie · histologie | Corruptions réalistes, sévérité 1–5 | Mécanique et balayage continu — indispensable à l'analyse de transition |
| **P1** | BUSI → UDIAT → BUS-UCLM | Échographie | **3 centres, 3 appareils, décalage réel** | **Expérience de tête** — la plus proche du terrain de la thèse |
| **P2** | Camelyon17-WILDS | Histopathologie | 5 hôpitaux réels | Proxy des biopsies de glandes salivaires accessoires ; baselines publiées |
| **P3** | Cohorte Sjögren | Échographie salivaire ou histologie | Inter-centre réel | Subordonné à l'accès aux données (question Q3) |

**P1 remplace RSNA → ChestX-ray14 de la v2.** Le raisonnement : la démonstration doit porter sur la modalité du terrain de thèse, et l'échographie mammaire multi-centres offre le seul décalage inter-appareils public, réel et documenté dans cette modalité. La littérature sur la généralisation inter-jeux y est active, ce qui fournit des points de comparaison.

**Tâches.** P0 et P1 : classification binaire bénin / malin. P2 : présence de métastase. P3 : à définir avec les cliniciens — cotation OMERACT ordinale de 0 à 3, ou binaire pSS versus non-pSS.

---

## 4. Matrice d'ablation

### Groupe A — sans adaptation

| Réf | Configuration | Ce que la ligne établit |
|---|---|---|
| A0 | Zero-shot, invite unique | Plancher |
| A1 | Zero-shot, ensemble d'invites | Baseline gratuite, souvent sous-estimée |
| A2 | Recalibration des statistiques LayerNorm visuelles | Baseline sans gradient |

### Groupe B — adaptation au moment du test

| Réf | Configuration | Ce que la ligne établit |
|---|---|---|
| B1 | TENT sur LayerNorm visuel | Voie V, référence historique |
| B2 | **TPT** — invites, entropie marginale sur augmentations | Voie P, référence du domaine vision-langage |
| B3 | **C-TPT** — TPT avec terme de calibration | Établit si notre découplage fait mieux qu'une correction de calibration |
| B4 | SAR ou EATA porté sur l'encodeur visuel | État de l'art TTA généraliste |

### Groupe C — arrêt anticipé seul

| Réf | Configuration | Ce que la ligne établit |
|---|---|---|
| C1 | Arrêt sur confiance, seuils calibrés, sans adaptation | Efficience sans adaptation |

### Groupe D — combinaisons

| Réf | Configuration | Ce que la ligne établit |
|---|---|---|
| **D1** | **TPT + arrêt calibré** | **Contrôle décisif** — la composition naïve des deux briques |
| D2 | Nôtre, $\lambda = 0$ | Isole l'apport du terme de répulsion (Proposition 1) |
| D3 | Nôtre, arrêt sur $\mathcal{H}$ au lieu de $F$ | Quantifie l'effet de circularité (Proposition 2) |
| D4 | Nôtre, adaptation côté texte (voie P) | Mesure le compromis décrit en §2.5 |
| D5 | Nôtre, complet (V + Vp, arrêt sur $F$) | Méthode proposée |

### Plafond hors compétition

| Réf | Configuration | Remarque |
|---|---|---|
| U1 | Sonde linéaire supervisée sur le domaine cible | Utilise des étiquettes cibles — plafond, jamais présenté comme comparable |

**Ordre d'exécution :** A0 → A1 → B2 → C1 → **D1** → D5 → D2 → D3 → B3 → B1 → B4 → D4 → U1.

> **D1 décide du sort de l'axe.** Si D5 ne dépasse pas D1 de façon statistiquement significative sur P1, la contribution se réduit à la composition de deux méthodes existantes. Cette ligne s'exécute donc en cinquième position, pas en dernière.

---

## 5. Métriques

### 5.1 Détection du décalage — objectif O1

**AUROC de l'énergie libre $F$** comme détecteur, domaine source contre domaine cible. C'est la métrique de la toute première expérience, et elle ne demande aucun entraînement.

À reporter conjointement : l'AUROC de l'entropie $\mathcal{H}$ comme détecteur. **La prédiction du cadre théorique est que $F$ sépare nettement et que $\mathcal{H}$ échoue** — c'est le corollaire de la Proposition 2, et c'est falsifiable en une journée.

### 5.2 Diagnostic

AUROC (primaire, robuste au déséquilibre), accuracy équilibrée, F1 macro, sensibilité à spécificité fixée à 0,95.

### 5.3 Calibration — non optionnel

ECE à 15 bacs, score de Brier, diagrammes de fiabilité, **mesurés avant et après adaptation**. C-TPT établit que l'adaptation au moment du test dégrade la calibration des modèles vision-langage ; si notre découplage ne l'évite pas, il faut le dire.

### 5.4 Sensibilité aux invites — propre au substrat

Écart-type de l'exactitude sur au moins cinq reformulations des invites, à protocole identique. Une méthode dont le gain est inférieur à cette dispersion n'a pas démontré grand-chose. **Cette métrique est un garde-fou contre nous-mêmes.**

### 5.5 Coût

FLOPs **passes arrière incluses**, latence p50 et p95 avec synchronisation CUDA explicite, profondeur moyenne de sortie $\bar{N}^\ast$, fraction $\rho$ effectivement adaptée.

> **Point de comptabilité déterminant.** TPT génère plusieurs dizaines de vues augmentées par image et rétropropage à travers l'encodeur : son coût par image se compte en dizaines de passes avant. Le comptabiliser honnêtement est à la fois une exigence de rigueur et, très probablement, l'argument le plus favorable à la méthode proposée. Le mesurer, ne pas l'estimer.

### 5.6 Sécurité clinique

Courbe risque–couverture et AURC sur les cas non cristallisés renvoyés à l'expert.

### 5.7 Diagnostic d'effondrement

Entropie de la distribution marginale des prédictions, par couche. Une chute signale un effondrement.

---

## 6. Traitement statistique et pièges propres au substrat

Cinq graines minimum, moyenne et écart-type, intervalles de confiance par bootstrap apparié, test de DeLong pour les différences d'AUROC, correction de Holm-Bonferroni par tableau.

Quatre fuites à prévenir explicitement, les trois premières étant propres aux modèles vision-langage :

1. **Fuite par les invites** — les invites sont figées sur la validation source, avant toute évaluation cible, et publiées telles quelles.
2. **Fuite par la température** — $T$ vaut $2\tau$ par défaut ; toute recalibration se fait sur la validation source.
3. **Fuite par le pré-entraînement** — CLIP a vu des données web à très grande échelle. Il faut vérifier et déclarer que les cohortes d'évaluation n'y figurent pas de façon identifiable, ou à défaut discuter la limite.
4. **Fuite par les seuils** — $\varepsilon_n$ est calibré sur un split dédié, dont le domaine d'origine est déclaré (cf. les trois régimes de garantie du cadre théorique).

---

## 7. Budget de calcul — RTX 4070, 12 Go

| Étape | Durée estimée |
|---|---|
| Expérience zéro-entraînement sur P0 (inférence seule) | 1 à 3 h |
| Sondes linéaires par couche, P0, backbone gelé | 1 à 2 h |
| Sondes linéaires par couche, P1 | 3 à 5 h |
| Un run d'évaluation complet, P0 | 15 à 30 min |
| Un run TPT complet, P1 | plusieurs heures — le coût des augmentations domine |
| Matrice complète × 5 graines, P1 | 3 à 6 jours de calcul |

Précision mixte `bf16` systématique. L'encodeur restant gelé, l'empreinte mémoire est dominée par les activations de la rétropropagation partielle : les tailles de lot se règlent en conséquence.

---

## 8. Tableaux de résultats — à remplir par mesure

### 8.1 Première expérience — détection du décalage, sans entraînement (P0)

*Rempli le 2026-09-01 depuis le run canonique speckle_noise (ensemble d'invites,
seed 0, commit a82b42d, arbre propre). Balayage complet des 7 corruptions
BreastMNIST-C et de 4 corruptions PneumoniaMNIST-C :
`tlsc-repo/docs/rapport-exp01.md` §7 et `tlsc-repo/outputs/aggregate/`.*

| Sévérité | Exactitude | $\mathcal{H}$ moyenne | $F$ moyenne | AUROC de $F$ | AUROC de $\mathcal{H}$ | run id |
|---|---|---|---|---|---|---|
| 0 (source) | 0,712 | 0,5165 | 1,2878 | — | — | 20260901T004257Z_breastmnist_de6dba0c |
| 1 | 0,686 | 0,5657 | 1,2844 | 0,461 | 0,619 | 20260901T004257Z_breastmnist_de6dba0c |
| 3 | 0,756 | 0,5800 | 1,3181 | 0,686 | 0,659 | 20260901T004257Z_breastmnist_de6dba0c |
| 5 | 0,712 | 0,6108 | 1,3514 | 0,814 | 0,734 | 20260901T004257Z_breastmnist_de6dba0c |

> **Verdict pré-enregistré (sévérité 5)** : C1 ✓ (0,814 ≥ 0,70), C3 ✓ (+0,080 ≥ 0,05),
> C2 ✗ (0,734 > 0,60) → **prédiction non confirmée**, sur les 11 configurations
> canoniques testées. Résultat exploitable issu de la réfutation : $F$ est un
> détecteur **signé** — il passe sous le hasard sur les corruptions photométriques
> (contrast_down 0,299 ; brightness_down 0,359 ; radiographie 0,127–0,209) — et la
> variante bilatérale $|F - \mathrm{med}_{\mathrm{source}}(F)|$ récupère 5 des
> 6 inversions sans toucher au bloc structurel. Papier : `tlsc-repo/paper/main.tex`.

### 8.2 Résultat principal (P1 — BUSI → UDIAT → BUS-UCLM)

| Réf | Méthode | AUROC | ECE | Écart-type invites | FLOPs rel. | $\bar{N}^\ast$ | $\rho$ | run id |
|---|---|---|---|---|---|---|---|---|
| A0 | Zero-shot | | | | 1,00 | 12,0 | 0 | |
| A1 | Ensemble d'invites | | | | | 12,0 | 0 | |
| B2 | TPT | | | | | 12,0 | 1,00 | |
| C1 | Arrêt calibré | | | | | | 0 | |
| **D1** | **TPT + arrêt calibré** | | | | | | 1,00 | |
| D5 | Cristallisation (nôtre) | | | | | | | |

### 8.3 Ablations

| Réf | Variante | AUROC | Δ vs D5 | Entropie marginale | run id |
|---|---|---|---|---|---|
| D2 | $\lambda = 0$ | | | | |
| D3 | Arrêt sur $\mathcal{H}$ | | | | |
| D4 | Adaptation côté texte | | | | |
| D5 | Complet | | — | | |

### 8.4 Garantie conforme

| Régime de calibration | $\alpha$ visé | Risque empirique | Couverture | Garantie |
|---|---|---|---|---|
| Source → source | 0,01 | | | valide (échangeabilité) |
| Source → cible | 0,01 | | | **approchée — à discuter** |
| Cible tenue à l'écart → cible | 0,01 | | | valide |

---

## 9. Correspondance objectifs → expériences

| Objectif | Expérience | Jalon |
|---|---|---|
| O1 — les observables de Gibbs détectent le décalage | §8.1, sans entraînement | J1 |
| O2 — l'adaptation sélective restaure la performance inter-centre | D5 contre D1 sur P1 | **J2** |
| O3 — l'arrêt réduit le coût sans dégradation garantie | §8.4 + comptabilité §5.5 | J3 |
| O4 — la répulsion prévient l'effondrement | D2 contre D5 | J2 |
| O5 — portabilité vers une seconde modalité | P2, puis P3 | J5 |

---

## 10. Séquence recommandée

1. **Expérience zéro-entraînement sur P0** — une journée, aucune annotation, produit le §8.1 et teste le corollaire de la Proposition 2.
2. **Sondes linéaires par couche** — donne la trajectoire $\mathcal{H}_n(n)$ et rend l'arrêt anticipé possible. Jalon J1.
3. **A0, A1, B2, C1, D1, D5 sur P1** — le verdict sur la contribution. Jalon J2.
4. **Comptabilité honnête du calcul** — jalon J3, où le coût réel de TPT joue en notre faveur.
5. **P2, puis P3 si l'accès aux données est obtenu.**

Le point à retenir : **la première mesure réelle du projet ne demande ni entraînement, ni annotations, ni GPU sérieux.** Elle est à portée d'une journée de travail, et elle teste directement l'énoncé central.
