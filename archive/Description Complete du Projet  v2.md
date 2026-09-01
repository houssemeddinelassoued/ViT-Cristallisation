---
title: "Description Complète du Projet"
subtitle: "Thermodynamic Latent Space Crystallization"
version: 2
date: 2026-08-30
auteur: "Houssem Eddine Lassoued"
encadrement: "Dr. Anis Ben Aicha, Pr. Habib Fathallah"
math: latex
---

# Description Complète du Projet

*Thermodynamic Latent Space Crystallization*

Version 2 — 30 août 2026. Remplace intégralement la version 1.

Houssem Eddine Lassoued — Encadrement : Dr. Anis Ben Aicha, Pr. Habib Fathallah

> **Statut** Cette version corrige douze défauts identifiés lors de l'audit du 30 août 2026, dont trois étaient bloquants : un terme d'énergie sans gradient, une circularité entre adaptation et sortie anticipée, et l'absence de têtes de lecture par couche. Elle supprime également le tableau de performances de la version 1, dont les valeurs n'avaient jamais été mesurées et étaient mutuellement incohérentes. **Le registre complet des corrections figure en section 6.**

## 1. Genèse, contexte et vision globale

### 1.1 Le problème initial : la fragilité des Vision Transformers

En imagerie médicale clinique — IRM, scanner CT, radiographies DICOM, histopathologie, dermoscopie — les données d'entrée varient continûment selon le constructeur du scanner, les doses de radiation, les produits de contraste, les protocoles de coloration, les bruits d'acquisition et le mouvement des patients.

Un Vision Transformer entraîné en laboratoire et confronté à ces images altérées projette celles-ci dans son espace latent avec des poids immuables. La projection perd sa structure, les représentations vectorielles se dispersent, et l'incertitude de décision augmente — avec des conséquences diagnostiques potentiellement critiques.

### 1.2 Le changement de paradigme : la cristallisation latente

Plutôt que de traiter l'inférence comme une passe avant figée, le projet modélise l'espace de représentation comme un système thermodynamique doté d'une mesure de Gibbs. Lorsqu'une image inconnue ou dégradée entre dans le réseau, le modèle ajuste dynamiquement sa profondeur d'analyse et réaligne une fraction minime de ses paramètres au moment du test. L'espace latent passe d'un état désordonné à un état ordonné autour des centroïdes de diagnostic.

L'analogie n'est pas décorative, à une condition qui structure tout le projet : **un cristal exige la condensation et le maintien de sites distincts.** Une condensation vers un site unique n'est pas un cristal — c'est un effondrement. La distinction est formalisée en section 2.4 et constitue la contribution théorique centrale.

### 1.3 Ce que le projet revendique, et ce qu'il ne revendique pas encore

Le projet revendique une formulation : un cadre énergétique où l'adaptation au moment du test et la sortie anticipée dérivent d'observables thermodynamiques distinctes et mesurables, avec une garantie statistique sur la dégradation induite par la sortie anticipée.

**Il ne revendique aucune valeur de performance à ce stade.** L'amplitude du gain hors-domaine, l'économie de calcul et la réalité de la transition de phase sont des *questions expérimentales ouvertes*, dont la réponse conditionne la narration finale du travail.

## 2. Fondations théoriques

### 2.1 Notation

Toute notation employée dans le code et le manuscrit est fixée ici ; en cas de divergence, ce tableau fait foi.

| **Symbole** | **Type**      | **Signification**                                       |
|-------------|---------------|---------------------------------------------------------|
| x           | $\mathbb{R}^{3\times 224\times 224}$ | Image d'entrée                                          |
| L           | $\mathbb{N}$             | Nombre de blocs Transformer, L = 12                     |
| D           | $\mathbb{N}$             | Largeur du backbone : 384 (ViT-S/16), 768 (ViT-B/16)    |
| $h^{(n)}_{\mathrm{cls}}$   | $\mathbb{R}^{D}$           | Jeton [CLS] en sortie du bloc n                       |
| $g_n$         | $\mathbb{R}^{D} \to \mathbb{R}^{d}$     | Tête de projection de la couche n (LayerNorm + Linear)  |
| $z^{(n)}$        | $\mathbb{S}^{d-1}$       | Embedding projeté puis normalisé                        |
| d           | $\mathbb{N}$             | Dimension latente, hyperparamètre ∈ {128, 256, 512}     |
| $\mu_k^{(n)}$     | $\mathbb{S}^{d-1}$       | Centroïde ancré de la classe k à la couche n            |
| T           | $\mathbb{R}_{>0}$          | Température de la mesure de Gibbs                       |
| $\theta_{\text{adapt}}$ | —             | LayerNorm du backbone ∪ prompts visuels                 |
| $\varepsilon_n$       | $\mathbb{R}$             | Seuil de sortie de la couche n — calibré, jamais choisi |
| $N^{\ast}$         | $\{1,\dots,L\}$        | Profondeur de sortie effective                          |

> **Correction** L'espace $\mathbb{R}^{512}$ mentionné dans la version 1 n'est pas l'espace natif du backbone : la largeur de ViT-B/16 est 768. $\mathbb{R}^{d}$ est l'image de la tête de projection $g_n$, et d est un hyperparamètre à ablater.
>
> **Correction** La version 1 annonçait 0,25 % de paramètres pour les LayerNorm et 0,75 % au total. Le comptage exact donne, pour ViT-B/16 : 25 LayerNorm × 2 × 768 = 38 400 paramètres sur 86,6 M, soit **0,044 %**, et 0,053 % avec dix prompts visuels. Pour ViT-S/16 : 0,087 % et 0,105 %. Les valeurs annoncées étaient surestimées d'un facteur 6 à 15 — la correction joue en faveur du projet, mais le chiffre publié devra être produit par le code, jamais recopié.

### 2.2 La mesure de Gibbs et ses trois observables

Pour un embedding z et des centroïdes ancrés, on définit l'énergie de configuration $E_k(z) = \lVert z - \mu_k \rVert^{2}$ et le postérieur de Boltzmann :

$$
p_k(z) \;=\; \frac{\exp\!\big(-E_k(z)/T\big)}{Z(z)}\,,\qquad Z(z) \;=\; \sum_{j=1}^{K} \exp\!\big(-E_j(z)/T\big)
$$

Ce n'est pas un softmax de logits appris, mais une distribution de Boltzmann sur un système à K états ; T est une température réelle et calibrable. Trois observables en découlent, toutes interprétables :

- **Entropie de Gibbs** — $\mathcal{H} = -\sum_k p_k \ln p_k$ : le désordre de l'assignation.

- **Énergie libre** — $F = -T \ln Z$ : exactement le score énergétique de Liu et al. (2020), détecteur hors-distribution reconnu.

- **Énergie moyenne** — $\langle E \rangle = \sum_k p_k E_k$.

**Lemme 1 (identité thermodynamique).** $F = \langle E \rangle - T\,\mathcal{H}$.

*Démonstration.* $\ln p_k = -E_k/T - \ln Z$, donc $\mathcal{H} = \langle E \rangle / T + \ln Z$ ; en multipliant par T, $T\mathcal{H} = \langle E \rangle + T\ln Z = \langle E \rangle - F$. ∎

Ce lemme est vérifié à la précision machine par un test unitaire du dépôt.

### 2.3 Pourquoi l'énergie libre, et non l'entropie, gouverne la sortie

**Proposition 2.** Sous une translation uniforme du paysage énergétique, E_k ↦ E_k + c pour tout k, l'entropie est invariante et l'énergie libre est covariante : H ↦ H, F ↦ F + c.

*Démonstration.* Le facteur exp(−c/T) se simplifie dans la normalisation, donc p est inchangé et H aussi. En revanche Z ↦ exp(−c/T)·Z, donc F ↦ F + c. ∎

**Corollaire.** Une image équidistante mais très éloignée de tous les centroïdes — la signature typique d'un échantillon hors-domaine — possède une entropie basse tout en ayant une énergie libre élevée. **L'entropie seule ne peut donc pas distinguer une décision confiante d'une décision confiante mais hors du domaine de validité du modèle.** L'énergie libre le peut.

> **Correction** La version 1 adaptait le modèle pour minimiser l'entropie, puis sortait dès que l'entropie passait sous un seuil. L'économie de calcul était donc **mécaniquement garantie même lorsque le modèle se trompait** : elle mesurait une sur-confiance induite, non une meilleure compréhension. La Proposition 2 fonde le découplage qui supprime cette circularité — on adapte sur H, on sort sur F.

### 2.4 L'énergie en ligne et l'exclusion de l'effondrement

Sur un tampon de test B non étiqueté, on définit des prototypes souples différentiables :

$$
\hat{\mu}_k(\theta) \;=\; \frac{\sum_{i \in B} p_k(z_i)\, z_i}{\sum_{i \in B} p_k(z_i)}
$$

d'où l'énergie minimisée au moment du test, exclusivement sur $\theta_{\text{adapt}}$ :

$$
\mathcal{E}_{\text{online}} \;=\; \underbrace{\frac{1}{|B|}\sum_{i \in B}\mathcal{H}\big(p(z_i)\big)}_{\text{condensation}}\;-\;\underbrace{\lambda\,\big\lVert \hat{\mu}_0 - \hat{\mu}_1 \big\rVert^2}_{\text{répulsion}}\;+\;\underbrace{\beta \sum_{k}\big\lVert \hat{\mu}_k - \mu_k \big\rVert^2}_{\text{ancrage}}
$$

soit, dans l'ordre : condensation, répulsion anti-effondrement, ancrage anti-dérive.

**Proposition 3.** Si $\mu_0$ et $\mu_1$ sont des constantes ancrées, alors le gradient de ||$\mu_0$ − $\mu_1$||² par rapport à theta est nul, et minimiser H − $\lambda$·||$\mu_0$ − $\mu_1$||² équivaut exactement à minimiser H.

> **Correction** C'est le défaut le plus grave de la version 1 : les centroïdes y étaient décrits comme « ancrés de manière immuable », ce qui rendait le terme répulsif **rigoureusement inerte**. La méthode décrite était donc identique à TENT (Wang et al., ICLR 2021), et la contribution thermodynamique se réduisait à un habillage lexical. Les prototypes souples ci-dessus rétablissent un gradient non nul.

**Proposition 1 (dégénérescence du minimiseur entropique, et son exclusion).** Soit S_col l'ensemble des configurations effondrées, où tous les échantillons du tampon sont assignés à un même indice.

- Toute configuration de S_col est un minimiseur global du terme de condensation seul, puisqu'elle atteint H = 0, borne inférieure de l'entropie.

- Sur S_col, les prototypes souples convergent vers la même moyenne, donc la répulsion tend vers 0 et l'énergie en ligne tend vers un réel positif ou nul.

- Une configuration séparée atteignant H ≈ 0 avec une répulsion Δ > 0 et un ancrage exact a une énergie −$\lambda$·Δ < 0.

Par conséquent, **pour tout $\lambda$ > 0, aucune configuration effondrée n'est un minimiseur global de l'énergie en ligne dès qu'une configuration séparée est atteignable.** ∎

> **Diagnostic obligatoire** La minimisation d'entropie non régularisée converge vers l'assignation de tous les échantillons à une classe unique — mode d'échec documenté par EATA et SAR, et objet d'une littérature 2025-2026 active. Or un espace effondré présente exactement la signature d'une cristallisation réussie : entropie basse, vecteurs condensés. Il faut donc reporter à chaque couche **l'entropie de la distribution marginale des prédictions**. Une chute de cette quantité signale un effondrement, quelle que soit la qualité apparente des autres courbes.

## 3. Architecture algorithmique en trois phases

### 3.1 Phase 1 — Ancrage hors ligne, couche par couche

Backbone gelé. Pour chaque couche n, une tête de projection légère est entraînée par perte contrastive supervisée sur la cohorte source, puis les centroïdes sont figés comme moyennes de classe renormalisées.

> **Correction** La version 1 ne prévoyait qu'un seul jeu de centroïdes, dans l'espace de la couche 12. Or les représentations intermédiaires vivent dans une géométrie différente : comparer un embedding de la couche 5 à un centroïde de la couche 12 n'a **aucun sens métrique**, et l'entropie de la Phase 3 était donc incalculable. Il faut L jeux de centroïdes et L têtes de projection — coût marginal, quelques millions de paramètres.

Sur la sphère unité, $\lVert z - \mu \rVert^{2} = 2 - 2\langle z, \mu \rangle \in [0,\,4]$ : l'énergie est bornée, ce qui stabilise numériquement la mesure de Gibbs et donne à T une échelle interprétable.

### 3.2 Phase 2 — Adaptation sélective au moment du test

L'adaptation n'est pas systématique. Trois mécanismes la rendent compatible avec un gain net de calcul :

- **Déclenchement conditionnel —** la TTA n'est lancée que si l'énergie libre à une couche sonde dépasse un seuil de détection hors-domaine. Une image typique cristallise tôt et n'est jamais adaptée.

- **Troncature —** la rétropropagation ne remonte que jusqu'à la couche sonde, pas jusqu'à la couche 1.

- **Amortissement —** les paramètres adaptables sont partagés sur un tampon glissant du même domaine, puis réinitialisés à la détection d'un changement de domaine.

### 3.3 Phase 3 — Sortie anticipée à risque contrôlé

À chaque couche, le critère de sortie est l'énergie libre, éventuellement combinée à l'accord entre la prédiction courante et une moyenne mobile des couches profondes. Les seuils ne sont pas choisis : ils sont calibrés par la procédure Learn-then-Test, qui garantit que la dégradation d'exactitude imputable à la sortie anticipée reste sous un niveau $\alpha$ avec une confiance 1 − $\delta$.

> **Correction** La version 1 fixait un seuil unique de 0,45 nats, constant sur les douze couches. Trois problèmes se cumulaient : le seuil n'était pas normalisé — l'entropie maximale vaut ln 2 = 0,693 pour deux classes mais ln 14 = 2,64 pour ChestX-ray14 ; il ignorait la décroissance structurelle de l'entropie avec la profondeur ; et pour K = 2 il correspondait à une confiance de 0,87 seulement, barre faible pour une décision diagnostique.
>
> **Réserve à conserver dans le manuscrit** La garantie conforme suppose l'échangeabilité entre calibration et test — hypothèse *violée par construction* sous décalage de distribution. Trois régimes doivent être rapportés séparément : source vers source (garantie valide), source vers cible (garantie approchée, à valider empiriquement), et domaine cible tenu à l'écart vers cible (valide). Omettre cette réserve serait la faille la plus facilement exploitable en relecture.

### 3.4 Comptabilité du calcul

Une passe arrière coûte environ le double d'une passe avant, et doit traverser tout le réseau pour atteindre les LayerNorm des couches basses — y compris lorsque seuls 0,05 % des paramètres reçoivent une mise à jour. Soit C(n) le coût d'une passe avant tronquée à la couche n :

$$
C_{\text{tot}} \;=\; \rho \cdot S \cdot (1+\kappa)\, C(n_{\text{sonde}})\;+\; C(N^{\ast})
$$

où S est le nombre d'itérations d'adaptation, $\kappa$ ≈ 2 le coût relatif de la passe arrière, et $\rho$ la fraction d'images effectivement adaptées. La condition de gain net s'écrit C_total < C(L). Avec S = 1, $\kappa$ = 2, une couche sonde en position 6 et une profondeur de sortie moyenne de 6, elle devient :

$$
\rho \;<\; \tfrac{1}{3}
$$

**Le gain net existe si et seulement si moins d'un tiers des images sont adaptées.** C'est une prédiction quantitative falsifiable, à confronter à la mesure.

> **Correction** Le tableau de la version 1 annonçait 24,6 ms pour ViT-B statique et 28,1 ms pour TENT. Or TENT effectue une passe avant et une passe arrière : son coût par image devrait avoisiner 70 ms. La méthode proposée, avec trois itérations, partirait de l'ordre de 220 ms, et n'atteindrait pas 14,2 ms même en sortant à la couche 5. **Les valeurs annoncées étaient physiquement incohérentes entre elles, d'un facteur voisin de dix.**

## 4. Protocole expérimental

### 4.1 Règle de tenue des résultats

> **Règle absolue Aucune valeur de performance n'est écrite avant d'avoir été produite par un script versionné,** et chaque cellule des tableaux porte l'identifiant du run qui l'a produite. Les valeurs 87,3 % / −42,5 % / 5,4 couches / 14,2 ms de la version 1 sont caduques et ne doivent réapparaître nulle part, pas même comme ordre de grandeur attendu.

### 4.2 Cohortes

| **Palier** | **Cohorte**         | **Nature du décalage**                             | **Rôle**                                                                   |
|------------|---------------------|----------------------------------------------------|----------------------------------------------------------------------------|
| P0         | MedMNIST-C          | Corruptions réalistes par modalité, sévérité 1 à 5 | Mécanique, itération rapide, balayage continu pour l'analyse physique      |
| P1         | Camelyon17-WILDS    | 5 hôpitaux réels, variabilité de coloration H&E    | Expérience de tête : décalage réel, split OOD officiel, baselines publiées |
| P2         | RSNA → ChestX-ray14 | Inter-institution, acquisition DICOM               | Généralisation à une seconde modalité                                      |
| P3         | ISIC 2019 → 2020    | Inter-appareil dermoscopique                       | Optionnel, si P1 et P2 concluent                                           |

> **Correction** Camelyon16 est écarté : ses lames entières gigapixel demanderaient plusieurs semaines de prétraitement sur le matériel disponible. Camelyon17-WILDS capture le même phénomène biologique sous forme de patchs exploitables, avec en prime une partition inter-hôpitaux officielle et des points de comparaison publiés. Par ailleurs, les corruptions synthétiques servent à *comprendre* — elles seules offrent un paramètre de contrôle continu — tandis que le décalage réel sert à *démontrer*. Un travail reposant uniquement sur du bruit gaussien ajouté ne convaincra pas.

### 4.3 Baselines et matrice d'ablation

| **Réf** | **Configuration**                        | **Ce que la ligne établit**                              |
|---------|------------------------------------------|----------------------------------------------------------|
| A0      | Source-only, statique                    | Plancher de référence                                    |
| A1      | Recalibration des statistiques LayerNorm | Baseline forte, souvent omise dans la littérature        |
| A2      | TENT                                     | Référence historique (ICLR 2021)                         |
| A3      | EATA · SAR · DeYO                        | État de l'art 2022-2024                                  |
| A4      | Sortie anticipée calibrée seule          | Efficience sans adaptation                               |
| A5      | TENT + sortie calibrée                   | Contrôle décisif : la composition naïve des deux briques |
| A6      | Nôtre, $\lambda$ = 0                        | Isole l'apport du terme répulsif (Proposition 1)         |
| A7      | Nôtre, sortie sur H au lieu de F         | Quantifie l'effet de circularité (Proposition 2)         |
| A8      | Nôtre, complet                           | Méthode proposée                                         |

**A5 décide du sort du projet.** Si A8 n'excède pas A5 de façon statistiquement significative sur P1, la contribution se réduit à une reformulation lexicale de deux méthodes existantes. Cette ligne doit être exécutée tôt, et non en fin de parcours.

### 4.4 Métriques

- **Diagnostic —** AUROC (primaire, robuste au déséquilibre), accuracy équilibrée, F1 macro, sensibilité à spécificité fixée à 0,95.

- **Calibration —** ECE à 15 bacs, score de Brier, diagrammes de fiabilité. Non optionnel : c'est la seule chose qui distingue une cristallisation d'une sur-confiance.

- **Coût —** FLOPs passes arrière incluses, latence p50 et p95 mesurées avec synchronisation CUDA explicite, profondeur moyenne de sortie, fraction $\rho$ d'images adaptées.

- **Sécurité clinique —** courbe risque-couverture et AURC sur les cas renvoyés à l'expert.

- **Effondrement —** entropie de la distribution marginale des prédictions, par couche.

### 4.5 Traitement statistique

Cinq graines aléatoires minimum, moyenne et écart-type systématiques, intervalles de confiance par bootstrap apparié, test de DeLong pour les différences d'AUROC, correction de Holm-Bonferroni sur l'ensemble des comparaisons d'un tableau. Aucun hyperparamètre n'est sélectionné sur le jeu de test cible : les seuils, la température et les coefficients $\lambda$ et $\beta$ sont calibrés sur un split de validation issu du domaine source, ou d'un domaine cible explicitement tenu à l'écart et déclaré comme tel.

### 4.6 Tableau de résultats — à remplir par mesure

*Camelyon17-WILDS, split OOD officiel, 5 graines. Chaque cellule renseignée doit porter l'identifiant du run qui l'a produite.*

| **Réf** | **Méthode**             | **AUROC** | **ECE** | **FLOPs rel.** | **$N^{\ast}$ moyen** | **run id** |
|---------|-------------------------|-----------|---------|----------------|---------------|------------|
| A0      | Source-only             |           |         | 1,00           | 12,0          |            |
| A2      | TENT                    |           |         |                | 12,0          |            |
| A4      | Exit calibré            |           |         |                |               |            |
| A5      | TENT + exit calibré     |           |         |                |               |            |
| A8      | Cristallisation (nôtre) |           |         |                |               |            |

## 5. Explicabilité clinique et intégration

### 5.1 Transparence pour le radiologue

- **Visualisation de la trajectoire latente —** projection UMAP de la position du cas analysé par rapport aux centroïdes, aux couches 1, 6 et 12 : le passage de l'état désordonné à l'état ordonné est directement lisible.

- **Alerte d'incertitude —** un cas qui ne cristallise pas même après la douzième couche est automatiquement marqué pour révision prioritaire. La valeur clinique de ce mécanisme se mesure par la courbe risque-couverture, non par une affirmation qualitative.

- **Traçabilité —** la profondeur de sortie, l'énergie libre finale et l'indicateur de déclenchement de l'adaptation sont journalisés pour chaque examen.

### 5.2 Déploiement hospitalier

L'intégration visée est un service d'inférence en amont du PACS, consommant des séries DICOM et renvoyant une prédiction assortie de sa profondeur de sortie et de son drapeau d'incertitude.

> **Correction** La version 1 affirmait qu'« aucune donnée patient n'est sauvegardée ni transmise ». C'est exact pour les poids du modèle, mais **l'adaptation amortie sur un tampon glissant conserve de fait des représentations de patients antérieurs en mémoire vive** pendant la durée de vie du tampon. C'est parfaitement défendable, à condition d'être décrit : durée de rétention, portée, moment de réinitialisation. Un relecteur clinique relèvera l'omission. Le dépôt inclut un test qui vérifie l'absence de dérive entre échantillons successifs — c'est l'argument technique de cette section.

La revendication de latence de la version 1 est retirée : la latence réelle sera mesurée sur le matériel cible, passes arrière comprises, et rapportée en p50 et p95.

## 6. Registre des corrections, version 1 vers version 2

| **Réf** | **Défaut de la version 1**                                        | **Correction apportée**                                  |
|---------|-------------------------------------------------------------------|----------------------------------------------------------|
| F01     | Gradient nul sur le terme répulsif : la méthode était TENT        | Prototypes souples différentiables (§2.4)                |
| F02     | Circularité entre l'objectif d'adaptation et le critère de sortie | Découplage H / F fondé sur la Proposition 2 (§2.3)       |
| F03     | Entropie par couche incalculable, faute de têtes de lecture       | L têtes de projection et L jeux de centroïdes (§3.1)     |
| F04     | Budget de calcul physiquement incohérent                          | Adaptation sélective et comptabilité explicite (§3.4)    |
| F05     | Deux définitions incompatibles de l'entropie                      | Définition unique par mesure de Gibbs (§2.2)             |
| F06     | Seuil de 0,45 nats arbitraire, non normalisé, constant            | Calibration conforme, seuil devenu une sortie (§3.3)     |
| F07     | Effondrement et cristallisation indiscernables                    | Proposition 1 et diagnostic d'entropie marginale (§2.4)  |
| F08     | Camelyon16 hors budget matériel                                   | Camelyon17-WILDS (§4.2)                                  |
| F09     | Baselines limitées à 2021-2022                                    | EATA, SAR, DeYO et le contrôle A5 (§4.3)                 |
| F10     | Décalage simulé par bruit gaussien seul                           | Décalage réel en tête, synthétique pour l'analyse (§4.2) |
| F11     | Confusion entre $\mathbb{R}^{512}$ et la largeur native 768                    | Dimension d explicitée et ablatée (§2.1)                 |
| F12     | Argument de conformité RGPD incomplet                             | Rétention du tampon décrite et testée (§5.2)             |
| —       | Fraction adaptable annoncée à 0,75 %                              | Mesure : 0,044 % à 0,105 % selon configuration (§2.1)    |
| —       | Tableau de performances non mesuré                                | Supprimé ; tableau vide à remplir par mesure (§4.6)      |

*Documents associés : claude/01-cadre-theorique.md (formalisme complet et démonstrations), claude/02-protocole-experimental.md (protocole détaillé), claude/03-guide-implementation.md (mise en œuvre), claude/audit-et-plan-action.md (diagnostic initial et jalons).*
