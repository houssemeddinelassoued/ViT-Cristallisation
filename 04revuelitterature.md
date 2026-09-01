---
title: "Revue de littérature et analyse d'antériorité"
version: 1
date: 2026-08-31
auteur: "Houssem Eddine Lassoued"
critere_qualite: "Journaux Q1–Q2 (JCR) et conférences CORE A/A* uniquement"
math: latex
---

# Revue de littérature et analyse d'antériorité

## 0. Méthode et critère de qualité

Seules figurent en sections 2 à 5 des publications parues dans des **journaux Q1–Q2** ou des **conférences CORE A/A\*** : NeurIPS, ICML, ICLR, CVPR, ICCV, ACM MM (A\*), MICCAI (A) ; *Annals of Statistics*, *Annals of Applied Statistics*, IJCV, IEEE TMI, *Medical Image Analysis*, *Scientific Data* (Q1).

Les préprints non relus sont relégués en §7. **Ils ne doivent jamais être cités comme établis** — mais deux d'entre eux occupent votre créneau et doivent être surveillés.

> **Réserve de transparence.** Les citations ont été établies par recherche documentaire avec lecture des sources primaires. Les deux affirmations les plus lourdes — le vocabulaire exact de Liu et al. 2020 et la formule de ProPos — méritent une vérification de votre part sur les PDF avant d'être écrites dans le manuscrit. Ma vérification directe a été interrompue par une limite technique.

---

## 1. Les trois résultats, en une page

**① La lecture thermodynamique de CLIP n'est pas assez neuve pour porter le papier.**
Liu et al. (NeurIPS 2020) écrivent déjà *Helmholtz free energy* pour $F = -T\log\sum_i e^{f_i(x)/T}$. CIDER (ICLR 2023) dérive déjà l'équivalence von Mises-Fisher ↔ softmax sur l'hypersphère avec $\kappa = 1/\tau$. L'identité $\lVert z-t\rVert^2 = 2-2\langle z,t\rangle$ est de l'algèbre élémentaire. **Le facteur $T = 2\tau$ ne doit jamais être présenté comme un résultat**, seulement comme une convention de normalisation.

**② Le terme de répulsion est une variante d'un mécanisme publié — et il comporte deux défauts.**
L'estimateur de prototype souple est l'équation (4) de SHOT (ICML 2020). La répulsion entre prototypes souples est la *Prototype Scattering Loss* de ProPos. Plus grave : le terme **n'empêche pas l'effondrement marginal** et il est **non borné**. Détail et correctifs en §3.

**③ La bonne nouvelle : le triplet TTA + sortie anticipée + contrôle du risque est libre.**
Les trois briques n'existent que par paires. C'est là que doit se loger la contribution, et le créneau est réel — les auteurs de *Fast yet Safe* (NeurIPS 2024) désignent nommément cette extension comme travail futur.

---

## 2. Antériorité ① — la formulation énergétique

| Référence | Venue | Rang | Ce qui y est déjà |
|---|---|---|---|
| Liu, Wang, Owens, Li — *Energy-based Out-of-distribution Detection* | NeurIPS 2020 | A\* | Distribution de Gibbs, fonction de partition $Z$, et **« Helmholtz free energy »** $E(x) = -T\log\sum_i e^{f_i(x)/T}$ nommée telle quelle |
| Grathwohl et al. — *Your Classifier is Secretly an Energy Based Model* | ICLR 2020 | A\* | Lecture énergétique d'un classifieur softmax |
| Ming, Cai, Gu, Sun, Li — *Delving into OOD Detection with Vision-Language Representations* (MCM) | NeurIPS 2022 | A\* | Score $\max_i e^{s_i/\tau}/\sum_j e^{s_j/\tau}$ sur similarités cosinus CLIP — **un postérieur de Boltzmann de fait**. Compare déjà au score énergétique de Liu appliqué à CLIP |
| Ming, Sun, Dia, Li — *How to Exploit Hyperspherical Embeddings for OOD Detection?* (CIDER) | ICLR 2023 | A\* | Densité von Mises-Fisher sur $\mathbb{S}^{d-1}$, **dérivation explicite $\kappa = 1/\tau$** — l'équivalence, au facteur 2 près |
| Jiang, Liu, Fang, Chen, Liu, Zheng, Han — *NegLabel* | ICLR 2024 (Spotlight) | A\* | Score structuré comme rapport de deux fonctions de partition, jamais nommé ainsi |
| Wang, Li, Yao, Li — *CLIPN for Zero-Shot OOD Detection* | ICCV 2023 | A\* | Encodeur textuel de négation ; aucun cadre énergétique |
| Miyai et al. — *GL-MCM* | IJCV 2025 | Q1 | Extension locale de MCM ; aucune lecture physique |
| Yuan, Xu, Hou, Sun, Shen, Cheng — *TEA: Test-time Energy Adaptation* | CVPR 2024 | A\* | Emploie explicitement « distribution de Boltzmann » et « fonction de partition » **en TTA** — mais sur classifieurs standards, sans CLIP |

**Verdict.** Chacune des briques est publiée par des équipes de premier plan ; l'énoncé unificateur ne l'est pas. Deux signaux à connaître : les deux revues de référence sur la détection OOD à l'ère des modèles vision-langage ne contiennent **aucune** occurrence de « free energy », « Gibbs » ou « partition function » appliquée à CLIP — c'est le signal le plus favorable. Mais MCM **re-règle $\tau$ librement** ($\tau = 1$, et non le $\tau \approx 0{,}01$ appris), ce qui affaiblit empiriquement la thèse « $\tau$ appris *est* la température physique du système ».

**Repositionnement obligatoire.** Écrire « unification et opérationnalisation d'observables déjà employées séparément », jamais « nouvelle formulation ». Ce que vous pouvez revendiquer : **$F$ mesurée couche par couche comme critère d'arrêt**, et non $F$ comme score.

---

## 3. Antériorité ② — le terme de répulsion, et ses deux défauts

| Référence | Venue | Rang | Formule et rapport à votre terme |
|---|---|---|---|
| Liang, Hu, Feng — *SHOT* | ICML 2020 | A\* | $\mathcal{L}_{\text{div}} = \sum_k \hat p_k\log\hat p_k$ (diversité, dans le simplexe) **et**, en Éq. (4), l'estimateur $c_k = \frac{\sum_x \delta_k(x)\,g(x)}{\sum_x \delta_k(x)}$ — **littéralement votre prototype souple** |
| Gomes, Krause et al. — *RIM* | NeurIPS 2010 | A\* | Origine de la maximisation d'information $I(x;y) = H(\bar\sigma) - \bar H(\sigma)$ |
| Wang, Shelhamer, Liu, Olshausen, Darrell — *TENT* | ICLR 2021 | A\* | Entropie pure, **aucun régularisateur** — c'est pourquoi il s'effondre |
| Niu et al. — *EATA* | ICML 2022 | A\* | La « diversité » y est un **filtre de redondance**, pas un terme additif. Le seul terme ajouté est de Fisher (anti-oubli) |
| Niu et al. — *SAR* | ICLR 2023 | A\* | Minimisation avec perturbation de netteté ; documente l'effondrement ; **ni diversité, ni prototype** |
| Zhao et al. — *DELTA* | ICLR 2023 | A\* | Re-pondération dynamique en ligne ; pas de prototype |
| Wang et al. — *TSD: Feature Alignment and Uniformity for TTA* | CVPR 2023 | A\* | Prototypes durs, auto-distillation ; pas de terme d'uniformité explicite |
| Bardes, Ponce, LeCun — *VICReg* | ICLR 2022 | A\* | Terme de variance à charnière **borné** — précédent méthodologique du correctif |
| Zbontar, Jing, Misra, LeCun, Deny — *Barlow Twins* | ICML 2021 | A\* | Décorrélation |
| Caron et al. — *SwAV* | NeurIPS 2020 | A\* | Équipartition par Sinkhorn |
| Caron et al. — *DINO* | ICCV 2021 | A\* | Centrage + sharpening |
| Lee et al. — *DeYO / Entropy is not Enough* | ICLR 2024 | A\* | Critère de fiabilité au-delà de l'entropie |

### Deux défauts techniques du terme, à corriger avant soumission

**(a) Le terme n'empêche pas l'effondrement marginal.** La normalisation par $\sum_i p_k(z_i)$ rend $\hat\mu_k$ invariant à l'échelle de la marginale $\hat\pi_k$. Contre-exemple : si $p_1(z_i) = \varepsilon\alpha_i$ avec $\varepsilon \to 0$ et $\alpha_i$ non constants, alors $\hat\pi_1 \to 0$ — effondrement total des prédictions vers la classe 0 — tandis que $\lVert\hat\mu_0 - \hat\mu_1\rVert^2$ **reste grand**. Seul l'effondrement *uniforme* est pénalisé. Le terme de diversité de SHOT, lui, interdit les deux.

**(b) Le terme est non borné.** $\lVert\hat\mu_0-\hat\mu_1\rVert^2$ n'a pas de borne supérieure. Comme $\theta_{\text{adapt}}$ contient les $\gamma$ des LayerNorm, le réseau peut minimiser $\mathcal{E}$ **en gonflant simplement l'échelle des représentations** ($\gamma \to c\gamma$ donne $D \to c^2 D$), sans améliorer aucune séparation angulaire. C'est exactement pourquoi VICReg emploie une charnière bornée.

> **Le second défaut est le plus grave, parce qu'il est silencieux.** Il produirait des courbes parfaitement plausibles et une répulsion en croissance, sans aucune amélioration réelle. La simulation embarquée dans le site ne le révèle pas : elle contraint les représentations sur le cercle unité, ce qui borne le terme par construction. **Sur le vrai modèle, cette borne n'existe pas.**

### Correctifs possibles

| Correctif | Effet | Coût |
|---|---|---|
| Normaliser $z$ avant le calcul des prototypes | Borne le terme dans $[0,4]$ | Aucun — c'est déjà le cas avec CLIP |
| Multiplier par $\hat\pi_0\hat\pi_1$ | Rétablit la pression anti-effondrement marginale, et donne exactement $\mathrm{tr}\,S_B$ souple | Rapproche du régularisateur de diversité, donc affaiblit la nouveauté |
| Utiliser le ratio de Fisher $D_{\text{inter}}/\mathrm{tr}\,S_W$ | Borné et invariant d'échelle | Un terme de plus à estimer |

**Verdict.** Ne présentez pas ce terme comme une nouveauté. Présentez-le comme la spécialisation binaire, à coût $O(Bd)$, d'un mécanisme connu — et **ajoutez SHOT-IM à la matrice d'ablation** : c'est le concurrent direct, et son absence serait relevée.

---

## 4. Le créneau libre — TTA + sortie anticipée + contrôle du risque

| Référence | Venue | Rang | Ce qu'elle couvre, et ce qui manque |
|---|---|---|---|
| Schuster et al. — *Confident Adaptive Language Modeling* (CALM) | NeurIPS 2022 (Oral) | A\* | Sortie anticipée + Learn-then-Test. **Garantit la fidélité au modèle complet, jamais la correction.** Pas de TTA |
| Jazbec, Timans, Hadži Veljković, Sakmann, Zhang, Naesseth, Nalisnick — *Fast yet Safe: Early-Exiting with Risk Control* | NeurIPS 2024 | A\* | **Votre référentiel obligatoire.** Sortie anticipée + contrôle conforme du risque, i.i.d. strict, sans TTA. Les auteurs désignent explicitement la relaxation i.i.d. vers le décalage au moment du test comme travail futur |
| Jia, Kwon, Orsino, Dang, Talia, Mascolo — *TinyTTA* | NeurIPS 2024 | A\* | **Le seul travail publié combinant TTA et sortie anticipée.** Critère d'arrêt entropique, exactement celui de votre v1. **Aucune garantie**, et la circularité n'est jamais discutée |
| Angelopoulos, Bates, Candès, Jordan, Lei — *Learn then Test* | Ann. Appl. Stat. 2025 | Q1 | Fondation du contrôle de risque par tests multiples |
| Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control* | ICLR 2024 | A\* | Théorème central, et section sur le décalage à ratio de densité **connu** |
| Tibshirani, Barber, Candès, Ramdas — *Conformal Prediction Under Covariate Shift* | NeurIPS 2019 | A\* | Échangeabilité pondérée ; les poids doivent être connus ou estimés |
| Barber, Candès, Ramdas, Tibshirani — *Conformal prediction beyond exchangeability* | Ann. Statist. 2023 | Q1 | Borne $1-\alpha-\sum_i\tilde w_i\,d_{TV}$ — **le terme de perte est non estimable en pratique** |
| Gibbs, Candès — *Adaptive Conformal Inference Under Distribution Shift* | NeurIPS 2021 | A\* | Couverture garantie **en moyenne temporelle**, pas instantanée |
| Bajpai, Hanawal — *Beyond Greedy Exits* | NeurIPS 2025 | A\* | Ajustement en ligne et non supervisé des seuils par bandits ; traite le décalage, sans conformal |
| Meng et al. — *AdaViT* | CVPR 2022 | A\* | Politique d'usage apprise ; aucune garantie |
| Yin et al. — *A-ViT* | CVPR 2022 | A\* | Arrêt adaptatif par token ; aucune garantie |
| Xu et al. — *LGViT* | ACM MM 2023 | A\* | Têtes hétérogènes ; seuil de confiance ; aucune garantie |
| Bajpai, Hanawal — *BEEM* | ICLR 2025 | A\* | Sorties traitées comme experts ; aucune garantie formelle |
| Geifman, El-Yaniv — *Selective Classification for DNNs* | NeurIPS 2017 | A\* | Fondation des courbes risque-couverture |
| Mehrtens, Bucher, Brinker — *Pitfalls of Conformal Predictions for Medical Image Classification* | UNSURE @ MICCAI 2023 | A | **À lire en priorité — voir §5** |

**Verdict.** Le triplet n'est pas publié. Trois démonstrations restent à faire pour se distinguer :

1. Un contrôle de risque valide alors que **la fonction de score est modifiée par échantillon** par la TTA — le prédicteur devient dépendant des données, ce qui casse l'hypothèse de score fixe de LTT et CRC. C'est la difficulté théorique centrale, et elle est réelle.
2. Une quantification explicite du terme de perte de couverture, par conformal pondéré ou par variante valide à tout instant si le déploiement est en flux.
3. Une comptabilité FLOPs où le coût des passes arrière est **soustrait** du gain annoncé.

Clarifiez aussi ce que « cohérence » signifie chez vous : fidélité au modèle gelé, ou au modèle adapté ? CALM garantit la première, jamais la correction.

---

## 5. Les cinq objections à préparer

**① La circularité est un défaut, pas une élégance.** SAR (ICLR 2023) documente que la minimisation d'entropie dérive vers des solutions triviales — inflation de la norme des logits, domination d'une classe. Un seuil $\mathcal{H}_n < \varepsilon$ appliqué après TTA mesure l'optimisation, pas la certitude. **Votre découplage $\mathcal{H}$ / $F$ répond précisément à cette objection : c'est votre meilleur argument, mettez-le en avant.**

**② Double violation d'échangeabilité.** Calibration source contre test hors-domaine, *et* modèle modifié par échantillon. Le terme de perte de couverture de Barber et al. (*Ann. Statist.* 2023) est non estimable en clinique.

**③ Biais de sélection conditionnel aux sorties.** La population atteignant la couche 8 n'est pas celle de la calibration : elle est conditionnée au non-déclenchement des sorties précédentes.

**④ Mehrtens et al. (MICCAI 2023) contredisent frontalement la Phase 3.** Quatre écueils, tous applicables : les prédictions conformes sont non fiables sous décalage **en dermatologie et histopathologie** — vos modalités ; elles **ne doivent pas servir à sélectionner des prédictions** ; elles ne sont pas fiables par sous-groupe — donc par constructeur de scanner ; et leur valeur est limitée quand le nombre de classes est faible — votre cas binaire. **Cet article doit être lu et cité, et l'objection traitée dans la discussion, pas contournée.**

**⑤ Couverture marginale ≠ conditionnelle.** Garantir un risque moyen ne protège aucun patient ni aucun site. Prévoyez une stratification de type Mondrian par appareil.

---

## 6. Conséquences sur les documents du projet

| Document | Modification requise |
|---|---|
| Cadre théorique | Reformuler la Proposition 1 : ce n'est pas la nouveauté du terme, mais son rôle dans le couplage avec l'arrêt. Ajouter la borne et le facteur $\hat\pi_0\hat\pi_1$. Retirer toute prétention de nouveauté sur $T = 2\tau$ |
| Protocole v3 | Ajouter **SHOT-IM** au groupe B des ablations. Ajouter **TinyTTA** comme comparaison directe. Ajouter une ablation « répulsion bornée contre non bornée » |
| Note aux encadrants | Q6 sur la marge de nouveauté reçoit une réponse documentée : la marge est dans le triplet, pas dans la formulation |
| Positionnement du papier | « Premier système combinant adaptation au moment du test et arrêt anticipé **à risque contrôlé**, avec un critère d'arrêt découplé de l'objectif d'adaptation » |

---

## 7. Préprints à surveiller — non citables comme établis

Deux occupent votre créneau et doivent être vérifiés avant toute soumission :

- **« Rethinking Entropy in Test-Time Adaptation: The Missing Piece from Energy Duality »** (NeurIPS 2025, OpenReview) — le titre annonce la dualité entropie ↔ énergie en TTA, c'est-à-dire l'articulation de votre Phase 2. **À lire en premier.**
- **« ΔEnergy »** (arXiv 2510.11296) — applique déjà $-\log\sum_j e^{s_j/\tau}$ aux similarités cosinus de CLIP.

Autres à surveiller : *SAR²/Adapt in the Wild* (régularisation géométrique anti-effondrement sur centroïdes en TTA), *ProPos* (Prototype Scattering Loss), *DPL* et *CPL-NC* (répulsion inter-prototypes en TTA), *NCTTA* (CVPR 2026, effondrement neuronal en TTA), *SAFE-KD* (sortie anticipée à risque contrôlé sur ViT), *EACP* (TTA + conformal, garanties explicitement abandonnées), *MedSeg-TTA* (la minimisation d'entropie dégrade les performances sous fort décalage inter-centre).

---

## 8. Ordre de lecture

1. **Fast yet Safe** (NeurIPS 2024) — votre référentiel direct
2. **TinyTTA** (NeurIPS 2024) — le seul TTA + sortie anticipée publié
3. **Mehrtens et al.** (MICCAI 2023) — l'objection clinique la plus dure
4. **SHOT** (ICML 2020) — l'origine du prototype souple et de la diversité
5. **CIDER** (ICLR 2023) — l'équivalence hypersphérique
6. **Liu et al.** (NeurIPS 2020) — l'énergie libre nommée
7. **SAR** (ICLR 2023) — la mécanique de l'effondrement
8. Le préprint NeurIPS 2025 « Energy Duality » — vérification d'antériorité
