---
title: "Note d'avancement et demande d'orientation (version antérieure)"
statut: "superseded"
date: 2026-08-30
auteur: "Houssem Eddine Lassoued"
math: latex
---

# Note d'avancement et demande d'orientation

*Thermodynamic Latent Space Crystallization*

|       |                                                                                                               |
|-------|---------------------------------------------------------------------------------------------------------------|
| De    | Houssem Eddine Lassoued                                                                                       |
| À     | Dr. Anis Ben Aicha · Pr. Habib Fathallah                                                                      |
| Date  | 30 août 2026                                                                                                  |
| Objet | Révision du cadre théorique, état d'avancement, et huit questions sur lesquelles je sollicite votre arbitrage |

## 1. En une page

Le projet propose de traiter l'espace latent d'un Vision Transformer comme un système physique doté d'une mesure de Gibbs, afin d'adapter le modèle au moment du test et d'interrompre l'inférence dès que la décision est suffisamment ordonnée. L'objectif clinique est de restaurer la performance diagnostique lorsque les images proviennent d'un scanner, d'un protocole ou d'un laboratoire non vus à l'entraînement.

J'ai conduit fin août une revue critique complète du cadre théorique et du protocole. **Elle a mis au jour trois défauts bloquants** : le terme de répulsion inter-classes avait un gradient nul, ce qui rendait la méthode mathématiquement identique à TENT (Wang et al., 2021) ; le critère de sortie anticipée était la quantité même que l'adaptation minimisait, ce qui garantissait mécaniquement l'économie de calcul même en cas d'erreur ; et l'entropie par couche n'était pas calculable, faute de têtes de lecture intermédiaires.

Elle a également établi que **le tableau de performances figurant dans la description du projet n'avait jamais été produit par une expérience**. Je détaille ce point en section 4.3, car il me paraît devoir vous être signalé explicitement plutôt que corrigé en silence.

Les trois défauts sont désormais corrigés, et les corrections produisent au passage deux propositions démontrées qui constituent, à mon sens, une contribution plus solide que la formulation initiale. L'ensemble de la documentation a été réécrit, et un plan d'implémentation en treize sprints est prêt à être exécuté.

**Ce que je vous demande.** Huit questions figurent en section 6. Elles portent sur des décisions qui ne m'appartiennent pas : validation du formalisme révisé, acceptation préalable du critère d'arrêt du projet, cible de publication, horizon institutionnel, et moyens. Une réponse même brève, question par question, me suffirait pour engager les six prochaines semaines sans risque de repartir en arrière.

## 2. Le projet, en termes simples

### 2.1 Le problème clinique

Un modèle de vision entraîné sur les images d'un hôpital fonctionne remarquablement bien sur ces images, et se dégrade dès qu'on lui présente celles d'un autre établissement. La cause n'est pas médicale mais technique : constructeur de scanner différent, dose de rayonnement, produit de contraste, protocole de coloration histologique, bruit d'acquisition. Ce phénomène — le décalage de distribution — est aujourd'hui le principal obstacle au déploiement clinique réel de ces modèles.

Face à une image issue d'un appareil inconnu, le réseau projette celle-ci dans son espace de représentation avec des poids figés. La projection perd sa structure : les représentations se dispersent au lieu de se regrouper autour des repères de décision, et la fiabilité du diagnostic chute.

### 2.2 L'idée : faire cristalliser la représentation

L'idée du projet est de cesser de considérer l'inférence comme une passe figée, et de la voir comme la recherche d'un état d'équilibre. On munit l'espace de représentation d'une mesure issue de la physique statistique, et l'on distingue deux régimes :

- **État désordonné —** l'image ne se rapproche d'aucun repère de décision de manière nette ; la représentation est diffuse, l'incertitude élevée.

- **État ordonné —** la représentation s'est condensée autour d'un repère identifié ; la décision est prise.

Le réseau dispose alors de deux leviers, activés au moment même du test, sans étiquette et sans réentraînement :

- **Il réajuste une fraction infime de ses paramètres —** les coefficients d'échelle et de décalage des couches de normalisation, plus quelques jetons de guidage — de l'ordre de 0,05 % du modèle, le reste demeurant strictement gelé.

- **Il adapte sa profondeur d'analyse —** dès que l'état est suffisamment ordonné, l'inférence s'arrête, les blocs suivants ne sont pas exécutés.

Un point conditionne toute la cohérence de l'analogie, et il est devenu le cœur théorique du travail : **un cristal exige la condensation et le maintien de sites distincts.** Une condensation vers un site unique n'est pas un cristal — c'est un effondrement. Or c'est précisément ce vers quoi converge, si l'on n'y prend garde, toute minimisation d'entropie non régularisée.

### 2.3 Les trois phases

| **Phase**      | **Moment**           | **Ce qui s'y passe**                                                                                                                                                                |
|----------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 — Ancrage    | Hors ligne, une fois | Le backbone étant gelé, on entraîne une tête de projection par couche, puis on fige un jeu de repères de décision par couche, sur la cohorte d'entraînement.                        |
| 2 — Adaptation | En ligne, sélective  | Si l'image est détectée hors du domaine de validité, le réseau effectue une à trois itérations de descente de gradient sur ses seuls paramètres adaptables. Sinon, il ne fait rien. |
| 3 — Sortie     | En ligne, par couche | À chaque couche, un critère est évalué ; dès qu'il passe sous un seuil calibré, l'inférence s'arrête. Un cas qui n'atteint jamais ce seuil est signalé pour relecture humaine.      |

## 3. La formalisation retenue

### 3.1 Une mesure de Gibbs, et trois observables au lieu d'une

Plutôt qu'un softmax de logits appris, le postérieur de classe est défini comme une distribution de Boltzmann sur les distances aux repères ancrés, d'énergie de configuration $E_k(z) = \lVert z - \mu_k \rVert^{2}$ :

$$
p_k(z) \;=\; \frac{\exp\!\big(-E_k(z)/T\big)}{Z(z)}\,,\qquad Z(z) \;=\; \sum_{j=1}^{K} \exp\!\big(-E_j(z)/T\big)
$$

La température T devient un paramètre réel et calibrable, et trois observables distinctes en découlent : l'entropie de Gibbs H, mesure du désordre de l'assignation ; l'énergie libre $F = -T \ln Z$ ; et l'énergie moyenne ⟨E⟩. Elles sont liées par l'identité thermodynamique $F = \langle E \rangle - T\,\mathcal{H}$, vérifiée dans le code par un test unitaire.

Cette reformulation n'est pas un raffinement esthétique : c'est elle qui fournit les deux quantités distinctes dont la correction des défauts avait besoin. On notera d'ailleurs que F coïncide avec le score énergétique de Liu et al. (2020), détecteur hors-distribution établi — le cadre n'invente donc pas un critère, il en retrouve un par une voie thermodynamique.

### 3.2 Les deux propositions

**Proposition 1 — le terme de répulsion exclut l'effondrement.** Toute configuration où tous les échantillons sont assignés à un même repère atteint H = 0, borne inférieure de l'entropie : c'est donc un minimiseur global du terme de condensation seul. Sur de telles configurations, les prototypes se confondent et le terme répulsif s'annule. Une configuration séparée, elle, atteint une énergie strictement négative. Par conséquent, pour tout coefficient de répulsion strictement positif, aucune configuration effondrée n'est un minimiseur global de l'énergie en ligne.

*C'est l'énoncé qui distingue formellement la méthode de TENT, et qui donne au terme de répulsion une fonction — dans la formulation initiale, il n'en avait aucune.*

**Proposition 2 — l'entropie ne peut pas servir de critère de sortie.** Sous une translation uniforme du paysage énergétique, l'entropie est invariante tandis que l'énergie libre est covariante. Il s'ensuit qu'une image équidistante mais très éloignée de tous les repères — la signature typique d'un échantillon hors-domaine — présente une entropie basse et une énergie libre élevée. L'entropie ne distingue donc pas une décision confiante d'une décision confiante mais formulée hors du domaine de validité du modèle. L'énergie libre le fait.

*C'est l'énoncé qui fonde le découplage : on adapte sur l'entropie, on sort sur l'énergie libre.*

### 3.3 Une garantie plutôt qu'un seuil

Les seuils de sortie ne sont plus choisis mais calibrés, par contrôle conforme du risque sur un jeu dédié. La garantie prend la forme : la dégradation d'exactitude imputable à la sortie anticipée reste inférieure à un niveau fixé, avec une confiance donnée. C'est, à mon sens, ce qui fait passer le travail d'une contribution d'efficience à une contribution d'IA médicale déployable.

> **Réserve** Cette garantie suppose l'échangeabilité entre le jeu de calibration et le jeu de test — hypothèse violée par construction sous décalage de distribution. Le manuscrit devra rapporter séparément les trois régimes de calibration et discuter honnêtement cette limite. Je préfère l'énoncer nous-mêmes plutôt que de la laisser à un relecteur.

## 4. État actuel : ce qui a été révisé

### 4.1 Les trois défauts bloquants

**Le terme de répulsion était inerte.** Les repères de décision étaient décrits comme « ancrés de manière immuable ». S'ils sont constants, le gradient de la distance qui les sépare est nul par rapport aux paramètres, et minimiser « entropie moins répulsion » revient exactement à minimiser l'entropie. La méthode décrite était donc indiscernable de TENT, et la contribution thermodynamique se réduisait à un habillage lexical. La correction consiste à ré-estimer des prototypes souples sur le flux de test, dont le gradient n'est pas nul.

**L'adaptation et la sortie mesuraient la même chose.** On adaptait le modèle pour faire baisser l'entropie, puis on sortait dès que l'entropie passait sous un seuil. L'économie de calcul était donc garantie par construction, y compris lorsque le modèle se trompait : elle mesurait une sur-confiance induite, non une meilleure compréhension. C'est le type de circularité qu'un relecteur de TMI identifie en première lecture. La Proposition 2 fonde le découplage qui l'élimine.

**L'entropie par couche n'était pas calculable.** Les repères étaient ancrés dans l'espace de la douzième couche, alors que les représentations intermédiaires vivent dans une géométrie différente : comparer un embedding de la couche 5 à un repère de la couche 12 n'a pas de sens métrique. Il faut douze têtes de projection et douze jeux de repères. Le coût est marginal, mais l'omission rendait toute la troisième phase inopérante.

### 4.2 Un défaut de budget

Une passe arrière coûte environ le double d'une passe avant, et doit traverser tout le réseau même lorsque seuls 0,05 % des paramètres sont mis à jour. Une adaptation systématique à chaque image ne peut donc pas rendre l'inférence plus rapide. La correction introduit une adaptation sélective : elle n'est déclenchée que sur les cas détectés hors-domaine, tronquée à la couche sonde, et amortie sur un tampon glissant. La condition de gain net s'écrit alors simplement — moins d'un tiers des images doivent être adaptées — et devient une prédiction falsifiable.

### 4.3 Un point que je tiens à vous signaler explicitement

La description initiale du projet comportait un tableau intitulé « Synthèse des Performances », annonçant 87,3 % d'exactitude hors-domaine, 42,5 % de FLOPs économisés, une profondeur moyenne de 5,4 couches et 14,2 ms par image. **Aucune de ces valeurs n'a été produite par une expérience.** Elles avaient été formulées comme objectifs lors de la conception, puis se sont figées en tableau de résultats au fil des versions successives du document.

La vérification interne montre en outre qu'elles sont mutuellement incohérentes. Le tableau attribuait 24,6 ms à un ViT-B statique et 28,1 ms à TENT ; or TENT effectue une passe avant et une passe arrière, soit un coût d'environ 70 ms sur cette base. L'écart est d'un facteur voisin de trois, et de dix pour la ligne de la méthode proposée.

Ces valeurs ont été retirées de l'ensemble de la documentation. Je vous le signale parce qu'une correction silencieuse me semblerait la mauvaise manière de traiter ce genre d'erreur, et parce que le mécanisme qui l'a produite mérite d'être nommé : **le plan de rédaction initial prévoyait d'écrire le résumé et l'introduction avant toute expérience.** Cet ordre est désormais inversé — méthodes, expériences, discussion, introduction, résumé en dernier.

### 4.4 Registre des corrections

| **Défaut initial**                                        | **Correction**                                                  |
|-----------------------------------------------------------|-----------------------------------------------------------------|
| Gradient nul sur le terme de répulsion                    | Prototypes souples différentiables ; Proposition 1              |
| Circularité entre adaptation et sortie                    | Découplage entropie / énergie libre ; Proposition 2             |
| Entropie par couche incalculable                          | Douze têtes de projection et douze jeux de repères              |
| Budget de calcul incohérent                               | Adaptation sélective ; comptabilité incluant les passes arrière |
| Deux définitions incompatibles de l'entropie              | Définition unique par la mesure de Gibbs                        |
| Seuil de 0,45 nats arbitraire et non normalisé            | Calibration conforme ; le seuil devient une sortie              |
| Effondrement et cristallisation indiscernables            | Proposition 1 et diagnostic d'entropie marginale                |
| Camelyon16 hors budget matériel                           | Camelyon17-WILDS, décalage réel inter-hôpitaux                  |
| Baselines limitées à 2021-2022                            | EATA, SAR, DeYO, et le contrôle décisif TENT + sortie calibrée  |
| Fraction adaptable annoncée à 0,75 %                      | Comptage exact : 0,044 % à 0,105 % selon la configuration       |
| Confusion entre la dimension 512 et la largeur native 768 | Dimension latente explicitée et ablatée                         |
| Tableau de performances non mesuré                        | Retiré ; tableau vide à remplir par mesure                      |

## 5. Ce qui est prêt, et ce qui vient

La documentation du projet a été intégralement réécrite : description complète, plan de recherche, guide de la base de connaissances, cadre théorique avec démonstrations, protocole expérimental, et guide d'implémentation. Le plan d'implémentation est découpé en treize sprints, chacun assorti d'un test d'acceptation.

| **Étape**      | **Contenu**                                            | **Durée**  | **Aboutissement**                                            |
|----------------|--------------------------------------------------------|------------|--------------------------------------------------------------|
| Sprints 0 à 2  | Environnement, éditeur, squelette du dépôt             | 3 h        | Dépôt exécutable                                             |
| Sprints 3 à 5  | Noyau formel, backbone instrumenté, gel des paramètres | 6 h        | Identité thermodynamique et Proposition 2 vérifiées par test |
| Sprints 6 à 8  | Ancrage hors ligne, baselines, énergie en ligne        | 4 j        | Proposition 1 vérifiée expérimentalement                     |
| Sprints 9 à 11 | Sortie anticipée, calibration, évaluation, données     | 2,5 j      | Chaîne complète mesurable                                    |
| Campagnes      | MedMNIST-C, puis Camelyon17-WILDS                      | 2 à 4 sem. | Jalons J1 à J4                                               |

**Le premier résultat mesuré du projet** est la courbe d'entropie en fonction de la profondeur, attendue au sprint 6, soit une dizaine de jours de travail. Elle teste déjà une conjecture : rien n'impose a priori que l'entropie décroisse avec la profondeur, et si elle ne décroît pas, c'est une réfutation à consigner, pas un bogue à corriger.

### 5.1 Les critères d'arrêt

Le plan comporte des jalons assortis de conditions d'échec explicites et de bifurcations prévues. Le plus important est le second :

| **Jalon**         | **Critère de passage**                                                                                 | **Bifurcation en cas d'échec**                                                                |
|-------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| J1 — mécanique    | Entropie décroissante, aucune divergence, gel du backbone vérifié                                      | Problème d'implémentation : on corrige                                                        |
| J2 — contribution | La méthode complète dépasse significativement la composition TENT + sortie calibrée, sur décalage réel | Repli sur la sortie anticipée à risque contrôlé sans adaptation — publiable en MICCAI ou MIDL |
| J3 — budget       | Coût net inférieur à l'inférence statique, passes arrière incluses                                     | Repositionnement en méthode de robustesse, et publication du comptage honnête                 |
| J4 — physique     | Pic de susceptibilité reproductible, s'affinant avec la dimension latente                              | Abandon du vocabulaire thermodynamique au profit d'« energy-based »                           |

> **Pourquoi ces critères sont écrits d'avance** Décider à l'avance ce qui constituerait un échec est ce qui distingue une recherche d'une illustration. C'est aussi, très concrètement, ce qui me protégera dans six semaines : face à un résultat décevant après des nuits de calcul, la tentation de « trouver » le bon chiffre est un mécanisme structurel, pas un défaut de caractère. Un critère validé par vous en amont la neutralise.

### 5.2 Les garde-fous mis en place

- **Registre des runs —** chaque valeur publiée porte l'identifiant du run qui l'a produite, avec sa configuration et le hachage du commit. Un chiffre absent d'un fichier de métriques n'existe pas.

- **Tests non négociables —** le gel effectif du backbone et l'absence de dérive entre échantillons successifs sont vérifiés par assertion, et non affirmés. Le second est aussi l'argument technique de la conformité RGPD.

- **Critère pré-enregistré —** le test de transition de phase et sa condition d'abandon sont fixés avant l'observation des courbes.

- **Tableaux créés vides —** aucune valeur n'est écrite avant d'être mesurée, dans aucun document.

## 6. Huit questions sur lesquelles je sollicite votre avis

*Une réponse brève, question par question, me suffirait. Les questions A relèvent de la validation scientifique, les questions B de la trajectoire du travail, les questions C des moyens.*

### A. Validation scientifique

#### Q1 — Validez-vous la reformulation par mesure de Gibbs, et les démonstrations des Propositions 1 et 2 ?

> Le formalisme complet, avec les démonstrations, figure dans le document « Cadre théorique ». La Proposition 1 est l'énoncé sur lequel repose la distinction entre la méthode et TENT ; la Proposition 2 fonde le découplage entre adaptation et sortie. Ce sont les deux points où une faiblesse de démonstration serait fatale.
>
> **Ce qui en dépend** Si les démonstrations tiennent, elles deviennent le cœur de la section théorique du manuscrit. Sinon, il faut les renforcer avant d'écrire la moindre ligne de code, car toute l'architecture logicielle en découle.

#### Q2 — Comment jugez-vous la marge de nouveauté face à la littérature 2025-2026 ?

> Une littérature active traite depuis 2025 de l'effondrement neuronal en adaptation au moment du test, dont un article accepté à CVPR 2026. Elle est conceptuellement proche de la notion de cristallisation, sans toutefois proposer de garantie sur la sortie anticipée. Mon appréciation est que la marge se situe dans la combinaison — énergie libre comme critère de sortie, plus contrôle conforme du risque — plutôt que dans la condensation seule.
>
> **Ce qui en dépend** Le positionnement du manuscrit, et la formulation de la contribution revendiquée dans le résumé. Une erreur d'appréciation ici coûte un refus au motif d'antériorité.

### B. Trajectoire

#### Q3 — Acceptez-vous par avance le critère d'arrêt du jalon J2 et sa bifurcation ?

> Si la méthode complète ne dépasse pas significativement la composition « TENT + sortie anticipée calibrée » sur décalage réel, le plan prévoit un repli sur la sortie anticipée à risque contrôlé, sans adaptation — contribution plus étroite, mais publiable. Je souhaiterais que cette bifurcation soit validée maintenant, et non discutée au moment où elle se présenterait.
>
> **Ce qui en dépend** Tout. C'est la question à laquelle je tiens le plus. Un critère d'échec accepté en amont transforme un résultat décevant en information ; un critère discuté après coup le transforme en pression.

#### Q4 — Quelle cible de publication, et dans quel ordre ?

> Mon analyse est que Nature Machine Intelligence est peu réaliste à ce stade : cette revue publie soit des paradigmes nouveaux étayés par des preuves exceptionnelles, soit des travaux validés cliniquement sur cohortes prospectives multi-sites. Je proposerais IEEE TMI ou Medical Image Analysis comme cible principale, avec éventuellement MICCAI, MIDL ou ISBI comme jalon intermédiaire une fois J2 et J3 franchis.
>
> **Ce qui en dépend** Le calendrier, le périmètre expérimental — un journal exige plus de cohortes qu'une conférence — et le format de rédaction. Votre lecture de l'opportunité stratégique prime sur la mienne.

#### Q5 — Quel est l'horizon réel et le cadre institutionnel du travail ?

> Le plan initial mentionnait une fenêtre de juin à août 2026, désormais close. Je ne sais pas si ce travail doit aboutir à un mémoire, à une première publication dans un parcours doctoral, ou à un article autonome.
>
> **Ce qui en dépend** Le dimensionnement du protocole : deux cohortes ou quatre, ViT-S seul ou confirmation sur ViT-B, ablations complètes ou réduites. Je peux calibrer l'effort dès que je connais l'échéance.

### C. Moyens

#### Q6 — Disposons-nous d'un accès à une cohorte hospitalière réelle ?

> Le protocole actuel repose sur des jeux publics — Camelyon17-WILDS pour le décalage inter-hôpitaux, RSNA et ChestX-ray14 pour la radiographie. Une cohorte interne, même modeste, provenant d'un partenariat avec un service d'imagerie ou d'anatomopathologie, changerait la nature du travail.
>
> **Ce qui en dépend** La solidité de la démonstration clinique, et l'accessibilité de revues plus exigeantes. C'est le seul élément qui pourrait rendre Nature MI envisageable à moyen terme.

#### Q7 — Une allocation de calcul est-elle disponible au laboratoire ?

> Je travaille sur une RTX 4070 de 12 Go. C'est suffisant pour ViT-S/16 et pour un Camelyon17 sous-échantillonné, mais l'entraînement de ViT-B/16 sur le jeu complet demanderait entre 24 et 36 heures, et la matrice d'ablation complète sur cinq graines, trois à cinq jours de calcul continu.
>
> **Ce qui en dépend** Le choix du modèle principal et l'étendue des ablations. Le protocole est actuellement calibré pour tenir sur ce seul GPU ; un accès à une machine de calcul permettrait de le desserrer.

#### Q8 — Un co-auteur clinicien est-il envisageable ?

> Le travail comporte des affirmations cliniques — pertinence du renvoi à l'expert, seuils de sensibilité, intégration dans le flux PACS — qu'un relecteur de TMI attend de voir portées par un praticien. Un radiologue ou un anatomopathologiste associé au projet renforcerait considérablement la section clinique.
>
> **Ce qui en dépend** La crédibilité de la partie applicative, et le choix des métriques cliniquement pertinentes — le seuil de spécificité retenu, notamment, devrait être fixé par un praticien plutôt que par convention.

## 7. En conclusion

La révision a été plus profonde que je ne l'anticipais, et je crois qu'elle laisse le projet en meilleur état qu'avant : les défauts corrigés ont produit deux propositions démontrables là où la formulation initiale ne contenait qu'une analogie, et le protocole est désormais assorti de critères qui permettront de savoir, en six semaines plutôt qu'en six mois, si la contribution existe.

Je suis prêt à engager l'implémentation dès réception de vos retours, et je reste évidemment disponible pour présenter le cadre révisé de vive voix si vous le jugez utile.

*Documents joints : Description Complète du Projet (v2), Plan de Recherche (v2), Guide des Connaissances (v2). Documents de travail disponibles : cadre théorique et démonstrations, protocole expérimental détaillé, guide d'implémentation en treize sprints.*
