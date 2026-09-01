---
title: "Note d'avancement et demande d'orientation"
subtitle: "Nouvel axe de recherche : cristallisation de l'espace latent"
date: 2026-08-30
auteur: "Houssem Eddine Lassoued"
destinataires: "Dr. Anis Ben Aicha, Pr. Habib Fathallah"
math: latex
---

# Note d'avancement et demande d'orientation

*Nouvel axe de recherche : cristallisation de l'espace latent pour l'adaptation au moment du test en imagerie médicale*

|       |                                                                                                                                   |
|-------|-----------------------------------------------------------------------------------------------------------------------------------|
| De    | Houssem Eddine Lassoued                                                                                                           |
| À     | Dr. Anis Ben Aicha · Pr. Habib Fathallah                                                                                          |
| Date  | 30 août 2026                                                                                                                      |
| Objet | Proposition d'un axe méthodologique pour la thèse, développement théorique, revue de littérature, et huit questions d'orientation |

## 1. En une page

Mes travaux de thèse portent sur les applications de l'intelligence artificielle à l'imagerie médicale, avec le syndrome de Sjögren comme terrain d'application principal. J'ai jusqu'ici travaillé avec un modèle vision-langage de type CLIP, exploité en encodeur gelé pour la classification d'images médicales.

Cette approche m'a conduit à une limite précise, et à mon sens généralisable : **un encodeur pré-entraîné gelé ne se dégrade pas seulement quand la pathologie change, mais quand l'appareil change.** En échographie des glandes salivaires — modalité de référence pour le syndrome de Sjögren — la variabilité liée au constructeur, aux réglages de gain et de profondeur, à la fréquence de sonde et à l'opérateur est du même ordre que la variabilité pathologique que l'on cherche à mesurer.

L'axe que je propose consiste à ne plus considérer l'inférence comme une passe figée, mais comme la recherche d'un état d'équilibre dans l'espace latent, formalisé au moyen d'une mesure de Gibbs. Le modèle réajuste alors, au moment du test et sans étiquette, une fraction infime de ses paramètres, et adapte sa profondeur d'analyse à la difficulté du cas.

**Le point qui articule cet axe à mon travail antérieur est une observation formelle que j'ai établie et qui m'a paru décisive :** l'inférence zero-shot de CLIP *est déjà* une mesure de Gibbs, dont les plongements textuels jouent le rôle de centroïdes et dont la température apprise est la température thermodynamique. Le cadre proposé n'est donc pas plaqué sur mon travail existant — il en constitue la lecture naturelle, et l'étend.

Le développement théorique est achevé et donne deux propositions démontrées. La revue de littérature est faite. La préparation expérimentale est engagée : protocole arrêté, cohortes de substitution identifiées, dépôt logiciel initialisé. Aucun résultat n'est encore mesuré.

**Ce que je vous demande.** Huit questions figurent en section 8. Elles portent sur des décisions qui ne m'appartiennent pas : l'articulation de cet axe avec l'économie générale de la thèse, le choix de la pathologie et l'accès aux données, la validation du formalisme, et les moyens.

## 2. Position de cet axe dans ma thèse

### 2.1 L'acquis : un encodeur vision-langage gelé

Mon travail antérieur repose sur un modèle de type CLIP, entraîné par alignement contrastif entre images et descriptions textuelles. Son intérêt en imagerie médicale est double. D'abord il fournit un espace de représentation structuré sans qu'il soit nécessaire de disposer d'un grand jeu annoté — atout déterminant pour une maladie rare. Ensuite il permet de définir les classes par du texte plutôt que par des exemples, ce qui autorise une classification dite zero-shot ou few-shot.

C'est précisément cette configuration — encodeur gelé, classes définies par des ancres — qui rend le cadre proposé ici immédiatement applicable, comme la section 4 le montre.

### 2.2 La limite rencontrée : le décalage d'acquisition

Un modèle mis au point sur les images d'un centre se dégrade sur celles d'un autre, sans que la pathologie ait changé. Les causes sont techniques : constructeur de l'échographe, préréglages, fréquence de la sonde, gain, profondeur, compression dynamique, et dépendance à l'opérateur.

Ce phénomène est particulièrement documenté pour l'échographie des glandes salivaires. Le système de cotation OMERACT, aujourd'hui central dans l'évaluation échographique du syndrome de Sjögren, fait l'objet d'études de concordance inter-observateurs qui montrent une variabilité résiduelle notable. **Autrement dit, le bruit d'acquisition et le bruit d'interprétation sont, dans cette modalité, du même ordre de grandeur que le signal recherché.**

Trois circonstances aggravent le problème dans le cas d'une maladie rare : les cohortes sont petites, les annotations coûteuses car elles requièrent un expert, et les études sont par nature multicentriques — donc exposées au décalage réel plutôt que simulé.

### 2.3 Pourquoi ce terrain est favorable à l'approche proposée

Une méthode qui adapte le modèle au moment du test, sans étiquette et sans réentraînement, apporte exactement ce qui manque ici : elle ne consomme pas d'annotations, elle n'exige pas de rassembler les données de plusieurs centres en un même lieu — ce qui simplifie considérablement les questions réglementaires — et elle traite le décalage au moment où il se manifeste, c'est-à-dire à l'examen.

> **Formulation de l'axe** Cet axe n'est pas une application supplémentaire de l'IA au syndrome de Sjögren. C'est une **contribution méthodologique transverse**, dont le syndrome de Sjögren constitue le terrain de validation le plus exigeant — et qui serait, si elle aboutit, transposable aux autres pathologies étudiées dans la thèse. C'est sur ce statut que je souhaiterais votre avis (question Q1).

## 3. L'axe proposé, en termes simples

On munit l'espace de représentation du modèle d'une mesure issue de la physique statistique, et l'on distingue deux régimes :

- **État désordonné —** l'image ne se rapproche nettement d'aucune ancre de décision ; la représentation est diffuse, l'incertitude élevée. C'est ce qui se produit face à une image issue d'un appareil non vu.

- **État ordonné —** la représentation s'est condensée autour d'une ancre identifiée ; la décision peut être prise.

Le modèle dispose alors de deux leviers, actionnés au moment de l'examen, sans étiquette :

- **Réajuster une fraction infime de ses paramètres —** les coefficients d'échelle et de décalage des couches de normalisation, et quelques jetons de guidage visuel, soit de l'ordre de 0,05 % du modèle. Le reste demeure strictement gelé, ce qui écarte tout risque d'oubli catastrophique.

- **Adapter sa profondeur d'analyse —** dès que l'état est suffisamment ordonné, l'inférence s'arrête et les blocs suivants ne sont pas exécutés. Un cas qui n'atteint jamais cet état est signalé pour relecture humaine.

Une condition gouverne toute la cohérence de l'analogie, et elle est devenue le cœur théorique du travail : **un cristal exige la condensation et le maintien de sites distincts.** Une condensation vers un site unique n'est pas un cristal, c'est un effondrement — et c'est précisément vers quoi converge toute minimisation d'incertitude qui n'est pas régularisée. La Proposition 1 de la section 4 traite ce point.

## 4. Développement théorique

### 4.1 La mesure de Gibbs, et le lien avec CLIP

Soit z le plongement normalisé d'une image et $\mu_k$ les ancres de décision. On définit l'énergie de configuration $E_k(z) = \lVert z - \mu_k \rVert^{2}$ et le postérieur de Boltzmann :

$$
p_k(z) \;=\; \frac{\exp\!\big(-E_k(z)/T\big)}{Z(z)}\,,\qquad Z(z) \;=\; \sum_{j=1}^{K} \exp\!\big(-E_j(z)/T\big)
$$

Sur la sphère unité, où vivent les plongements de CLIP après normalisation, on a l'identité $\lVert z - \mu_k \rVert^{2} = 2 - 2\langle z, \mu_k \rangle$. Il vient donc :

$$
-\,\frac{\lVert z - \mu_k \rVert^{2}}{T} \;=\; \frac{2}{T}\,\langle z, \mu_k \rangle \;-\; \frac{2}{T}
$$

Le second terme ne dépend pas de k et disparaît dans la normalisation. Par conséquent :

$$
p_k(z) \;=\; \operatorname{softmax}_k\!\left( \frac{\langle z, \mu_k \rangle}{T/2} \right)
$$

> **Observation centrale** C'est **exactement** l'inférence zero-shot de CLIP, à condition d'identifier les ancres $\mu_k$ aux plongements textuels des classes et de poser $T = 2\tau$, où τ est la température apprise du modèle. **L'inférence de CLIP est donc déjà une mesure de Gibbs sur l'hypersphère** — un fait qui, à ma connaissance, n'est pas exploité comme tel dans la littérature. Le cadre thermodynamique n'est pas une métaphore appliquée de l'extérieur : c'est la lecture exacte de ce que fait le modèle que j'utilise déjà.

Trois conséquences pratiques en découlent. Les ancres n'ont pas à être apprises : les descriptions textuelles des classes les fournissent directement, quitte à les affiner ensuite sur une petite cohorte annotée. La température cesse d'être un paramètre technique pour devenir une grandeur physique interprétable. Et les observables introduites ci-dessous sont calculables sur tout modèle de ce type, sans le modifier.

### 4.2 Trois observables au lieu d'une

De la mesure de Gibbs découlent trois grandeurs distinctes, là où la pratique courante n'en exploite qu'une :

- **L'entropie** $\mathcal{H} = -\sum_k p_k \ln p_k$, qui mesure le désordre de l'assignation. C'est la grandeur classiquement minimisée en adaptation au moment du test.

- **L'énergie libre** $F = -T \ln Z$, qui mesure le niveau absolu d'adéquation aux ancres.

- **L'énergie moyenne** $\langle E \rangle = \sum_k p_k E_k$.

Elles sont liées par l'identité thermodynamique $F = \langle E \rangle - T\,\mathcal{H}$, que le code vérifie par un test unitaire à la précision machine.

### 4.3 Proposition 1 — la répulsion exclut l'effondrement

Toute configuration où tous les échantillons d'un lot sont assignés à une même ancre atteint une entropie nulle : c'est donc un minimiseur global du terme de condensation pris isolément. Une méthode qui minimise l'incertitude sans autre contrainte peut ainsi converger vers une solution où le modèle prédit une classe unique avec une confiance parfaite — un effondrement dont la signature est indiscernable d'une cristallisation réussie.

En ajoutant un terme de répulsion entre prototypes ré-estimés sur le flux de test, on montre que sur toute configuration effondrée les prototypes se confondent et le terme de répulsion s'annule, tandis qu'une configuration séparée atteint une énergie strictement négative. Il s'ensuit que :

> **Proposition 1** Pour tout coefficient de répulsion strictement positif, aucune configuration effondrée n'est un minimiseur global de l'énergie en ligne, dès lors qu'une configuration séparée est atteignable.

Cet énoncé donne au terme de répulsion une fonction précise, et fournit la distinction formelle entre la méthode proposée et la minimisation d'entropie classique. Sur le plan clinique, il garantit que le modèle ne peut pas « résoudre » son incertitude en déclarant toutes les glandes pathologiques.

### 4.4 Proposition 2 — l'entropie ne peut pas servir de critère d'arrêt

Sous une translation uniforme du paysage énergétique, l'entropie est invariante tandis que l'énergie libre est covariante. Il en résulte qu'une image éloignée de toutes les ancres — signature typique d'un examen issu d'un appareil non vu — peut présenter une entropie basse, donc une confiance apparente élevée, tout en ayant une énergie libre élevée.

**C'est le mode de défaillance le plus dangereux en contexte clinique :** un modèle confiant et hors de son domaine de validité. L'entropie ne le détecte pas ; l'énergie libre le détecte. Cette proposition fonde le découplage retenu — on adapte le modèle en agissant sur l'entropie, on décide de l'arrêt de l'inférence en observant l'énergie libre.

*Précision d'honnêteté :* l'énergie libre entretient une parenté avec des scores déjà employés isolément pour la détection hors-distribution, notamment le score énergétique de Liu et al. (2020) et le maximum de similarité conceptuelle utilisé pour CLIP. L'apport revendiqué n'est donc pas le score lui-même, mais son unification avec l'entropie dans un même cadre, et son emploi comme critère d'arrêt assorti d'une garantie.

### 4.5 Une garantie statistique plutôt qu'un seuil

Les seuils d'arrêt ne sont pas choisis mais calibrés, par contrôle conforme du risque sur un jeu dédié. La garantie s'énonce ainsi : la dégradation diagnostique imputable à l'arrêt anticipé reste inférieure à un niveau fixé à l'avance, avec une confiance donnée.

C'est, à mon sens, ce qui distingue une contribution d'efficience d'une contribution utilisable en clinique. Un praticien n'accepte pas une économie de calcul ; il accepte une économie de calcul assortie d'une borne sur le risque.

> **Réserve** Cette garantie suppose l'échangeabilité entre le jeu de calibration et le jeu de test, hypothèse violée par construction en présence de décalage. Le manuscrit devra rapporter séparément les régimes de calibration et discuter cette limite. Je préfère l'énoncer nous-mêmes plutôt que de la laisser à un relecteur.

## 5. Revue de littérature et positionnement

### 5.1 Adaptation au moment du test

Le domaine s'organise autour de la minimisation d'entropie sur les couches de normalisation, introduite par TENT (Wang et al., ICLR 2021), puis enrichie par le filtrage d'échantillons (EATA, ICML 2022), l'adaptation continue (CoTTA, CVPR 2022), la perturbation de netteté (SAR, ICLR 2023) et des critères de fiabilité alternatifs (DeYO, ICLR 2024). Une littérature récente traite explicitement de l'effondrement des représentations lors de l'adaptation, dont un travail accepté à CVPR 2026 — c'est la ligne la plus proche de la notion de cristallisation, et celle que je surveille pour l'antériorité.

### 5.2 Adaptation des modèles vision-langage

Le test-time prompt tuning (TPT, NeurIPS 2022) adapte les invites textuelles par minimisation d'entropie sur des augmentations d'une même image. Sa version calibrée (C-TPT, ICLR 2024) montre que cette adaptation dégrade la calibration — observation qui rejoint directement la Proposition 2 et conforte le choix de découpler l'objectif d'adaptation du critère d'arrêt. Des travaux récents interrogent par ailleurs la robustesse réelle des modèles vision-langage médicaux.

### 5.3 Inférence dynamique

La sortie anticipée pour Vision Transformers est explorée depuis AdaViT et DynamicViT, puis LGViT (2023) et BEEM (ICLR 2025). Ces travaux fondent l'arrêt sur la confiance de classification, sans garantie statistique et sans traiter le décalage de distribution.

### 5.4 IA et syndrome de Sjögren

Le domaine est actif et récent. Les travaux publiés portent principalement sur deux fronts : la classification de biopsies de glandes salivaires accessoires par réseaux convolutifs, y compris l'évaluation automatisée du focus score ; et l'analyse d'images échographiques, avec des résultats encourageants mais des cohortes monocentriques. Une revue de synthèse parue en 2025 dans l'International Journal of Rheumatic Diseases dresse l'état des méthodes d'apprentissage automatique et profond appliquées au diagnostic du syndrome de Sjögren.

**Le décalage inter-centre en échographie salivaire n'y est, à ma connaissance, pas traité comme un problème méthodologique en soi.** C'est l'espace que cet axe se propose d'occuper.

### 5.5 Où se situe l'apport revendiqué

| **Élément**           | **Existant**                                      | **Apport proposé**                                                                                                    |
|-----------------------|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| Cadre formel          | Entropie employée seule, comme heuristique        | Mesure de Gibbs explicite ; trois observables liées par une identité ; lecture thermodynamique de l'inférence de CLIP |
| Objectif d'adaptation | Minimisation d'entropie, sujette à l'effondrement | Terme de répulsion sur prototypes souples, avec exclusion démontrée de l'effondrement                                 |
| Critère d'arrêt       | Confiance de classification, seuil heuristique    | Énergie libre, seuils calibrés par contrôle conforme du risque                                                        |
| Terrain               | Cohortes monocentriques, décalage simulé          | Décalage d'acquisition réel en échographie salivaire, maladie rare                                                    |

## 6. Objectifs et critères de succès

Les objectifs sont formulés comme des questions vérifiables. Aucune valeur de performance n'est avancée : elles seront produites par l'expérimentation, ou ne le seront pas.

| **Réf** | **Objectif**                                                                                             | **Critère de vérification**                                                       |
|---------|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| O1      | Établir que les observables de Gibbs, mesurées sur un encodeur gelé, détectent le décalage d'acquisition | Séparation mesurable de l'énergie libre entre domaine source et domaine cible     |
| O2      | Déterminer si l'adaptation sélective restaure la performance inter-centre sans étiquette                 | Gain d'AUROC significatif sur décalage réel, à budget de calcul égal ou inférieur |
| O3      | Déterminer si l'arrêt anticipé réduit le coût sans dégradation garantie                                  | Risque empirique conforme à la borne annoncée sur le jeu de calibration           |
| O4      | Vérifier que le terme de répulsion prévient effectivement l'effondrement                                 | Effondrement observé lorsque le coefficient est nul, absent sinon                 |
| O5      | Évaluer la portabilité vers une seconde modalité ou pathologie                                           | Reproduction des conclusions sur un second jeu de données                         |

> **Méthode de travail** Le protocole prévoit des critères d'arrêt écrits à l'avance, y compris des conditions d'échec explicites. En particulier, si la méthode complète ne dépasse pas la composition naïve « adaptation classique plus arrêt anticipé calibré », l'axe se replie sur l'arrêt anticipé à risque contrôlé, contribution plus étroite mais publiable. **Je souhaiterais que ce critère soit validé maintenant plutôt que discuté au moment où il se présenterait** (question Q4).

## 7. Préparation expérimentale engagée

Le développement théorique et la revue étant achevés, j'ai engagé la préparation de l'implémentation. À ce stade il s'agit de mise en place, non de résultats.

### 7.1 Ce qui est fait

- **Protocole expérimental arrêté —** cohortes, configurations d'ablation, métriques diagnostiques et de calibration, traitement statistique à cinq graines avec bootstrap apparié et test de DeLong.

- **Cohortes de substitution identifiées —** en attendant l'accès à des données cliniques, deux jeux publics permettent de valider la mécanique sur des phénomènes analogues : MedMNIST-C, qui fournit des corruptions réalistes par modalité, y compris échographiques, avec une sévérité continue ; et Camelyon17-WILDS, qui offre un décalage inter-hôpitaux réel en histopathologie — proxy direct du problème posé par les biopsies de glandes salivaires accessoires.

- **Environnement et dépôt logiciel —** chaîne PyTorch et CUDA installée, dépôt structuré, plan d'implémentation découpé en treize étapes, chacune assortie d'un test d'acceptation.

- **Garde-fous méthodologiques —** registre des exécutions associant chaque valeur publiée à sa configuration et au commit correspondant ; tests vérifiant par assertion le gel effectif du modèle et l'absence de dérive entre patients successifs, ce dernier point constituant aussi l'argument technique de conformité au RGPD.

### 7.2 Ce qui vient

| **Étape**        | **Contenu**                                                    | **Aboutissement attendu**                                                  |
|------------------|----------------------------------------------------------------|----------------------------------------------------------------------------|
| Court terme      | Noyau formel, encodeur instrumenté, mécanique de gel           | Vérification par test de l'identité thermodynamique et de la Proposition 2 |
| Moyen terme      | Ancrage des repères, méthodes de comparaison, énergie en ligne | Vérification expérimentale de la Proposition 1                             |
| Campagnes        | MedMNIST-C, puis Camelyon17-WILDS                              | Premiers résultats mesurés, et décision sur la poursuite de l'axe          |
| Terrain clinique | Données échographiques ou histologiques Sjögren                | Subordonné à l'accès aux données (question Q3)                             |

**Le premier résultat mesuré** sera la trajectoire de l'entropie en fonction de la profondeur du réseau, attendue sous une dizaine de jours de travail. Elle teste déjà une hypothèse non triviale : rien n'impose a priori que le désordre décroisse avec la profondeur, et une réfutation serait à consigner comme telle.

## 8. Huit questions d'orientation

*Une réponse brève, question par question, me suffirait pour engager les prochaines semaines sans risque de revenir en arrière.*

### A. Positionnement dans la thèse

#### Q1 — Cet axe doit-il être un chapitre méthodologique transverse, ou un volet applicatif ?

> Le cadre proposé est indépendant de la pathologie : il porte sur la robustesse d'un encodeur gelé face au décalage d'acquisition. Le syndrome de Sjögren en serait le terrain de validation. Je le vois comme une contribution méthodologique dont les autres volets de la thèse pourraient bénéficier, mais cette lecture engage l'architecture du manuscrit.
>
> **Ce qui en dépend** La structure de la thèse, l'ordre de rédaction des chapitres, et le poids relatif à donner à la validation clinique par rapport à la démonstration méthodologique.

#### Q2 — Le syndrome de Sjögren reste-t-il le terrain le plus pertinent pour cet axe ?

> Il présente selon moi la conjonction la plus favorable : modalité fortement dépendante de l'appareil et de l'opérateur, cohortes petites, études multicentriques par nécessité. Une autre pathologie où j'aurais un accès plus rapide aux données pourrait toutefois être préférable pour une première démonstration.
>
> **Ce qui en dépend** Le choix des cohortes et l'ordre des expériences. Je peux réorienter sans coût tant que l'implémentation n'est pas engagée sur des données réelles.

### B. Données et validation

#### Q3 — Quel accès pouvons-nous obtenir à des données cliniques réelles, et de combien de centres ?

> La démonstration repose entièrement sur l'existence d'un décalage inter-centre réel. Des données provenant d'un seul établissement ne permettraient de valider que la mécanique, pas la thèse. Une cohorte, même modeste, issue de deux ou trois services d'imagerie ou d'anatomopathologie serait décisive.
>
> **Ce qui en dépend** La validité de la démonstration principale, le niveau de revue accessible, et le calendrier — la constitution d'un accès aux données étant généralement le poste le plus long.

#### Q4 — Validez-vous par avance le critère d'arrêt de l'axe et son repli ?

> Si la méthode complète ne dépasse pas significativement la composition « adaptation classique plus arrêt anticipé calibré », le plan prévoit un repli sur l'arrêt à risque contrôlé sans adaptation. Je souhaiterais que cette bifurcation soit actée maintenant.
>
> **Ce qui en dépend** La sérénité de la phase expérimentale. Un critère d'échec accepté en amont transforme un résultat décevant en information ; discuté après coup, il le transforme en pression.

### C. Validation scientifique

#### Q5 — Validez-vous le formalisme et les démonstrations des deux propositions ?

> La Proposition 1 fonde la distinction entre la méthode et la minimisation d'entropie classique ; la Proposition 2 fonde le découplage entre adaptation et arrêt. Ce sont les deux points où une faiblesse de démonstration serait rédhibitoire. Le développement complet est disponible dans le document de cadre théorique.
>
> **Ce qui en dépend** La section théorique du manuscrit, et l'ensemble de l'architecture logicielle qui en découle.

#### Q6 — Comment jugez-vous la marge de nouveauté face au test-time prompt tuning et à ses variantes ?

> Mon appréciation est que la marge tient à la combinaison — énergie libre comme critère d'arrêt, terme de répulsion démontré, garantie conforme, application au décalage d'acquisition en échographie — plutôt qu'à l'un de ces éléments pris isolément. Je préfère que cette appréciation soit vérifiée avant l'investissement expérimental.
>
> **Ce qui en dépend** La formulation de la contribution revendiquée, et le risque d'un refus au motif d'antériorité.

### D. Moyens

#### Q7 — Quel encodeur de base retenir, et disposons-nous des moyens de calcul correspondants ?

> Trois options se présentent : le modèle CLIP généraliste que j'utilise déjà, un modèle biomédical spécialisé de type BiomedCLIP, ou un modèle dédié à l'échographie. Je travaille sur une carte graphique de 12 Go, suffisante pour les architectures de taille intermédiaire mais contraignante au-delà.
>
> **Ce qui en dépend** Le dimensionnement du protocole et la durée des campagnes. Une allocation de calcul au laboratoire permettrait d'élargir les ablations.

#### Q8 — Un co-encadrement clinique est-il envisageable ?

> Le travail comporte des choix qui relèvent du praticien plutôt que de l'ingénieur : définition des classes et rédaction des descriptions textuelles servant d'ancres, seuil de spécificité cliniquement acceptable, pertinence du renvoi à l'expert. Un rhumatologue ou un radiologue associé au projet renforcerait considérablement la partie applicative.
>
> **Ce qui en dépend** La crédibilité clinique du travail, et le choix des métriques. C'est également ce qu'attend un relecteur de revue d'imagerie médicale.

## 9. En conclusion

Cet axe me paraît présenter trois qualités qui justifient de l'engager : il prolonge naturellement mon travail sur les encodeurs vision-langage plutôt que de s'en écarter, puisque le cadre thermodynamique est la lecture exacte de l'inférence zero-shot ; il traite un problème que la littérature sur l'IA appliquée au syndrome de Sjögren n'aborde pas encore comme tel ; et il produit des énoncés démontrables là où le domaine procède le plus souvent par heuristique.

Il présente en contrepartie un risque que je préfère nommer : la contribution pourrait se révéler plus étroite qu'espéré, si l'apport se réduisait à la composition de deux techniques existantes. C'est pourquoi le protocole place très tôt l'expérience qui tranche cette question, et prévoit explicitement le repli correspondant.

Je suis prêt à engager les campagnes dès réception de vos retours, et reste disponible pour présenter le cadre de vive voix si vous le jugez utile.

*Documents de travail disponibles : cadre théorique et démonstrations, protocole expérimental détaillé, guide d'implémentation, et revue de littérature référencée.*
