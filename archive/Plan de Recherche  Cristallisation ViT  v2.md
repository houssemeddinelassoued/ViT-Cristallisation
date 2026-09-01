---
title: "Plan de Recherche"
subtitle: "Cristallisation ViT"
version: 2
date: 2026-08-30
auteur: "Houssem Eddine Lassoued"
encadrement: "Dr. Anis Ben Aicha, Pr. Habib Fathallah"
math: latex
---

# Plan de Recherche

*Thermodynamic Latent Space Crystallization : Energy-Driven Test-Time Adaptation and Risk-Controlled Early Exiting for Medical Vision Transformers*

Version 2 — 30 août 2026. Remplace intégralement la version 1.

Houssem Eddine Lassoued — Encadrement : Dr. Anis Ben Aicha, Pr. Habib Fathallah

Matériel : RTX 4070 (12 Go) · i7 · 32 Go RAM

> **Ce qui change** Le titre provisoire est amendé : « Dynamic Entropy-Driven » devient « Energy-Driven », et « Early Exiting » devient « Risk-Controlled Early Exiting ». Ces deux modifications ne sont pas cosmétiques — elles reflètent le découplage entre l'entropie, qui gouverne l'adaptation, et l'énergie libre, qui gouverne la sortie, ainsi que le passage d'un seuil heuristique à une garantie statistique.

## 1. Introduction et problématique

**Contexte.** Les modèles de fondation de type ViT atteignent d'excellentes performances en imagerie médicale, mais souffrent de décalages de distribution lors du passage en production : constructeurs de scanner différents, protocoles de contraste, coloration inter-laboratoires, bruits d'acquisition.

**Problème.** L'inférence statique est sous-optimale. Traiter une image dégradée avec les mêmes poids fixes qu'une image nette conduit à une projection désordonnée dans l'espace latent, et à une incertitude de décision élevée.

**Contribution visée.** Une architecture qui traite l'espace latent comme un système muni d'une mesure de Gibbs, et qui adapte au moment du test à la fois sa profondeur d'analyse et une fraction minime de ses paramètres, en recherchant un état d'équilibre : minimiser le désordre de l'assignation tout en maintenant des sites de décision distincts.

**Ce que la contribution doit encore établir.** Trois questions restent ouvertes et conditionnent la narration finale : la méthode bat-elle la composition naïve TENT + sortie calibrée ; le gain de calcul survit-il à une comptabilité honnête incluant les passes arrière ; la transition de cristallisation présente-t-elle les signatures d'une véritable transition de phase. Les jalons de la section 5 y répondent.

## 2. Cadre théorique

Le formalisme complet, avec démonstrations, figure dans le document « Description Complète du Projet, version 2 », section 2, et dans claude/01-cadre-theorique.md. On en rappelle ici les quatre énoncés structurants.

### 2.1 Mesure de Gibbs et observables

Le postérieur est une distribution de Boltzmann sur les distances aux centroïdes ancrés, de température T. Trois observables distinctes en découlent : l'entropie de Gibbs H, l'énergie libre $F = -T \ln Z$, et l'énergie moyenne ⟨E⟩, liées par l'identité $F = \langle E \rangle - T\,\mathcal{H}$ (Lemme 1).

### 2.2 Les quatre énoncés du projet

- **Proposition 1 —** le minimiseur du terme entropique seul est dégénéré, et tout $\lambda$ > 0 exclut les configurations effondrées du minimum global. C'est la justification théorique du terme répulsif, et la contribution centrale face à TENT.

- **Proposition 2 —** l'entropie est invariante par translation uniforme des énergies, l'énergie libre est covariante. Un échantillon loin de tous les centroïdes a donc une entropie basse et une énergie libre élevée : seule cette dernière peut servir de critère de sortie.

- **Proposition 3 —** avec des centroïdes figés, le gradient du terme répulsif est nul. Énoncé trivial, mais c'est exactement le défaut de la version 1 du projet.

- **Conjecture C3 —** la cristallinité présente une transition abrupte en fonction de la sévérité du décalage, avec pic de susceptibilité. Statut : à établir ou à abandonner (jalon J4).

## 3. Architecture et méthodologie

### 3.1 Phase 1 — Apprentissage et ancrage des centroïdes, hors ligne

Backbone gelé, entraînement de L têtes de projection par perte contrastive supervisée, puis ancrage de L jeux de centroïdes, un par couche. Ce point est une correction majeure de la version 1, qui ne prévoyait qu'un jeu unique et rendait l'évaluation par couche métriquement dénuée de sens.

### 3.2 Phase 2 — Adaptation sélective au moment du test

Optimisation de $\theta_{\text{adapt}}$ — paramètres d'échelle et de décalage des LayerNorm du backbone, et prompts visuels — par descente de gradient sur l'énergie en ligne. L'adaptation est conditionnée à un dépassement de seuil d'énergie libre, tronquée à la couche sonde, et amortie sur un tampon glissant.

### 3.3 Phase 3 — Sortie anticipée à risque contrôlé

Évaluation séquentielle couche par couche du critère d'énergie libre, avec des seuils calibrés par contrôle conforme du risque plutôt que choisis. La garantie prend la forme : dégradation d'exactitude imputable à la sortie anticipée inférieure à $\alpha$, avec probabilité 1 − $\delta$.

**Note d'implémentation.** La version 1 prescrivait des forward hooks PyTorch pour la sortie anticipée. Les hooks observent le calcul mais *ne l'interrompent pas* : une implémentation par hooks exécuterait les douze couches et n'économiserait aucun FLOP, tout en produisant des courbes plausibles. La sortie anticipée exige une boucle explicite sur les blocs du backbone.

## 4. Protocole expérimental

Le protocole complet figure dans claude/02-protocole-experimental.md. En résumé : trois paliers de cohortes — MedMNIST-C pour la mécanique et le balayage continu, Camelyon17-WILDS pour la démonstration principale sur décalage réel inter-hôpitaux, RSNA vers ChestX-ray14 pour la seconde modalité ; neuf configurations d'ablation, dont A5 (TENT + sortie calibrée) qui constitue le contrôle décisif ; et un traitement statistique à cinq graines avec bootstrap apparié et test de DeLong.

**Modèles.** ViT-S/16 comme modèle principal pour les explorations et ablations, ViT-B/16 en confirmation sur la configuration finale. Initialisation depuis des poids ImageNet-21k ou DINOv2 ; aucun entraînement depuis zéro, hors budget matériel et sans intérêt scientifique ici.

## 5. Jalons go / no-go

Sans date imposée, la progression est pilotée par des critères de passage. Chaque jalon comporte une condition d'échec explicite et une bifurcation prévue — décider à l'avance ce qui constituerait un échec est ce qui distingue une recherche d'une illustration.

| **Jalon** | **Objet**                                   | **Critère de passage**                                                               | **Bifurcation en cas d'échec**                                                          |
|-----------|---------------------------------------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| J0        | Assainissement documentaire et socle formel | Document théorique autoportant, sans les contradictions F01, F05, F11                | —                                                                                       |
| J1        | Mécanique opérationnelle sur MedMNIST-C     | Entropie décroissante avec la profondeur, aucune divergence, gel du backbone vérifié | Problème d'implémentation : on corrige, on ne pivote pas                                |
| J2        | Supériorité sur le contrôle A5              | Gain d'AUROC significatif à budget égal ou inférieur, sur Camelyon17                 | Pivot vers la sortie anticipée à risque contrôlé sans TTA — publiable en MICCAI ou MIDL |
| J3        | Honnêteté du budget de calcul               | Coût net inférieur à l'inférence statique, passes arrière incluses                   | Repositionnement en méthode de robustesse, et publication du comptage honnête           |
| J4        | Verdict sur la transition de phase          | Pic de susceptibilité reproductible, s'affinant avec la dimension latente            | Abandon du vocabulaire thermodynamique au profit d'« energy-based »                     |
| J5        | Consolidation multi-cohortes et rédaction   | Résultats cohérents sur au moins deux modalités, dépôt reproductible                 | —                                                                                       |

## 6. Stratégie de publication

### 6.1 Cible réaliste

*Nature Machine Intelligence* publie soit des paradigmes véritablement nouveaux étayés par des preuves exceptionnelles, soit des travaux validés cliniquement sur cohortes prospectives multi-sites. Une méthode d'adaptation au moment du test évaluée sur benchmarks publics n'y passera pas — par inadéquation de format, non de qualité. La viser d'emblée coûterait plusieurs mois pour un refus de bureau.

- **Jalon —** MICCAI, MIDL ou ISBI, format huit pages, dès que J2 et J3 sont franchis. Retours de relecture rapides et matière consolidée.

- **Cible principale —** *IEEE Transactions on Medical Imaging* ou *Medical Image Analysis* : version longue avec garanties conformes, multi-cohortes et analyse physique. C'est l'audience exacte de ce travail.

- **Nature MI —** envisageable ultérieurement si J4 réussit et qu'une validation sur données hospitalières réelles s'y ajoute. Objectif à deux ans, pas première soumission.

### 6.2 Ordre de rédaction

Contre-intuitif mais nettement plus efficace : Méthodes, puis Expériences, puis Discussion, puis Introduction, et le Résumé en dernier.

> **Correction** La version 1 organisait la rédaction en trois sprints dont le troisième produisait l'Abstract et l'Introduction **avant toute expérience**. C'est précisément ce qui conduit à écrire un article autour de chiffres souhaités plutôt que mesurés — et c'est l'origine du tableau de performances fabriqué. L'introduction ne peut affirmer le manque dans la littérature qu'une fois les résultats connus ; le résumé se rédige quand la contribution réelle, parfois différente de celle espérée, est établie.

### 6.3 Figures qui portent la démonstration

- Entropie en fonction de la profondeur, par sévérité de décalage, avant et après adaptation — la figure identitaire de l'article.

- Projection UMAP de l'espace latent aux couches 1, 6 et 12, avec les centroïdes ancrés.

- Diagramme de phase de la cristallinité en fonction de la sévérité et de la profondeur, en carte de chaleur.

- Courbe risque-couverture, méthode contre baselines — l'argument clinique.

- AUROC en fonction du budget de calcul réel : le graphique où A8 doit visiblement dominer A5.

## 7. Registre des risques

| **Risque**                                      | **Gravité** | **Parade**                                                        |
|-------------------------------------------------|-------------|-------------------------------------------------------------------|
| L'apport se réduit à TENT + sortie anticipée    | Critique    | Exécuter A5 dès le jalon J2, avant tout investissement lourd      |
| L'efficience nette est négative                 | Critique    | Adaptation sélective gatée ; sinon repositionnement en robustesse |
| Effondrement pris pour cristallisation          | Élevé       | Terme répulsif actif et entropie marginale reportée par couche    |
| Physique jugée décorative en relecture          | Élevé       | Test de transition pré-enregistré et repli assumé                 |
| Camelyon17 trop lourd pour le matériel          | Modéré      | ViT-S/16, précision bf16, sous-échantillonnage stratifié          |
| Instabilité de l'adaptation à lot unitaire      | Modéré      | Entropie marginale sur augmentations, ou tampon glissant          |
| Antériorité 2025-2026 sur l'effondrement en TTA | Modéré      | Veille mensuelle ; positionner l'apport sur la garantie conforme  |

## 8. Mise en œuvre

Le plan d'implémentation détaillé — installation, configuration de l'environnement de développement, treize sprints avec code critique, prompts d'assistance et tests d'acceptation — figure dans claude/03-guide-implementation.md. Séquence résumée :

| **Sprints** | **Objet**                                               | **Durée**  | **Aboutissement**                          |
|-------------|---------------------------------------------------------|------------|--------------------------------------------|
| S0 – S2     | Environnement CUDA, éditeur, squelette du dépôt         | 3 h        | Dépôt exécutable, tests et lint verts      |
| S3 – S5     | Noyau formel, backbone instrumenté, gel et restauration | 6 h        | Lemme 1 et Proposition 2 vérifiés par test |
| S6 – S8     | Ancrage hors ligne, baselines, énergie en ligne         | 4 j        | Proposition 1 vérifiée expérimentalement   |
| S9 – S11    | Sortie anticipée, calibration, évaluation, données      | 2,5 j      | Chaîne complète mesurable                  |
| S12         | Campagnes expérimentales                                | 2 à 4 sem. | Jalons J1 à J4                             |

*Le premier résultat scientifique réel du projet est la courbe d'entropie par couche du sprint S6, atteignable en une dizaine de jours de travail.*
