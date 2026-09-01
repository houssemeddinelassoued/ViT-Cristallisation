---
title: "Guide des Connaissances"
subtitle: "Base documentaire du projet"
version: 2
date: 2026-08-30
auteur: "Houssem Eddine Lassoued"
math: latex
---

# Guide des Connaissances

*Base documentaire du projet ViT Cristallisation*

Version 2 — 30 août 2026. Met à jour la version 1 : jeu de documents actualisé, littérature 2022-2026, règles d'hygiène de la base.

> **Pourquoi cette mise à jour** La base de connaissances d'un projet Claude est lue à chaque nouvelle session. Un document obsolète qui y demeure n'est pas neutre : il **réinjecte silencieusement ses erreurs** dans toutes les réponses futures. La section 6 fixe les règles d'hygiène qui préviennent cela.

## 1. Documents de vision et cadre théorique

Ce sont les fichiers stratégiques du projet. Ils définissent la vision globale, le formalisme et le protocole.

| **Document**                        | **Contenu**                                                                                                    | **Statut**        |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------|-------------------|
| Description Complète du Projet (v2) | Genèse, formalisme de Gibbs, trois phases corrigées, protocole, intégration clinique, registre des corrections | Actif             |
| Plan de Recherche (v2)              | Feuille de route, jalons go/no-go, stratégie de publication, registre des risques                              | Actif             |
| claude/00-index.md                  | Point d'entrée, statut des documents, corrections à appliquer                                                  | Actif             |
| claude/01-cadre-theorique.md        | Formalisme complet avec démonstrations, notation canonique, Algorithme 1                                       | Actif — référence |
| claude/02-protocole-experimental.md | Cohortes, ablations, métriques, statistiques, tableaux vides                                                   | Actif             |
| claude/03-guide-implementation.md   | Installation, treize sprints, code critique, tests d'acceptation                                               | Actif             |
| claude/audit-et-plan-action.md      | Diagnostic des douze défauts et plan de correction                                                             | Actif — archive   |

## 2. Articles de référence à importer

L'importation des articles fondateurs permet des citations exactes, l'adoption du vocabulaire de la littérature, et un positionnement précis par rapport à l'état de l'art.

### 2.1 Adaptation au moment du test

| **Référence**                 | **Année** | **Pourquoi elle compte ici**                                                                                                                            |
|-------------------------------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| TENT — Wang et al., ICLR      | 2021      | Minimisation d'entropie sur LayerNorm. C'est la méthode dont il faut se démarquer, et dont la version 1 du projet était mathématiquement indiscernable. |
| MEMO — Zhang et al.           | 2022      | Entropie marginale sur augmentations : la réponse au problème du lot unitaire.                                                                          |
| EATA / ETA — Niu et al., ICML | 2022      | Filtrage d'échantillons et régularisation anti-oubli.                                                                                                   |
| CoTTA — Wang et al., CVPR     | 2022      | Adaptation continue et prévention de la dérive.                                                                                                         |
| SAR — Niu et al., ICLR        | 2023      | Minimisation avec perturbation de netteté ; documente l'effondrement de l'entropie.                                                                     |
| DeYO — Lee et al., ICLR       | 2024      | Critère de fiabilité au-delà de l'entropie.                                                                                                             |
| Neural Collapse in TTA — CVPR | 2026      | Effondrement neuronal en adaptation : littérature la plus proche de la notion de cristallisation. À surveiller pour l'antériorité.                      |

### 2.2 Inférence dynamique et sortie anticipée

| **Référence**                                | **Année** | **Pourquoi elle compte ici**                                      |
|----------------------------------------------|-----------|-------------------------------------------------------------------|
| AdaViT — Meng et al.                         | 2022      | Arrêt précoce par portes d'attention ; baseline historique.       |
| DynamicViT — Rao et al.                      | 2021      | Élagage dynamique de jetons ; mécanisme complémentaire.           |
| LGViT — Xu et al.                            | 2023      | Sortie anticipée pour ViT ; baseline moderne.                     |
| BEEM — ICLR                                  | 2025      | Amélioration des performances de sortie anticipée.                |
| Conformal Risk Control — Angelopoulos et al. | 2023      | Fondement de la calibration des seuils avec garantie statistique. |

### 2.3 Apprentissage métrique et détection hors-distribution

| **Référence**                                            | **Année** | **Pourquoi elle compte ici**                                                                         |
|----------------------------------------------------------|-----------|------------------------------------------------------------------------------------------------------|
| Supervised Contrastive Learning — Khosla et al., NeurIPS | 2020      | Fondement de la Phase 1 : structuration des centroïdes.                                              |
| Center Loss — Wen et al.                                 | 2016      | Attraction intra-classe et répulsion inter-classes.                                                  |
| Energy-based OOD Detection — Liu et al., NeurIPS         | 2020      | Le score énergétique que la formulation de Gibbs retrouve comme énergie libre. Citation obligatoire. |
| Neural Collapse — Papyan, Han, Donoho, PNAS              | 2020      | Cadre théorique de la géométrie des représentations condensées.                                      |

### 2.4 Benchmarks et données

| **Référence**                      | **Année** | **Pourquoi elle compte ici**                                                                     |
|------------------------------------|-----------|--------------------------------------------------------------------------------------------------|
| WILDS — Koh et al., ICML           | 2021      | Camelyon17 : décalage réel inter-hôpitaux, split OOD officiel, baselines publiées.               |
| MedMNIST-C — ADSMI @ MICCAI        | 2024      | Corruptions réalistes par modalité, sévérité continue : indispensable à l'analyse de transition. |
| Benchmark TTA en imagerie médicale | 2025      | Point de comparaison méthodologique récent pour le protocole.                                    |

## 3. Spécifications techniques et code

Les extraits de code à conserver dans la base sont ceux qui portent la sémantique du projet, non ceux qui changent à chaque itération.

- **tlsc/core/gibbs.py —** les trois observables. C'est le fichier dont dépend la validité de tous les résultats.

- **tlsc/core/energy.py —** l'énergie en ligne, avec le diagnostic d'effondrement.

- **tlsc/adapt/state.py —** collecte, sauvegarde et restauration de $\theta_{\text{adapt}}$.

- **tlsc/models/backbone.py —** la boucle explicite sur les blocs, qui rend la sortie anticipée réelle.

- **tests/test_freeze.py —** les deux tests non négociables : gel effectif du backbone, absence de dérive entre échantillons.

- **CLAUDE.md —** le fichier de règles lu par l'assistant à chaque requête : notation, standards, et la liste des pièges connus du projet.

Hyperparamètres clés à documenter dans un fichier de configuration : température T, coefficients $\lambda$ et $\beta$, seuil de déclenchement de l'adaptation, nombre d'itérations S, dimension latente d, niveaux $\alpha$ et $\delta$ de la garantie conforme. Les seuils de sortie n'y figurent pas : ils sont calibrés, non configurés.

## 4. Fiches descriptives des cohortes

Chaque cohorte utilisée mérite une fiche courte, qui évite les approximations dans le manuscrit.

| **Rubrique**       | **À renseigner**                                                                                   |
|--------------------|----------------------------------------------------------------------------------------------------|
| Identité           | Nom exact, version, source, licence, date de téléchargement                                        |
| Volume             | Nombre d'échantillons par split, résolution native, format de fichier                              |
| Classes            | Nombre, définition clinique, distribution — le déséquilibre conditionne le choix des métriques     |
| Nature du décalage | Réel (institution, appareil, protocole) ou simulé (type de corruption, paramétrage de la sévérité) |
| Partition          | Définition exacte des splits train / calibration / test, et domaine d'origine de chacun            |
| Prétraitement      | Redimensionnement, normalisation, augmentations, gestion des DICOM                                 |

> **Point de vigilance** Le split de **calibration** doit être défini et déclaré dès la fiche. C'est lui qui porte la garantie conforme, et son domaine d'origine détermine si cette garantie est valide ou approchée. Un split de calibration improvisé en cours d'expérience invalide toute la section 3.3 du projet.

## 5. Templates de rédaction

- **Fichier LaTeX maître —** préambule au format IEEE TMI ou Elsevier, avec amsmath, amssymb, booktabs, graphicx, algorithm2e.

- **Fichier de bibliographie —** references.bib alimenté au fil de la lecture, jamais reconstitué à la fin.

- **Fichier de notation —** docs/NOTATION.md, recopie du tableau de la section 2.1 de la Description Complète. C'est la source unique de vérité pour le code comme pour le manuscrit.

## 6. Règles d'hygiène de la base de connaissances

Ces règles sont nouvelles en version 2. Elles découlent directement de l'audit.

- **Un seul document actif par sujet.** Lorsqu'une version est remplacée, l'ancienne est supprimée de la base, non conservée « au cas où ». Une base qui contient deux formulations contradictoires produit des réponses contradictoires.

- **Aucune valeur de performance non mesurée, nulle part.** Ni dans les documents, ni dans les instructions du projet, ni en commentaire de code. Un chiffre souhaité finit toujours par être cité comme un chiffre obtenu.

- **Les tableaux de résultats sont créés vides,** avec une colonne d'identifiant de run. Une cellule sans identifiant est une cellule non fiable.

- **Les conjectures portent leur statut.** Un énoncé non démontré s'écrit au conditionnel et figure dans la liste des questions ouvertes, jamais au présent de l'indicatif.

- **Les instructions du projet sont révisées en même temps que les documents.** Elles sont lues avant tout le reste et pèsent davantage que n'importe quel fichier.

## 7. Ordre d'importance

| **Priorité** | **Type de document**                                              | **Utilité majeure**                                  |
|--------------|-------------------------------------------------------------------|------------------------------------------------------|
| P0 — vital   | Description Complète v2, Plan de Recherche v2, 01-cadre-theorique | Ancrer le formalisme, les trois phases et les jalons |
| P1 — haute   | 03-guide-implementation, code noyau, CLAUDE.md                    | Guider la génération de code sans régression         |
| P2 — moyenne | Articles de la section 2, par ordre chronologique inverse         | Comparaisons théoriques et citations exactes         |
| P3 — confort | Fiches cohortes, references.bib, template LaTeX                   | Rédaction directe et protocole sans approximation    |
