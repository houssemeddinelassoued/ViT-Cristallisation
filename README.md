# Cristallisation de l'espace latent (ViT)

Détection de décalage de domaine par thermodynamique statistique sur encodeurs
vision-langage gelés (CLIP), appliquée à l'imagerie médicale.

**Site du projet (GitHub Pages)** : https://houssemeddinelassoued.github.io/ViT-Cristallisation/

## Structure du dépôt

| Chemin | Contenu |
|---|---|
| [`tlsc-repo/`](tlsc-repo/) | Code, tests, expériences, résultats mesurés (`outputs/<run_id>/metrics.json`), papier LaTeX. Voir [`tlsc-repo/README.md`](tlsc-repo/README.md) pour l'installation et l'exécution. |
| [`docs/`](docs/) | Site public GitHub Pages : accueil, page de résultats, plateforme explicative (copie unique), figures. Ne pas confondre avec `tlsc-repo/docs/` (notes internes : journal de bord, cadre théorique, statut des sources). |
| [`review-paper/`](review-paper/) | Revue de littérature complémentaire (LaTeX). |
| [`archive/`](archive/) | Documents de cadrage historiques (feuilles de route, plans, guides, notes aux encadrants). Conservés pour trace ; en cas de divergence, `02protocoleexperimentalv3.md` et `tlsc-repo/docs/` prévalent — voir [`tlsc-repo/docs/STATUT.md`](tlsc-repo/docs/STATUT.md). |
| Racine | `README.md` (ce fichier) et [`02protocoleexperimentalv3.md`](02protocoleexperimentalv3.md), le protocole expérimental autoritatif (référencé par `tlsc-repo/docs/`). |

## Règle de citation des résultats

Seule une valeur présente dans un `tlsc-repo/outputs/<run_id>/metrics.json` versionné
est un résultat mesuré. La page `docs/plateforme-explicative.html` est un support
pédagogique ; ses chiffres ne doivent jamais être cités comme mesures.

## Convention de branches et de push

- **`main`** : toujours stable (tests verts). C'est la branche déployée sur GitHub Pages
  (dossier `docs/`) et la seule source de vérité pour l'historique du projet.
- **Branches courtes par tâche**, préfixées par intention, ouvertes depuis `main` :
  - `feature/<sujet>` — nouvelle capacité (code, métrique, page).
  - `exp/<sujet>` — nouvelle expérience ou balayage de runs.
  - `fix/<sujet>` — correction de bug ou de régression.
  - `docs/<sujet>` — documentation seule (journal, rapport, README).
- **Workflow** : `git checkout -b <type>/<sujet>` depuis `main` à jour → commits atomiques
  → tests verts (`pytest` dans `tlsc-repo/`) → merge dans `main` (`--no-ff` pour garder
  la trace de la branche) → suppression de la branche locale une fois fusionnée.
- Pas de commit direct sur `main` pour des changements substantiels ; les corrections
  triviales de documentation peuvent y aller directement.
- Chaque exécution d'expérience reste liée à un commit Git précis
  (`environment.git_sha` dans `metrics.json`) ; ne pas lancer de run sur un arbre sale
  (`git_dirty: true`) pour un résultat destiné à être cité.

## Historique

L'historique du dossier `tlsc-repo/` est préservé depuis son dépôt d'origine
(fusionné par `git subtree`) : socle thermodynamique, expérience 1 zero-training,
balayage systématique des corruptions, détecteurs bilatéral et bivarié.
