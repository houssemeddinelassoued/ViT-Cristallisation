# Instructions Claude Code — ViT-Cristallisation

Projet de recherche (thèse) : détection de décalage d'acquisition par observables
thermodynamiques (énergie libre F vs entropie H) sur encodeurs vision-langage **gelés**
(CLIP), imagerie médicale. Langue de travail : **français** (code commenté, docs, commits).

> **Fichier jumeau.** Ce document est la version Claude Code de
> `.github/copilot-instructions.md` : même contenu, deux assistants. Toute règle
> ajoutee ici doit etre reportee dans le fichier Copilot, et reciproquement.
> Les regles de code du sous-depot restent dans `tlsc-repo/CLAUDE.md`
> (notation figee, pieges connus) ; ce fichier-ci porte l'architecture du depot,
> la checklist de coherence et les commandes.

## Architecture du dépôt — où va chaque chose

| Chemin | Rôle | Règle |
|---|---|---|
| `tlsc-repo/tlsc/` | Bibliothèque scientifique (noyau Gibbs, ancres CLIP, métriques) | Tout module arrive avec son test dans `tlsc-repo/tests/` |
| `tlsc-repo/experiments/` | Scripts d'expérience versionnés (`exp01_*`) | Seule source légitime de chiffres |
| `tlsc-repo/outputs/<run_id>/` | Résultats mesurés (`metrics.json`, `scores.npz`, `analysis.json`, figures) | Jamais modifiés à la main ; `outputs/aggregate/` = synthèse régénérable |
| `tlsc-repo/docs/` | Notes internes : `JOURNAL.md`, `rapport-exp01.md`, `STATUT.md`, `cadre-theorique.md` | Une entrée JOURNAL par lot de runs |
| `papers/contribution/` | Papier LaTeX (contribution) | Compiler avec `latexmk -pdf main.tex` (MiKTeX) |
| `docs/` (racine) | **Site public GitHub Pages** : `index.html`, `results.html`, `research_works.html` (copie unique), `assets/figures/` | Déployé depuis `main` — voir checklist de cohérence |
| `papers/review/` | Revue de littérature LaTeX | Indépendante du papier de contribution |
| `archive/` | Documents de cadrage historiques (feuilles de route, plans, guides, notes) | Lecture seule ; jamais autoritatifs |
| Racine | `README.md` + `02protocoleexperimentalv3.md` (protocole autoritatif, référencé par `tlsc-repo/docs/`) | Ne pas créer de nouveaux fichiers en racine ; les nouveaux docs vont dans `tlsc-repo/docs/`, `docs/` ou `archive/` |

Hiérarchie des sources en cas de divergence (voir `tlsc-repo/docs/STATUT.md`) :
`outputs/<run_id>/metrics.json` > `02protocoleexperimentalv3.md` (v3) >
`tlsc-repo/docs/cadre-theorique.md` > documents HTML historiques de la racine.

## Règles scientifiques non négociables (issues de `tlsc-repo/CLAUDE.md`)

- **Aucune valeur de performance en dur** — nulle part, même en commentaire ou en HTML.
  Un chiffre absent d'un `outputs/<run_id>/metrics.json` **n'existe pas** ; chaque valeur
  publiée (papier, site, rapport) cite son `run_id`.
- **Invites textuelles figées a priori** (`tlsc/models/clip_anchors.py`) : les choisir au
  vu des résultats cibles est une fuite qui invalide la démonstration.
- **Température T = 2τ**, jamais recalibrée sur la cible. Notation figée : `z`, `mu`,
  `T`, `H`, `F`, `E`, `chi` — ne jamais renommer.
- Stabilité numérique : `log_softmax`/`logsumexp`, jamais `p.log()`.
- L'encodeur (visuel et texte) reste **gelé** ; les ancres ne s'adaptent jamais.
- Une réfutation proprement exécutée est un résultat : la consigner telle quelle
  (le verdict pré-enregistré C1/C2/C3 est actuellement **réfuté** — ne pas « corriger »).
- Runs citables : arbre git **propre** uniquement (`environment.git_dirty: false`).

## Environnement et commandes

Python : venv dédié `tlsc-repo/.venv` (3.11, torch CPU — pas de CUDA sur cette machine).
Piège pwsh : préférer `Set-Location tlsc-repo; .venv\Scripts\python.exe -m ...`
(l'opérateur `&` avec arguments provoque parfois des ParserError).

```powershell
Set-Location tlsc-repo
.venv\Scripts\python.exe -m pytest tests -q          # DOIT être vert avant/après toute modif
.venv\Scripts\python.exe -m ruff check tlsc experiments tests
.venv\Scripts\python.exe -m experiments.exp01_zero_training --dataset breastmnist --source medmnistc --corruption speckle_noise --severities 0 1 2 3 4 5 --batch-size 32
.venv\Scripts\python.exe -m experiments.exp01_analysis outputs    # DeLong + bootstrap → analysis.json
.venv\Scripts\python.exe -m experiments.exp01_aggregate outputs   # summary.{json,csv} + figures → outputs/aggregate/
```

Coût : ~0,2 s/image CPU (BreastMNIST n=156 ≈ 3 min/corruption ; PneumoniaMNIST n=624 ≈ 13 min).
Encodage fichiers : toujours `encoding="utf-8"` explicite (Windows/cp1252 a déjà cassé
des runs) ; pas de caractères non-ASCII dans les `print()` de scripts (console cp1252).

## Checklist de cohérence — OBLIGATOIRE après chaque modification

Après tout changement de résultats, de code ou de documents, propager dans **tous** les
emplacements concernés avant de committer :

1. **Nouveaux runs** → relancer `exp01_analysis` puis `exp01_aggregate` ; entrée dans
   `tlsc-repo/docs/JOURNAL.md` (format : date · run_id · testé · observé · décision) ;
   section dans `tlsc-repo/docs/rapport-exp01.md` si le résultat est substantiel.
2. **Figures** — trois copies à synchroniser depuis `tlsc-repo/outputs/aggregate/` :
   `papers/contribution/figures/` et `docs/assets/figures/`. Ne jamais éditer une copie seule.
3. **Site public** (`docs/`) :
   - `docs/results.html` : tableau des runs canoniques (AUROC, IC, p, run_id) et figures
     à jour avec `outputs/aggregate/summary.csv` ;
   - `docs/index.html` : cartes et liens cohérents avec les pages existantes ;
  - `docs/research_works.html` : **copie unique** (la copie racine a été retirée
     le 2026-09-01) ; support pédagogique — aucun chiffre ne doit y être présenté comme mesure.
4. **Papier** (`papers/contribution/main.tex`) : si un chiffre cité change, mettre à jour le
   tableau + l'abstract, recompiler (`latexmk -pdf`), vérifier zéro référence indéfinie
   dans `main.log`.
5. **READMEs** : `README.md` racine (structure, règles, liens) et `tlsc-repo/README.md`
   (commandes, état de l'expérience) reflètent l'état réel — pas d'aspirationnel.
6. **Protocole** : nouveaux résultats de la matrice → remplir la ligne correspondante
   des tableaux §8 de `02protocoleexperimentalv3.md` avec le run_id.
7. **`tlsc-repo/docs/STATUT.md`** : tenir à jour la liste des sources autoritatives si
   un document est ajouté/retiré/déprécié.
8. Tests + lint verts, puis commit.

## Convention git (voir README racine)

- `main` = stable, tests verts, déployé sur GitHub Pages. Pas de commit direct pour du
  substantiel ; corrections triviales de docs tolérées.
- Branches courtes depuis `main` : `feature/…`, `exp/…`, `fix/…`, `docs/…` ;
  merge `--no-ff`, suppression de la branche après fusion.
- Messages de commit en français, impératif, ASCII de préférence (pas d'accents cassés).
- Ne jamais pousser sans demande explicite de l'utilisateur.

## Style de code

- Python : type hints partout, docstrings NumPy en français,
  `from __future__ import annotations`, lint `ruff`.
- Modules scientifiques : validation des entrées aux frontières (`ValueError` explicites),
  reproductibilité par graine, chaque nouvelle métrique testée dans `tests/test_metrics.py`.
- HTML du site : pages autonomes (CSS inline, pas de framework), sombres, en français,
  avec la mention systématique de la règle de citation des résultats.
