# Règles de code — projet TLSC

## Contexte
Adaptation au moment du test et arrêt anticipé à risque contrôlé, sur un **encodeur
vision-langage gelé** de type CLIP, en imagerie médicale (terrain : syndrome de Sjögren,
échographie). Formalisme de référence : `docs/cadre-theorique.md`.

## Notation (ne jamais renommer)
z        embedding image L2-normalisé, (B, d)
mu       ancres = plongements TEXTUELS des classes, L2-normalisées, (K, d)
T        température de Gibbs ; T = 2*tau où tau est la température apprise de CLIP
H        entropie de Gibbs      = -sum_k p_k ln p_k
F        énergie libre          = -T * logsumexp(-d2/T)
E        énergie moyenne        = sum_k p_k d2_k
chi      cristallinité          = 1 - H / ln K
theta_adapt   LayerNorm de l'encodeur VISUEL + prompts visuels
theta_frozen  tout le reste, y compris l'encodeur texte et les ancres

## Standards
- Type hints partout, docstrings NumPy, `from __future__ import annotations`.
- Aucune valeur de performance en dur, jamais, même en commentaire.
- Stabilité numérique : `log_softmax` et `logsumexp`, jamais `p.log()`.
- Tout module scientifique arrive avec son test dans `tests/`.
- Toute figure ou tableau publié provient d'un `outputs/<run_id>/metrics.json`.

## Pièges connus — à ne pas reproduire
1. Les forward hooks NE PEUVENT PAS interrompre le calcul. L'arrêt anticipé exige une
   boucle explicite sur les blocs du transformeur visuel.
2. Ne jamais entraîner ni adapter l'encodeur TEXTE dans la voie par défaut : les ancres
   doivent rester fixes, sinon l'ancrage perd son sens.
3. Mesure de latence : échauffement + `torch.cuda.synchronize()` + `cuda.Event`.
4. Le comptage de FLOPs inclut les passes ARRIÈRE (~2x une passe avant) et, pour TPT,
   les dizaines de vues augmentées par image.
5. theta_adapt est restauré après chaque lot. Une dérive entre patients est un bug.
6. Les invites textuelles sont figées sur la validation SOURCE. Les choisir au vu des
   résultats cibles est une fuite qui invalide la démonstration.
