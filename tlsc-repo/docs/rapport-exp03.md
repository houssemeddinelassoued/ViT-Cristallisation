# Rapport de résultats — Expérience 3 (régimes d'ancrage, cohorte P0)

**Date du rapport** : 2026-09-01
**Expérience** : `exp03_anchor_regimes` — comparaison des trois régimes d'ancrage du
protocole v3 §2.3 : R1 (invites textuelles), R2 (centroïdes few-shot k = 16),
R3 (centroïdes sur le split train source complet, n = 546).
**Modèle** : ViT-B-16-quickgelu · poids `openai` (CLIP gelé), T = 2τ ≈ 0,0200
**Cohorte** : BreastMNIST 224×224 — centroïdes sur train source (sévérité 0),
évaluation sur test (n = 156), corruptions MedMNIST-C, sévérités 0–5
**Statut épistémique** : **exploratoire déclaré** — aucun seuil pré-enregistré.
**Anti-fuite** : les centroïdes ne voient jamais une image cible ni une étiquette de test.

Origine : croquis fondateur de l'encadrant — ancres = centroïdes de classe μ₀, μ₁
calculés depuis les données, distance inter-classes D_inter comme grandeur de géométrie.

---

## 1. Runs

| run_id | Corruption | Statut |
|---|---|---|
| 20260901T173740Z_breastmnist_fe6d6b35 | speckle_noise | complet, git 00f9f37 propre ✅ |
| 20260901T174749Z_breastmnist_bccf7433 | brightness_down | complet, git 00f9f37 propre ✅ |

## 2. Géométrie et classification (identiques dans les deux runs)

| Régime | D_inter | Bal. accuracy (sév. 0) | Accuracy (sév. 0) |
|---|---|---|---|
| R1 — invites textuelles | 0,1032 | 0,517 | 0,712 |
| R2 — few-shot k=16 | 0,0208 | 0,597 | — |
| R3 — centroïdes complets | 0,0073 | 0,692 | 0,692 |

## 3. Détection du décalage à sévérité 5

| Régime | speckle AUROC(F) | speckle AUROC(H) | bright_down AUROC(F) | bright_down AUROC(H) |
|---|---|---|---|---|
| R1 — texte | 0,814 | 0,734 | **0,359** | 0,563 |
| R2 — few-shot | 0,977 | 0,249 | 0,845 | 0,433 |
| R3 — centroïdes | 0,977 | 0,387 | 0,844 | 0,539 |

## 4. Lecture

1. **L'inversion photométrique était une propriété des ancres textuelles, pas de F.**
   Sous R2/R3, brightness_down passe de 0,359 (sous le hasard) à 0,844–0,845 — dès
   16 exemples étiquetés par classe. Le mécanisme d'inversion documenté dans le papier
   (les images corrompues se rapprochent des ancres *texte*) ne survit pas à des ancres
   situées dans le nuage image : la modalité des ancres est la variable causale la plus
   probable.
2. **La faiblesse zero-shot venait des ancres** : balanced accuracy 0,517 (R1) → 0,597
   (R2) → 0,692 (R3) sur données propres. La limite « ancres quasi aveugles » du
   rapport exp01 §5.4 est levée expérimentalement.
3. **D_inter seule est trompeuse.** Les ancres textuelles sont les plus écartées
   (0,103) mais classifient le moins bien ; les centroïdes R3 sont les plus proches
   entre eux (0,007) mais les plus utiles. À D_inter il faut adjoindre la position des
   ancres relativement au nuage d'images (gap de modalité) — raffinement à formaliser
   dans le cadre théorique.
4. **H se dégrade comme détecteur sous R2/R3** (0,25–0,54, souvent sous le hasard) :
   des ancres dans le nuage rendent l'hésitation insensible — voire inversement
   sensible — au décalage. La dominance de F s'accentue.
5. **Coût de la classification sous corruption** : sous R3, l'accuracy chute avec la
   sévérité (0,692 → 0,385 sur speckle) — comportement attendu d'un vrai décalage,
   contrairement au zero-shot R1 qui s'améliorait paradoxalement (rapport exp01 §3.4).

## 5. Décision

- R2 (few-shot k=16) est le meilleur compromis coût/bénéfice observé : quasi
  identique à R3 en détection, avec 32 étiquettes source seulement.
- À élargir : autres corruptions, PneumoniaMNIST, et balayage de k (4, 8, 16, 32).
- À formaliser (cadre théorique) : critère de qualité d'ancrage combinant D_inter et
  distance ancres–nuage (gap de modalité).
- Chantier suivant : règle d'arrêt par échantillon (exp04), à croiser ensuite avec
  R2 (meilleure couche × meilleures ancres).
