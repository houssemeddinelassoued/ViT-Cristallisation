# Journal de bord

Une entrée par exécution. **Un chiffre absent d'un `outputs/<run_id>/metrics.json`
n'existe pas** — c'est la règle qui rend le protocole opposable plutôt que déclaratif.

Format : date · run_id · ce qui était testé · ce qui a été observé · décision.

---

## 2026-08-31 · socle initial

- Dépôt créé, noyau formel écrit, 10 tests verts, lint propre.
- Chaîne d'expérience validée de bout en bout en mode `--dry-run`.
- **Aucune mesure réelle.** Les runs `dry_run: true` présents dans `outputs/` sont
  des validations de tuyauterie sur embeddings synthétiques ; ils portent un champ
  `avertissement` explicite et ne doivent jamais être cités.
- Décision : passer à l'expérience réelle dès que `open_clip` et `medmnist` sont
  installés sur la machine cible.

## 2026-09-01 · première exécution réelle

| Champ | Valeur |
|---|---|
| Date | 2026-09-01 |
| run_id | 20260901T000620Z_breastmnist_c2a0cdaf |
| Modèle · poids | ViT-B-16-quickgelu · openai |
| Cohorte · corruption | breastmnist · medmnistc:speckle_noise (sév. 0, 1, 3, 5) |
| AUROC(F) à sévérité max | 0,836 |
| AUROC(H) à sévérité max | 0,748 |
| Prédiction confirmée ? | **Non** — F ≥ 0,70 ✓ et F−H = 0,088 ≥ 0,05 ✓, mais H = 0,748 > 0,60 ✗ |
| Décision | Réfutation partielle : F domine H mais H n'est pas au hasard. Observations annexes : à sévérité 1, AUROC(F) = 0,461 (sous le hasard) contre AUROC(H) = 0,619 ; la classification s'améliore avec la corruption (AUROC 0,636 → 0,784), signe d'un zero-shot de base quasi hasard (bal. acc. 0,517 à sév. 0). Prochaines étapes : tester d'autres corruptions et cohortes, renforcer les invites, envisager un modèle plus adapté au domaine médical. |

- Le run `20260901T000843Z_breastmnist_4aecb603` (figures sans `metrics.json`) est
  incomplet et ne doit pas être cité.
