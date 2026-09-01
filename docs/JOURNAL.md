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

## À remplir — première exécution réelle

| Champ | Valeur |
|---|---|
| Date | |
| run_id | |
| Modèle · poids | |
| Cohorte · corruption | |
| AUROC(F) à sévérité max | |
| AUROC(H) à sévérité max | |
| Prédiction confirmée ? | |
| Décision | |
