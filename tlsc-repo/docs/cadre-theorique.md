# Cadre théorique canonique

Ce document fixe les notations implémentées par `tlsc/core/gibbs.py`. Le protocole
expérimental autoritatif est `../02protocoleexperimentalv3.md` à la racine du workspace.

## Représentation et ancres

Pour un embedding image L2-normalisé $z \in \mathbb{R}^d$ et des ancres textuelles
L2-normalisées $\mu_k$, l'énergie de configuration est

$$
E_k(z) = \lVert z - \mu_k \rVert_2^2 = 2 - 2\langle z, \mu_k\rangle.
$$

La distribution de Gibbs à température $T > 0$ est

$$
p_k(z) = \frac{\exp(-E_k(z)/T)}{\sum_j \exp(-E_j(z)/T)}.
$$

Pour CLIP, les logits sont $\langle z,\mu_k\rangle/\tau$. Les deux distributions
sont donc identiques pour $T = 2\tau$.

## Observables

$$
H(z) = -\sum_k p_k(z)\log p_k(z),
\qquad
F(z) = -T\log\sum_k \exp(-E_k(z)/T),
$$

$$
\langle E\rangle(z) = \sum_k p_k(z)E_k(z),
\qquad
\chi(z) = 1 - \frac{H(z)}{\log K}.
$$

L'identité thermodynamique utilisée comme invariant logiciel est

$$
F = \langle E\rangle - TH.
$$

## Prédiction falsifiable de l'expérience 1

Une translation uniforme des énergies $E_k \mapsto E_k+c$ laisse $p$ et $H$
inchangés, mais produit $F \mapsto F+c$. Sous un décalage d'acquisition qui éloigne
les images de toutes les ancres sans modifier fortement les écarts relatifs entre
classes, $F$ doit donc mieux détecter le domaine cible que $H$.

Le critère pré-enregistré de l'expérience 1 est évalué à la sévérité maximale:

- $\operatorname{AUROC}(F) \ge 0{,}70$;
- $\operatorname{AUROC}(H) \le 0{,}60$;
- $\operatorname{AUROC}(F)-\operatorname{AUROC}(H) \ge 0{,}05$.

Ces seuils sont des critères de décision, pas des résultats. Toute exécution doit
conserver la configuration, les scores individuels, l'environnement et le commit Git.

## Limites de portée

Cette proposition ne garantit pas que tout décalage augmente $F$, ni que $H$ reste
toujours au hasard. Sa validité empirique dépend du modèle, des invites, de la cohorte
et du type de corruption. Une réfutation correctement exécutée est un résultat.
