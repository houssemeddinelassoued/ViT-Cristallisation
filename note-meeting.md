 (idée générale VIT dans CLIP):

on a une Image (box 1) -> TECHNIQUE d'emboitement (VIT) (sans aucune sémantique, qui tient compte de la localité) -> Vecteur (512) -> Classifieur

avec des couches de 1 à 12,

Image médicale : très spécifique,

il faut trouver une certaine fonction ou transformation, qui va transformer l'image en vecteur V,

V = une certaine fonction f(I) : fonction paramétrique qui se base sur ViT, et ce qui change dans f est son paramètre et le nombre de couche de l'encodeur, on cherche à optimiser, argmax (V),

- soit je choisi la fonction f qui maximise l'accuracy (ou les autres )

- soit je choisi la fonction f qui maximise la distance d (l'espace entre les centroïdes) ( je calcule le centroïde dans les différents Classes (0 et 1) ,

Formalisme ceci mathématiquement, 

Comment rendre ceci  applicable :
- Offline : cas par cas 
et Online : dynamique selon les images.

Il faut trouver comment optimiser f en fonction des images dynamiquement, et minimiser l'entrepie : agrmin H  sur N, et on cherche le nombre de couches N qui réduit H avec les autres paramètres de ViT (p1, p2, p3 ... à identifier) , ce qui donne le meilleure emboitement possible.


Adapte la formulation,

L'entropie dans le sens physique de terme : le désordre dans l'hyper espace latent , on vise à maximiser la distance entre les classes et minimiser l'entropie

C'est une analogie physique extrêmement pertinente. En considérant l'espace latent (votre hyper-espace à 512 dimensions) comme un système thermodynamique, le processus d'optimisation du réseau de neurones s'apparente à une "cristallisation" ou un refroidissement.Voici la reformulation de l'architecture ViT sous l'angle de la physique statistique et de la dynamique de l'espace latent.L'Hyper-Espace Latent comme Système PhysiqueDans son état initial (un réseau ViT non entraîné ou face à une image totalement inconnue), les vecteurs $V$ générés sont éparpillés aléatoirement dans l'hyper-espace $\mathbb{R}^{512}$. C'est un état de haute entropie : le désordre est maximal, la densité est uniforme, et il est impossible de séparer les classes.L'objectif de la fonction paramétrique $f_{\theta, N}$ est d'agir comme une force structurelle qui modifie la géométrie de cet espace pour atteindre un état de basse entropie (hautement ordonné).Ce processus obéit à deux forces antagonistes mais complémentaires, que l'on peut formuler comme un problème de minimisation d'énergie libre :1. La Force Répulsive : Maximisation de la Distance ($D_{inter}$)Les classes différentes (ex: sain vs pathologique) agissent comme des pôles magnétiques de même signe. La fonction d'optimisation cherche à repousser leurs centres de gravité (les centroïdes $\mu_0$ et $\mu_1$) le plus loin possible.$$D_{inter} = \Vert{}\mu_0 - \mu_1\Vert{}^2$$2. La Force Attractive (Condensation) : Minimisation de l'Entropie ($\mathcal{H}$)La minimisation de l'entropie réduit le "chaos" ou la variance à l'intérieur d'une même classe. Les vecteurs $V$ d'une même classe s'effondrent gravitationnellement vers leur centroïde respectif.Si l'on assimile la distribution des vecteurs à une probabilité $p(V\vert{}k)$ pour la classe $k$, minimiser l'entropie de Shannon force la distribution à devenir un pic de Dirac (certitude absolue) :$$\mathcal{H} = - \sum_{k} p_k \log(p_k)$$La Formulation Unifiée : L'Équilibre ThermodynamiquePour adapter dynamiquement le nombre de couches $N$ et les paramètres $\theta$, nous cherchons l'état d'équilibre du système. Le réseau cherche les paramètres optimaux pour minimiser une énergie globale $\mathcal{E}$ (la perte) :$$\theta^*, N^* = \arg\min_{\theta, N} \left( \mathcal{H}_{intra}(\theta, N) - \lambda \cdot D_{inter}(\theta, N) \right)$$Où :$\mathcal{H}_{intra}$ est l'entropie (le désordre) que l'on veut écraser.$D_{inter}$ est la distance inter-classes que l'on veut étirer.$\lambda$ est un paramètre de température qui gère le compromis entre ces deux forces.$N$ (la profondeur du ViT) agit comme le "temps" ou la "durée" d'application de ces forces. Plus $N$ est grand, plus le réseau a l'opportunité d'appliquer ces déformations spatiales, jusqu'à ce que l'entropie soit suffisamment basse ($\mathcal{H} < \epsilon$).