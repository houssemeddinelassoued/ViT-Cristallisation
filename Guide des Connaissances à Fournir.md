# **Guide des Connaissances (Knowledge Base) pour le Projet ViT Cristallisation**

Pour que Claude fonctionne comme votre **Co-Chercheur Principal** avec un niveau de précision digne des revues *Nature Machine Intelligence* et *IEEE TMI*, il est recommandé d'alimenter la section **Project Knowledge** avec 5 catégories de documents.

## **1\. Documents de Vision & Cadre Théorique (Déjà Prêts)**

Ce sont les fichiers stratégiques de votre projet. Ils définissent la vision globale et les équations piliers :

* **project\_overview.md** *(fourni)* : Contient la genèse du projet, la modélisation physique (énergie ![][image1], entropie ![][image2], répulsion ![][image3]) et le workflow à 3 phases.  
* **master\_plan\_vit\_medical.md** *(fourni)* : Contient la feuille de route scientifique, les baselines comparatives et le plan de rédaction par sprints.

## **2\. Papiers Scientifiques de Référence (Articles SoTA au format PDF ou Markdown)**

L'importation des articles fondateurs permet à Claude de faire des citations exactes, d'adopter le vocabulaire exact de la littérature et de situer votre apport par rapport au SoTA (*State-of-the-Art*).

### **A. Adaptation au Moment du Test (Test-Time Adaptation \- TTA)**

* **TENT** (*Test-Time Entropy Minimization*, Wang et al., ICLR 2021\) : Papier de référence pour la minimisation d'entropie sur LayerNorm.  
* **EOT / MEMO / CoTTA** : Papiers récents sur l'adaptation continue et la prévention du collapse latente.

### **B. Inférence Dynamique & Sortie Anticipée (Early Exiting)**

* **AdaViT** (*Adaptive Vision Transformers*, Meng et al.) : Modèle de référence pour l'arrêt précoce basé sur des portes d'attention.  
* **DynamicViT** (Rao et al.) : Élagage dynamique de jetons (Token Pruning) dans ViT.

### **C. Apprentissage Métrique & Contrastif en Imagerie Médicale**

* **Supervised Contrastive Learning** (Khosla et al., NeurIPS 2020\) : Fondement de la Phase 1 pour la structuration des centroïdes.  
* **Center Loss** (Wen et al.) : Concepts d'attraction intra-classe et de répulsion inter-classes.

## **3\. Spécifications Techniques & Codebase PyTorch Existante**

Si vous avez déjà du code ou des prototypes, ajoutez des extraits structurés :

* **Fichiers Python Noyaux (.py)** :  
  * Définition du squelette du modèle ViT-B/16.  
  * Implémentation des crochets PyTorch (*forward hooks*) pour l'évaluation couche par couche (![][image4]).  
  * Implémentation du calcul d'entropie latente en ligne.  
* **Fichier de Configuration (config.yaml ou constants.py)** :  
  * Hyperparamètres clés : Seuil de cristallisation ![][image5], taux d'apprentissage TTA ![][image6], facteur de pondération ![][image7], dimensions d'embedding (512).

## **4\. Fiches Descriptives des Cohortes & Jeux de Données Médicales**

Afin d'assurer la cohérence du protocole expérimental, fournissez une fiche descriptive des jeux de données utilisés :

1. **RSNA Pneumonia Detection Challenge** (Radiographies thoraciques DICOM, variabilité de contraste).  
2. **Camelyon16** (Histopathologie, variabilité d'exposition/staining inter-laboratoires).  
3. **ISIC Skin Lesions** (Dermoscopie cutanée, artéfacts d'éclairage).  
4. **ChestX-Ray14** (Classification multi-label sous fort décalage de distribution).

> **Exemple de contenu pour la fiche Data :** Nombre d'échantillons, résolution d'entrée (![][image8]), type de bruits/simulations de décalage appliqués (Bruit Gaussien, Flou de mouvement, changement de constructeur Siemens/GE/Philips).

## **5\. Templates de Rédaction Académique & Style Guide**

Pour générer du texte immédiatement publiable :

* **Fichier LaTeX Master (main.tex / Style IEEE TMI / Nature)** :  
  * Le préambule LaTeX avec les packages autorisés (amsmath, amssymb, booktabs, graphicx).  
* **Fichier de Bibliographie (references.bib)** :  
  * Votre banque de références BibTeX pour générer des citations valides.

## **Synthèse : Ordre d'Importance pour Démarrer**

| Priorité | Type de Document | Nom de Fichier Conseillé | Utilité Majeure |
| :---- | :---- | :---- | :---- |
| **P0 (Vital)** | Vue d'Ensemble & Master Plan | project\_overview.md, master\_plan.md | Ancrer les équations et les 3 phases |
| **P1 (Haute)** | Code PyTorch / Hooks | vit\_crystallization\_core.py | Guider la génération de code sans régression |
| **P2 (Moyenne)** | PDFs des Articles SoTA | TENT\_Wang2021.pdf, AdaViT\_2022.pdf | Comparaisons théoriques et citations précises |
| **P3 (Confort)** | Fichier BibTeX & LaTeX | references.bib, template\_nature.tex | Rédaction directe d'articles formatés |

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAYCAYAAAAh8HdUAAAA+ElEQVR4XmNgGCFAXFycW05OLkZeXv4EEP8D4jsKCgrh6OrAQElJiR+oYLWsrKwpTAzIrwXi/yADkNXCAAvQ9OkgBciCQP4UIH4NxE7I4jBJSyD+CcRfkcVBthsbG7Mii8EB0BYXqPtBTonGqRAZgBQBFfdANcHwc3R1GADoFDmgwv1A/A6ILwBtD0NXgwKAin4DFZUBmYzoclgBUIMhEB8QFRXlQZfDBViAGpYDcTS6BBSwQDECABVbyUOCupkBzWlAsWCsTgZKKALxE3lIaC1XVFTUl5GRUYVG9DegEmYUDTAAlPSEaoJjYFpbJS0tLYOudhQMCAAAplM92//CWzkAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADEAAAAYCAYAAABTPxXiAAACkklEQVR4Xu2WPWgUQRTHN0TBL/zC8/A+du/2DlESFTnEJoVFilgoglZ+YCUKik0kaWMRsIlFsLCwUQmKjZ0iBgwWNukEFQSLgCIIGhAT0GDi78/O6GS4Ey+c3BX7h8eb+b83b9/beTO7QZAiRYq/olarrS6VSsfCMDxSrVY3+naDrjiOd/pku9EVRdE8cofkT6BfIEvIjO9YLBZ74WeRN76tnVABl0h+SGMRjIdNEVPLXYOAXRoxtnHf1jaQzH692Uwms8Hhxk2iE64vxW2Bm0YWGPe5to4DSb5SEbTOUY+3xY26fEeCJOeQz8huj1dxi+xCv8t3Grorlcp2vW16/znJxox3OLKgQnK53DZ/YUdAb103E8lfYPwduaK5I2dMcefdddlsdj38ZLlc3ufyjYDf3kKhsNbnWwqSHCCpaR1il1eRyDzn5IDLC/qm8N3Y5PP1QPxr2lWfbyV03d7QNeob4A8j73iTWd/WDIjx6L8WYXZh0b1uDVTcBPbbHh/QGnvgH7AbNbUJ+hy+T+GuUnCEPgvXI1/GB7HNykctiv9W9OUouUgGkQ/ILT0f/dDE62N83H9uI6zC+R6y5BtM0CkCDmuu1mF83x5wxv0qwvH93Y75fL6gdrOxsM24O2FjE2udzp35nVEuY9JmzaTOnl3TEGFyE31E3vs2PVQPx+ctelA3F/qHtasAtwh3x7T2X4qwcwtirEEOsXZI9jrdsSJ0E3SzTwpNFjEnX1tIvSKUPDJg57Iz30V75V2/lqLJIr4a/9j6+0UwP2VvQZ0L2ZGTbpyWopR8V75Iwj9/wD+lmZ9Gf0K+Idflj34M/wR9V5cC+nWUfH9eai4fnTnN4S/iezNK2vyZLorlT0+RIkU9/AJ/b7dHtpD8PQAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAaCAYAAADxNd/XAAACW0lEQVR4Xu2WvWuUQRDG78UIih8Q9bx4X3vnHRx+oMUrNjYWNoJoEUvRMulEJaRIm0D+gCAhIGLhF5gyYGERLCyMIDYqWCnKIaKCYGEgxN94s+cx3CYX4nGHvA8M78w8u7Mzu7N7l0olSJCgCefcIeQrshqQFWShWCwetnP7CiRZl4QLhcL5NtywFEIRRyzXN9DdrpPkQcuBAbj7yAx6ZMm+gBawmE6nd1pOQGHj8Et8By3Xc8iuh9rHQwt4jxywXM9BcmfWaB9BBH+3bwsgqal12mdQ2gd5kslkdli+p9D2qXfQPtJiR8VGPyu2HRfAQLlcPiZfS/wT+GRC7ZPL5faWSqXnmvCfF6haraaxJ83QtqjVaruYPx063U1D22c1tACLjwqPvLFcJ5BTY+58KP6mIEEJvrhWO8B9R76QyAmx8/n8djk1Tuyi2nso8oKcYKVSKaCPuL8XPcJ+iv0W/pKM9XE5xd34b0ocicm406LjG0a/gq/qx7ZDpK1xTXf3hyzqRRbj+1I4xgzZyfi2wS14W+c9Smmfo38kRiy63p/mAyHJ4nuM3FY+hv/cMlZew1PIuVToR9Ot///nEwFmuXw1O9eDIu543TWKHvc2+odQAcQ87hob9gKZc43n+VfL2CUfp6vYYAHNX3Dd8Z+t4z18sdbfFXRaANx1pz+A+C6r7wbyTodH+K/qvO4XwEXdzyLPkBWKeEhPn0R/jSyTwAO+Y67Rht/gR2kZVPcKuce9y0uMOI63Yk8gtyQGMqJzl2WuzLPr9hpbstnsPuukuIw8CNafIEGC/xy/AfAnu8Q9gXMZAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAaCAYAAAAkJwuaAAAChklEQVR4Xu2Xv2sUQRTHd7kIBn9ENOeR+7XeDzhSiMUhqFimsNFCLSxMZWEt0Vha+Q+IsUgvsUjaqJDKFAYFO9MJxsZKAoKCkeT8PG4mjC97GzfCLgfzgS+7896b2d23b3Zmg8Dj8Xg8Q8+ZPoe1/aAw1j10QtuFdrt9PIqiO2hejpVK5ZSOGSp4iB7a5IHfyXmxWDyqY/6Ver3+0Iy3hd6iDTSh47Cto0umGZqE7qAbfwUOIyTh6v8m0oXxugmJ3ML/qNvtHpJ2tVodpf0K+1qz2RzT8UNFhoksyHXQL3TRGk01f280Gufc4D3wDagSuI0+EMwheoA2zaAz9u3kRYaJDFqt1ml9HeIeo6/0a7p2zQhBz2q12nmOPwl+jWalpAPzhvhG3dKdXGQRkJtKI7qFepxBZJlIjUxn4tbQAs0R7d+FgEm0zOCXOf5G28rfk9J2bRpiZoj5kkZpVsI8E0nsbbm2zFTti4Xg59IBXbc2U2k7DDblxmZNXonEv4I+oUntGwjBH9E3t5Oco89M84obmzV5JLJUKh3B/17WD+1LJOpP69VOp3PM2qjIu9ieBvt8z6TsubmbaUS3gh5nEFknUhZXfPMU0ElrM+2zblwscqPoiW3b/RPJvCLtesJ3koXqmlwojQb9VcSRcSJD/LP4ll0j9/tm3+osl8vjkZrWdhXntGAGnnO6ZII8qNF9SSQPc0HaskWxMSYpPbQi09Gx/XBt7s5CdiHiNwUw4W7vov42UMbTWgySVm1BSlaqz2x5rG2UzktoFd+LPHb1MQ9jtWFjzPZE/kamle2la7NVHSdJvAmTreAev9HubE0ilDemjRCabUriN9Lj8Xg8Ho/H4/EcjD+8iPwypNso8wAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHMAAAAaCAYAAACEuGN0AAAE80lEQVR4Xu1YTYiVVRi+wygUKtrPeHPmzj3fzFSDuJIxRfwNihZmtDCMNBRcJDKL8HcTgQsJXbgYBWFUxIVk1iKpdiIXFIsK2mgT/YCFNjgDiQsFG8br89zzvtf3nr575xuaO5GcB17OOc95z/u957zn98vlIiIiIiIiIv63SJJkv3NuEHIIxZawPgvQ9p1isfhzwG3q7OxcBftPKNfV1ZUH12P1IqYAGNQ3MOBjORNAlMchu4zahECwzqFNGfJ7wJ8W3soRqzNdoC+YbOtD/rFAd3f3XAzsN5Bhy6P8HWSIK8jyjQD9PyFjdYJ5hisfA/nWZGxONeDHpcc2mOjYSgnAJcuj/BlXEFet5euhUCg8j2BtQZu7acHEd/ZZ7r8C/WtqMPP5/CzOWMgfIu+GOs0COjfAoEFKltetkanl0wB/N0PvPNK+RsFkPxlwrMzVfX19M61OCOifdX6ll6GPxPVDbopP5yDPWX2U11Cf44f0L8geTLAnTf179IvtWS96h42JFuEuQvcTpD9ALrS1tc02Oo2BhhvR6A4/BGMbUH4T+QWhnqKnp2c+67PKRFsaB9o1CGbIh4BeAp3L8H1Ro2CC+xryI2Sb8wM1anVCoM1a2PvI+bP7K+Q/ZHDAbxW/PlZd9hHl32hbqBbRGdRJg3bLOL7kkT8oY71MbXAHsmXonYCUJhPMykc5Czlrw8o0QH+IMyirsJOhDYt6QavHh0D9SehuZ77YOJgXEYynWe7o6HgG3LfgFlu9EM5PSK6mEyjOIIe2BZRv2G/AbgfK1yH9pi3P/DH4tFI54Tne/9hm+Q3a1jKC+xK4L7IGk8v6FBqMSsonwSC3oFCxmagXtHq8hWx/1YDUC2YaEr/tDaNNd1incBJM2g05fsfqEpwkDJTzR8c9+h8GLo0j+HRincjfkHWgW0O9VMDgU87PHm4/lUDSCdwuXwx1mwl884B0oGR5DSZTyxvMQB+OIaBLnWzp0H3N+UHk2cZy5V2J/Mss28Yy6LyMVAMVQuxmCibHDdwDrvjEX8Q+dSmBS+MUqBtivQr7N9HZXoFx9JWwrhHQZh3abJiEpDqugL1N4nzJ8mZlHrC8Qi5tZ4u1WzovHjoQ5Cp9E+4qz3ttT7/cFAVTJtRt+LwxJ29l9T/sv+WQ74deYutRnEe/xedxlNfa+lS0t7c/C+VrjTqTBue3EF3JmSS0YcHvO781XrM8yhcgmZ8mhASo7Gq3Wb2MXOfZZnT3hVwIly2YelyVmTd6Z8iJT3oxqgkm0s/VdrgDgd8sutmeVFBe7GovBgUYfd9eqacDcPhV+DGeezQYDAC3rMrFRnQ4+GXITuUMWnheoW636Iy42m32e7R/XZWRXwHuLs/cRyZqQXuy4kZg523k87SnHGRMVnqr+sYfINK8FeX75KRuQO2Kf0dzvo9fahsGM6ld2VtRf4PvZ207IRJ/q7uX+LfNLUh/pn16asGO7YAP5zlwyB+HfGD9wApdAm407UznjQ91JRmoqugKQH45ZJg6/Ibzz40roR2LFHuVx37AcYUukCfLQeRvc7Uh/RXcC0ivUs9OmqJ/vz6AjLBPyjOYzv8J+0nyvyBdo/WZgUbz6FQu6+2pSZCtZYBpWPdvIWfseshefGNhrgl95RhSEtkROBl117Mw411Fb2/vnKCuumVHREREREREREREREREREwnHgKkxvKLuAVI9AAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFEAAAAaCAYAAADPELCZAAACxklEQVR4Xu2Xz2sTQRzFN7RCRVFRY2h+7KZJDoqCSBBBxIsiXvQkRdCbBz2KBf0LPHgoiBQLIhQPRZTePAheDHgQRPAkiCAYL55UFJWKlPp57sRORjduhI21zoNHdt58Z+c7b2ZnJkHg4ZElwjA8MDY2tt/VPVIiiqIpuIiRF9w6j5SoVquXMLG90kzM8WkdcsUOGPAonIDX4F6kITcmLUqlUhkTt64IE40xbfiMwbzS5+XGCOi7ZF6z2VylMgbsofyJNtMUh53wnqDNQdqf7fT9z5tog4HdSDBxGP0mg93naHPwXaVS2SGB51MyOomKqdfrW3i+WqvV1v9XJjLIGvprfYK2jnZR8f2YQPw52DLGzsIv8DF9b3BjE2Hcn4Rv4QK868b8LSSZiH5aunK3dZln9JatpwXtm1G8JaSehIBlf1SN4KzK5XJ5I88P2MwjlZnpTVru3a2WwGBGzCSkJs1y7nuSkGSiZVaSiW1bT4NCobAmiheS2n/sNe4uELxAxzOdzVkg8ePoUzzmzIY7YjXpAnET2vz7oSbGfU8SBmninyAno+Bc4JxkSgy+IaFxfu/bdYPGsjaxWCxupqOn2lvcOmPiVxJ6Dk+69YNEkoloJ35jYsvWM0G4tIHaV4TvMCYqkTvskavdehvaO3nHsX4Y9HEhTjJReWuie5ioLyxbMPiddPTBvSIIxsTPHDq73ToX5mD66f7Vi9U+rg5JJjK5JfSXcJutU76ueJlp65mBBM/Q2fnAnJZ0Pkp5mt+HJpEjsNZoNNY5TbNGToePVjk53lMuelZ+9gSQ23bKt61/LIej+KC8rHf8eFuWUOd0Og+fkMCtMN4Dx6kaCuO/Wy+kue2yRj6fXxvFl99Fl1qZdizae7RHUbxHzpPvjK4qdkzmqFr3vF/pGpCtLzfIMH0x5HqFe13Frffw8PDw8PDw8PAYNL4BdLn+1CIoiroAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAbCAYAAABIpm7EAAAA8ElEQVR4XmNgGNSAWU5OzkZRUdFMQUGBA10SA4iKivIANUyQl5f/BcLo8jgBUFM5UMN/GRkZTnQ5rADkHKCGrUCcgy6HEwAVRwPxAZAz0eWwAqCzlIAavgJpY3Q5XIAF5A+g8xrQJXACoIYnQPwTiC3R5bABRqDCvyBbgLgVXRIdgBTnAPEUqIbTQL8IoiuCAZjir8B4kAbSD4D4NygFoCsEhX8C1BmzjI2NWaFiHkD+PyC+iq4e5uY9SkpK/DBBYLoSB4pdB2lCVgwKkWCgtbeACuRRJCByk4D4P5DJCBcEWi2AUIIJQPJEpeBRMHgBAH+5NPzn/natAAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFcAAAAYCAYAAACPxmHVAAADf0lEQVR4Xu1XPWiUQRC9Q4WIgn+EqGdu7+IPSASVE22sxE60iIUBRQtthIBFQLETbFOICDGihGCpjYWKGogS0RSWQkACIoiFIViliZj43n0zcZh8X+5yiQnKPhi+3ZnZ2X1zsz+Xy0VERERE/E/IFwqFLaVSqckb/mFUOXllKiqVypoQwlC5XN6P7zbIGGQG8tz6of8Z0oXAO4Dd4jMF6bB+FsVi8RP98D3pbX8L5IP5eslJ+HQop7a2tg3q19raekg5kY+MIadMPshRUE7elgpUVSeCF7SP9i4M/iYB8tRxUVyE9gn0b8pixlVnIT8a7cuaXPLBnB/TOGEdt3LCAf0RxykvnMbh165jLaDvV07elgosZgDOo6xIUXGSOxKkTAWCVqTfrePwy+9D/0fWRBhzDrYPtC9zcsmHax01auX0VTnhO5nBaQYxbsyONACPN8rJ21KBqtyDYJdZaapDkGuSlAr7tMGnj77qE5It98VPhCpZC98nktzXEqdmcjFuM3wH7ToMmJzqkeQNHsKnj5ysXjhNKifaMzgxuQN/Rv7h1NzcvF45WftCsBqDHzOAPaM8sMij8PnpJ0K/G/JAjoW6k0vI8dPjEszEMuaQ0S0UymmkDk5MbqfVK6dFJxeBj4Rky/zyNgs9f+D/0uqhe8uDX9oLSi4B/2fw75UEr0L869ANsrK9b71QTtj2p7xNIRdhP/kwiarHuIPKaVHJlWqbwiTnc+by8sBEh0kYC9lq1HmMu2qrrpHkEi0tLesw7iETPV+l1YLwuWc4pUL48P4YdCZyuqucFpPcanIw8Jg3WPAXBEaZAKfnAoetbqWTK3y+1+JEPqzaNE72nG80uZrYMVWgKpsgJeMzm9iSuU3RPsBvUS7BLCm5SyILcu4+gtyH9EBeNZhgntV83+5VBdpdaZzIR6uTfOB3ke0l4QTH9wjUHpLbsioYeAG6TeqD9gvIFZxb29UH7Z3BVatFkNdEvZW7VBcafE+Tk+Uj8k45cS7l5HyGIWd9TELsc15ImYDjCTqnCauXPkJ6jl1k2seU7cOFcEsyDp89G72fxVI9xULCR9+wXp4qp5DsDm+nTCPhx31ccpKjr8opJAU4L6fZZ1eaqFNRnigZMmEDiv+c7VRrC8F+JiT/mLLABF/ySo8wDx/IbeNXfUamyEQwR4miEU4RERERERERERErgt9RGo0OXldNoQAAAABJRU5ErkJggg==>