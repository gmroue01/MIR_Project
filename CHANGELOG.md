# Changelog — MIR Image Search Engine

Chaque évolution décrit : **le changement**, **pourquoi**, **les concepts théoriques mobilisés**, et **les fichiers concernés**.

---

## v1.0 — Architecture initiale du moteur de recherche par image

### Changement
Mise en place complète du système MIR (Multimedia Information Retrieval) : backend FastAPI, frontend React, pipeline d'indexation, moteur de recherche, et évaluation.

### Pourquoi
Construire un moteur de recherche d'images de voitures permettant de retrouver des images similaires à partir d'une image requête, en combinant plusieurs types de descripteurs visuels et plusieurs mesures de similarité.

### Concepts théoriques
- **Descripteurs visuels** : représentation compacte du contenu d'une image sous forme de vecteur de features. Deux grandes familles :
  - *Descripteurs classiques* : basés sur des propriétés statistiques ou géométriques calculées manuellement.
  - *Descripteurs deep learning* : vecteurs extraits des couches internes de CNN/ViT pré-entraînés.
- **Indexation** : pré-calcul hors ligne de tous les descripteurs pour chaque image de la base. Évite de recalculer à chaque requête.
- **Recherche par plus proche voisin (kNN)** : pour une requête, calculer la distance entre son vecteur et tous les vecteurs de la base, puis trier.
- **Normalisation L2** : mise à l'échelle des vecteurs pour que leur norme soit 1, rendant la distance euclidéenne et la similarité cosinus cohérentes.
- **Métriques d'évaluation retrieval** :
  - *Precision@K* : proportion de résultats pertinents parmi les K premiers.
  - *Recall@K* : proportion de documents pertinents retrouvés parmi les K premiers.
  - *Average Precision (AP)* : moyenne des précisions à chaque rang où un résultat pertinent est trouvé.
  - *MAP (Mean AP)* : moyenne des AP sur toutes les requêtes — métrique standard en retrieval.
  - *R-Precision* : précision au rang R où R = nombre total de documents pertinents.

### Descripteurs implémentés

| Descripteur       | Type        | Dimension | Principe |
|-------------------|-------------|-----------|----------|
| Color Histogram   | Classique   | ~512      | Distribution des couleurs en HSV |
| HOG               | Classique   | ~8100     | Gradients orientés sur une grille de cellules |
| SIFT              | Classique   | variable  | Points clés locaux invariants à l'échelle et la rotation |
| ORB               | Classique   | variable  | Points clés binaires (FAST + BRIEF), descripteur Hamming |
| MobileNetV2       | Deep        | 1280      | CNN léger via timm, features avant la tête de classification |
| ResNet50          | Deep        | 2048      | CNN résiduel via timm |
| ViT-Base          | Deep        | 768       | Vision Transformer, token CLS |
| DinoV2            | Deep        | 384       | ViT-Small auto-supervisé, token CLS |

### Mesures de similarité implémentées

| Mesure            | Usage recommandé |
|-------------------|-----------------|
| Euclidienne       | Vecteurs normalisés L2 |
| Cosine            | Vecteurs de directions |
| Chi-carré         | Histogrammes / distributions |
| Jensen-Shannon    | Distributions de probabilité |
| Hamming           | Descripteurs binaires (ORB uniquement) |

Chaque mesure existe en version **scalaire** (deux vecteurs) et **vectorisée batch** (requête vs matrice entière) pour la performance.

### Fichiers créés
| Fichier | Rôle |
|---------|------|
| `app/indexer.py` | Calcul hors ligne de tous les descripteurs → `indexes/*.npz` |
| `app/searcher.py` | Chargement des index, recherche kNN vectorisée, cache mémoire |
| `app/similarity/measures.py` | 5 mesures × 2 formes (scalaire + batch) |
| `app/evaluation/metrics.py` | P@K, R@K, AP, MAP, R-Precision |
| `app/descriptors/color_histogram.py` | Histogramme HSV |
| `app/descriptors/hog.py` | HOG via OpenCV |
| `app/descriptors/sift.py` | SIFT via OpenCV |
| `app/descriptors/orb.py` | ORB via OpenCV |
| `app/descriptors/deep_model.py` | Classe générique d'extraction deep (timm) |
| `app/descriptors/mobilenetv2.py` | Wrapper MobileNetV2 |
| `app/descriptors/resnet50.py` | Wrapper ResNet50 |
| `app/descriptors/vit_base.py` | Wrapper ViT-Base |
| `app/descriptors/dinov2.py` | Wrapper DinoV2 |
| `app/main.py` | API FastAPI : `/api/search`, `/api/map`, `/api/images`, `/api/classes` |
| `frontend/` | Interface React : SearchPanel, ResultsPanel, BenchmarkPanel, MapPanel |

---

## v1.1 — Réduction PCA pour les mesures de distribution

### Changement
Ajout d'une réduction dimensionnelle par PCA pour les descripteurs à haute dimension (HOG : 8100 → 256 dimensions), appliquée uniquement quand la mesure de similarité est Jensen-Shannon ou Chi-carré.

### Pourquoi
Jensen-Shannon et Chi-carré traitent les vecteurs comme des **distributions de probabilité**. Sur des espaces de très haute dimension (HOG à 8100 dims), ces mesures deviennent numériquement instables et très coûteuses. La PCA réduit la dimension tout en conservant la variance principale, stabilisant le calcul et accélérant la recherche.

### Concepts théoriques
- **PCA (Principal Component Analysis)** : décomposition en valeurs singulières (SVD) de la matrice centrée. Les premières composantes principales capturent la variance maximale des données. Projeter sur les `k` premiers vecteurs propres réduit la dimension de `d → k` avec perte minimale d'information.
- **Malédiction de la dimensionnalité** : en haute dimension, les distances perdent leur pouvoir discriminant — tous les points tendent à être équidistants. La PCA atténue ce phénomène.
- **Ajustement lazy (fit-on-demand)** : la PCA est calculée à la première utilisation puis sauvegardée sur disque. Les requêtes suivantes chargent les composantes depuis le cache disque puis mémoire.

### Fichiers créés / modifiés
| Fichier | Changement |
|---------|-----------|
| `app/pca_reducer.py` | **Créé** — fit PCA (SVD), sauvegarde `indexes/{name}_pca256.npz`, projection en ligne |
| `app/searcher.py` | `_prepare_descriptor()` applique PCA si `measure in {"jensen", "chi_square"}` et descripteur dans `PCA_TARGETS` |

---

## v1.2 — Pipeline d'entraînement metric learning (Triplet Loss)

### Changement
Ajout d'un pipeline complet d'entraînement de modèles de deep metric learning sur Google Colab, avec fine-tuning partiel des backbones timm et triplet loss avec hard mining en ligne.

### Pourquoi
Les descripteurs deep pré-entraînés sur ImageNet ne sont pas optimisés pour discriminer des modèles de voitures spécifiques. En fine-tunant avec une loss de metric learning, on apprend un espace d'embedding où les images d'un même modèle de voiture sont proches et les images de modèles différents sont éloignées.

### Concepts théoriques
- **Metric Learning** : apprendre une fonction d'embedding `f(x)` telle que `d(f(x), f(y)) < d(f(x), f(z))` si `x` et `y` sont de la même classe et `z` d'une classe différente.
- **Triplet Loss** : pour un triplet (ancre `a`, positif `p`, négatif `n`) :
  `L = max(0, d(a,p) − d(a,n) + margin)`
  Pénalise les configurations où le négatif est plus proche que le positif (+ une marge).
- **Hard Mining en ligne (Batch Hard)** : pour chaque ancre dans le batch, sélectionner le positif le plus difficile (distance max) et le négatif le plus difficile (distance min). Plus efficace que le mining aléatoire.
- **Référence** : Hermans et al., *"In Defense of the Triplet Loss"*, 2017.
- **Hypersphère unitaire** : normalisation L2 des embeddings (`F.normalize(emb, p=2, dim=1)`). Sur l'hypersphère, la distance euclidéenne et la similarité cosinus sont équivalentes : `||a−b||² = 2 − 2·(a·b)` quand `||a||=||b||=1`.
- **Fine-tuning partiel (partial freeze)** : on gèle les premières couches du backbone (features génériques) et on dégèle seulement les dernières (features spécifiques au domaine), pour éviter le catastrophic forgetting avec un petit dataset.
- **Projection head** : couche linéaire `feat_dim → 512` ajoutée après le backbone. Découple la dimension d'embedding de la dimension interne du backbone.

### Stratégies de freeze par architecture

| Modèle | Couches dégelées |
|--------|-----------------|
| MobileNetV2 | 3 derniers blocs invertis + conv_head |
| ResNet50 | layer3 + layer4 |
| ViT-Base | 4 derniers blocs transformer + norm |
| DinoV2 | 4 derniers blocs transformer + norm |

### Fichiers créés
| Fichier | Rôle |
|---------|------|
| `app/training/models.py` | `MetricModel` : backbone timm + projection head + normalisation L2 dans `forward()` |
| `app/training/triplet_loss.py` | `pairwise_distances()` + `batch_hard_triplet_loss()` avec batch hard mining |
| `app/training/dataset.py` | `CarDataset` + `build_splits()` (split stratifié 80/20) + transforms |
| `app/training/trainer.py` | Boucle d'entraînement : AdamW + CosineAnnealingLR + évaluation Recall@1 |
| `app/training/integrate_weights.py` | Chargement des poids entraînés dans l'indexeur de production |
| `colab_training.ipynb` | Notebook Colab prêt à l'emploi |

---

## v1.3 — Proxy-Anchor Loss + Balanced Batch Sampler

### Changement
Remplacement de la triplet loss par la **Proxy-Anchor Loss** et ajout d'un **sampler balanced par classe** pour l'entraînement. L'ancienne triplet loss reste disponible via le paramètre `loss_type="triplet"`.

### Pourquoi
L'entraînement avec triplet loss hard mining stagnait autour de la margin après ~10 epochs sur MobileNetV2. Cause principale : une fois le modèle partiellement convergé, la majorité des triplets devient "trop facile" (perte nulle), les gradients s'annulent et l'entraînement cesse de progresser. La Proxy-Anchor Loss résout ce problème structurellement en comparant les embeddings à des proxies apprenables plutôt qu'entre eux.

### Concepts théoriques

**Proxy-Anchor Loss** (Kim et al., CVPR 2020)
- Un vecteur **proxy** `p_c` (paramètre apprenable, L2-normalisé) est associé à chaque classe `c`.
- La loss compare chaque embedding à **tous les proxies** du batch, pas à d'autres embeddings :

$$\mathcal{L} = \frac{1}{|P^+|} \sum_{p \in P^+} \log\!\left(1 + \sum_{x \in X_p^+} e^{-\alpha(s(x,p) - \delta)}\right) + \frac{1}{|P|} \sum_{p \in P} \log\!\left(1 + \sum_{x \in X_p^-} e^{\alpha(s(x,p) + \delta)}\right)$$

  - `s(x, p)` = similarité cosinus entre embedding et proxy
  - `α` = facteur d'échelle (défaut : 32) — contrôle la pente de la sigmoïde
  - `δ` = marge angulaire (défaut : 0.1)
  - `P+` = proxies ayant au moins un positif dans le batch
  - Terme positif : tire les embeddings vers leur proxy de classe
  - Terme négatif : pousse les embeddings loin de tous les autres proxies

- **Avantages vs triplet loss** :
  - Pas d'explosion combinatoire des triplets (O(N³) → O(N·C))
  - Convergence ~3× plus rapide
  - Gradients denses à chaque step (pas de triplets "easy" qui annulent tout)
  - Les proxies servent de "prototypes de classe" apprenables

**Balanced Batch Sampler**
- Chaque batch contient exactement `n_classes × n_samples` images (ex. 16 classes × 4 images = 64).
- Garantit la diversité de classes nécessaire pour que la Proxy-Anchor Loss reçoive des signaux utiles de tous les proxies à chaque step.
- Avec un `DataLoader` shuffle classique, certains batchs peuvent être dominés par une seule classe.

**Taux d'apprentissage différentiel pour les proxies**
- Les proxies sont initialisés aléatoirement → ont besoin d'un LR plus élevé pour converger rapidement.
- `lr_proxies = lr_backbone × 100` (ex. `1e-2` vs `1e-4`).

**Partage du `class_to_idx` train → val**
- Correction d'un bug latent : `CarDataset` construisait son propre `class_to_idx` indépendamment pour train et val. Si une classe était absente d'un split, les indices entiers divergeaient. Le val set utilise maintenant le mapping du train set, garantissant que les indices correspondent aux proxies.

### Fichiers créés / modifiés
| Fichier | Changement |
|---------|-----------|
| `app/training/proxy_anchor_loss.py` | **Créé** — `ProxyAnchorLoss(nn.Module)` avec proxies `(C, D)` apprenables |
| `app/training/dataset.py` | **Modifié** — `CarDataset` accepte `class_to_idx` optionnel ; ajout de `ClassBalancedSampler` |
| `app/training/trainer.py` | **Modifié** — paramètre `loss_type`, sampler balanced, LR différentiel proxies, checkpoint enrichi |

---

## v1.4 — Régularisation configurable, planificateur cosinus et early stopping

### Changement
Trois hyperparamètres de pilotage de l'entraînement rendus explicites et configurables : le **weight decay**, le **plancher du cosine decay** et la **patience pour l'arrêt précoce**. Ces valeurs étaient soit absentes soit codées en dur.

### Pourquoi
Des hyperparamètres non configurables empêchent toute recherche de la configuration optimale sans modifier le code source. Le weight decay et le cosine decay influencent directement la généralisation et la convergence ; l'early stopping économise du calcul et protège contre le sur-apprentissage en arrêtant dès que la métrique de validation cesse de progresser.

### Concepts théoriques

**Weight decay (L2 regularization)**
- Pénalise les poids de grande magnitude en ajoutant `λ||W||²` à la loss totale.
- Dans **AdamW**, le weight decay est découplé de l'adaptation du gradient (différence clé avec Adam + L2) : la mise à jour devient `W ← W − lr·(grad_adapt + λ·W)`.
- Améliore la généralisation en contraignant les poids à rester petits.
- **Les proxies de la Proxy-Anchor Loss sont exemptés** (`proxy_weight_decay=0.0` par défaut) : ils sont re-normalisés L2 à chaque forward, donc un weight decay les forcerait vers zéro, en contradiction directe avec la normalisation.

**Cosine Annealing LR**
- Le taux d'apprentissage suit une demi-période de cosinus de `lr_max` à `eta_min` :

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\frac{\pi t}{T_{\max}}\right)$$

- Descend progressivement vers un plancher `eta_min = lr × lr_min_factor` (défaut : 1 % du LR initial).
- Avantage : évite un LR trop faible en début d'entraînement (qui ralentirait la convergence) et trop élevé en fin (qui empêcherait la convergence fine).

**Early Stopping**
- Surveille le **Recall@1** sur le jeu de validation à chaque époque.
- Un compteur `patience_counter` s'incrémente chaque fois qu'aucune amélioration n'est constatée.
- Si `patience_counter ≥ patience`, l'entraînement s'arrête et le meilleur modèle (déjà sauvegardé) est conservé.
- Évite d'itérer inutilement sur des époques qui ne font plus progresser le modèle, ce qui peut même dégrader la généralisation (sur-entraînement tardif).

### Fichiers modifiés
| Fichier | Changement |
|---------|-----------|
| `app/training/trainer.py` | Ajout des paramètres `weight_decay`, `lr_min_factor`, `proxy_weight_decay`, `patience` ; suppression des valeurs codées en dur ; logique d'early stopping dans la boucle d'entraînement ; hyperparamètres sauvegardés dans le checkpoint |
| `app/training/proxy_anchor_loss.py` | Ajout du paramètre `proxy_weight_decay` (défaut `0.0`) stocké comme attribut ; le `trainer` lit `loss_fn.proxy_weight_decay` pour configurer le groupe AdamW des proxies |

---

## v1.5 — Résolution d'entrée configurable

### Changement
La résolution des images d'entrée devient un hyperparamètre explicite (`img_size`). Auparavant, 224 px était codé en dur pour tous les modèles sauf DinoV2 (518 px). Il est maintenant possible d'entraîner à n'importe quelle résolution (ex. 320, 384) sans modifier le code source.

### Pourquoi
La résolution a un impact direct sur la qualité des représentations, en particulier pour les architectures de type Vision Transformer dont les patches captent plus de détails à haute résolution. Des voitures sont des objets où des détails fins (logos, grilles, feux) peuvent être discriminants — une résolution plus élevée peut améliorer le Recall@1. C'est aussi utile pour trouver le bon compromis vitesse/précision selon les ressources GPU disponibles.

### Concepts théoriques

**Résolution et Vision Transformers**
- Les ViTs découpent l'image en patches de taille fixe (patch16 pour ViT-Base, patch14 pour DinoV2). La résolution doit être un multiple de la taille du patch :
  - ViT-Base/patch16 : multiples de 16 — ex. 224, 256, 320, 384
  - DinoV2/patch14 : multiples de 14 — ex. 224, 280, 336, 392, 448, 518
- À une résolution différente de celle du pré-entraînement, les **position embeddings** doivent être interpolés en 2D bilinéaire vers la nouvelle grille. `timm` gère cette interpolation automatiquement via le paramètre `img_size` dans `create_model`.

**Résolution et CNNs (MobileNetV2, ResNet50)**
- Les CNNs avec Global Average Pooling sont naturellement **résolution-agnostiques** : ils acceptent toute taille d'entrée sans modification de l'architecture.
- Une résolution plus élevée augmente la précision des feature maps intermédiaires mais aussi la mémoire et le temps de calcul (quadratique en `img_size`).

**Compromis mémoire / précision**
| Résolution | VRAM approx. (batch 64) | Gain qualitatif |
|------------|------------------------|----------------|
| 224 | baseline | baseline |
| 320 | ×2 | +modéré |
| 384 | ×2.9 | +significatif pour ViT |
| 518 | ×5.3 | natif DinoV2 |

### Fichiers modifiés
| Fichier | Changement |
|---------|-----------|
| `app/training/dataset.py` | Remplacement des 4 constantes de transforms par une factory `make_transforms(img_size, augment)` ; les anciennes constantes (`TRAIN_TRANSFORMS`, etc.) restent comme alias pour la rétrocompatibilité |
| `app/training/models.py` | `MetricModel.__init__` accepte `img_size: int \| None` ; `timm.create_model` reçoit `img_size` pour les ViTs (interpolation des position embeddings) ; `self.img_size` stocké sur le modèle ; `load_for_inference` relit `img_size` depuis le checkpoint |
| `app/training/trainer.py` | Paramètre `img_size: int \| None = None` (None → valeur par défaut du modèle) ; suppression de la logique `use_518` remplacée par `make_transforms(actual_size)` ; `img_size` sauvegardé dans le checkpoint |

---

## v1.6 — Gradient Accumulation et Mixed Precision

### Changement
Ajout de deux techniques d'optimisation mémoire activables indépendamment via des paramètres du trainer : la **mixed precision** (entraînement en float16 via AMP) et le **gradient accumulation** (simulation d'un batch plus grand en accumulant les gradients sur plusieurs micro-batches).

### Pourquoi
ViT-Base à haute résolution (ex. 320 ou 384 px) consomme beaucoup plus de VRAM qu'à 224 px, pouvant dépasser la capacité GPU (ex. 16 Go sur T4 Colab). Ces deux techniques permettent de conserver des batchs effectivement grands sans augmenter la VRAM, ce qui est crucial pour la Proxy-Anchor Loss qui dépend de la diversité de classes dans chaque batch.

### Concepts théoriques

**Mixed Precision (AMP — Automatic Mixed Precision)**
- Les activations et les calculs du forward/backward sont effectués en **float16** (moitié de la mémoire de float32).
- Les poids du modèle restent en **float32** (master weights) pour préserver la précision des mises à jour.
- Un **GradScaler** multiplie la loss par un facteur d'échelle avant le backward pour éviter l'*underflow* des gradients en float16 (valeurs trop petites représentées comme 0). Il divise ensuite les gradients avant `optimizer.step()` (`unscale_`).
- Gain mémoire typique : **40–50 %** sur les activations. Gain de vitesse : **2–3×** sur GPU avec Tensor Cores (A100, V100, RTX).
- Uniquement disponible sur CUDA ; désactivé automatiquement si `device = cpu`.

**Gradient Accumulation**
- Au lieu de faire `optimizer.step()` à chaque micro-batch, on accumule les gradients sur `accumulation_steps` micro-batches avant de mettre à jour les poids.
- La loss de chaque micro-batch est divisée par `accumulation_steps` avant le `.backward()`, de sorte que le gradient accumulé est équivalent à la moyenne sur tous les micro-batches — identique mathématiquement à un batch `accumulation_steps` fois plus grand.

$$g_{\text{eff}} = \frac{1}{K} \sum_{k=1}^{K} \nabla \mathcal{L}_k \quad \text{où } K = \texttt{accumulation\_steps}$$

- **Batch effectif** = `batch_size × accumulation_steps` (ex. 64 × 4 = 256 images) sans augmenter la VRAM.
- Dernier micro-batch partiel : si `len(train_loader)` n'est pas divisible par `accumulation_steps`, le step est quand même effectué au dernier batch pour ne pas perdre de gradients.

**Interaction AMP + Gradient Accumulation**
- `scaler.unscale_(optimizer)` est appelé uniquement au moment du step (tous les `accumulation_steps` micro-batches), pas à chaque micro-batch — sinon le facteur d'échelle serait divisé plusieurs fois.
- Le gradient clipping (`clip_grad_norm_`) est appliqué **après** `unscale_` et **avant** `scaler.step()` pour opérer sur les vraies valeurs de gradient.
- Les proxies de la Proxy-Anchor Loss sont maintenant inclus dans le clipping (corrige un oubli des versions précédentes).

### Fichiers modifiés
| Fichier | Changement |
|---------|-----------|
| `app/training/trainer.py` | Paramètres `mixed_precision: bool = False` et `accumulation_steps: int = 1` ; initialisation de `GradScaler` et `use_amp` après le scheduler ; boucle interne restructurée avec `autocast`, accumulation conditionnelle, `unscale_` + clipping unifié sur backbone et proxies ; `mixed_precision` et `accumulation_steps` sauvegardés dans le checkpoint |

---

## v2.0 — Retrieval multimodal CLIP + Flickr8K + FAISS

### Changement
Ajout d'une page dédiée au retrieval **image–texte cross-modal** basé sur CLIP (Contrastive Language–Image Pre-training) et la base Flickr8K, avec recherche efficace par plus proche voisin via **FAISS**. La page implémente trois fonctionnalités : recherche texte→image, recherche inverse image→texte, et évaluation quantitative sur un corpus personnalisé.

### Pourquoi
Le moteur v1.x est purement **unimodal** (image requête → images similaires). CLIP ouvre la modalité texte : une description en langage naturel peut retrouver des images sémantiquement proches, et inversement une image peut retrouver ses légendes. C'est le paradigme fondamental des moteurs de recherche modernes (Google Images, Pinterest, DALL-E).

### Concepts théoriques

**CLIP (Radford et al., OpenAI 2021)**
- Entraîné en **contrastive learning** sur 400 millions de paires (image, texte) du web.
- Deux encodeurs distincts : un encodeur image (ViT ou ResNet) et un encodeur texte (Transformer).
- Objectif : maximiser la similarité cosinus entre les embeddings d'une paire (image, caption) correcte et minimiser ceux des paires incorrectes — même principe que la InfoNCE loss.
- Résultat : un espace d'embedding **partagé** où une image de chien et la phrase *"a dog running on the beach"* sont proches.
- Zéro-shot : sans fine-tuning sur Flickr8K, CLIP généralise naturellement au domaine photo.

**Flickr8K**
- Dataset de **8 091 images** photographiques, chacune annotée avec **5 captions** rédigées manuellement.
- Référence standard pour l'évaluation du retrieval cross-modal (image–texte et texte–image).
- Pertinence ground-truth : une caption est pertinente pour une image si et seulement si elle en est l'annotation.

**FAISS (Johnson et al., Facebook AI 2019)**
- Bibliothèque de recherche par plus proche voisin sur des vecteurs denses de haute dimension.
- Index `IndexFlatIP` : produit scalaire exact (équivalent similarité cosinus sur vecteurs normalisés L2).
- Pour des bases plus grandes, des index quantifiés (`IVF`, `HNSW`) permettent une recherche approximative en O(log N) au lieu de O(N).
- Avantage vs numpy : implémentation BLAS optimisée, support GPU natif, batch queries vectorisées.

**Métriques cross-modal**
- **Recall@K** (R@K) : proportion de requêtes pour lesquelles le résultat pertinent est dans les K premiers — métrique principale pour Flickr8K.
- **Precision@K** : proportion de résultats pertinents parmi les K premiers.
- **Average Precision (AP)** : aire sous la courbe précision-rappel pour une requête.
- **MAP** : moyenne des AP sur toutes les requêtes du corpus d'évaluation.

### Fonctionnalités de la page CLIP

| Section | Fonctionnalité |
|---------|---------------|
| 01 — Texte → Image | Saisie libre d'une description → top-K images les plus similaires via FAISS |
| 02 — Image → Texte | Sélection d'une image Flickr8K → top-K captions les plus proches (inverse search) |
| 03 — Évaluation | Corpus de 3 images + 3 textes → P@k, R@k, AP par requête, MAP agrégé |

### Architecture backend

| Endpoint | Rôle |
|----------|------|
| `GET /api/clip/images` | Liste paginée des images Flickr8K indexées |
| `POST /api/clip/text-to-image` | Encode le texte avec CLIP → recherche FAISS sur l'index image |
| `POST /api/clip/image-to-text` | Récupère l'embedding image stocké → recherche FAISS sur l'index caption |
| `POST /api/clip/evaluate` | Évalue P@k, R@k, AP, MAP sur un corpus images + textes |

Images Flickr8K servies via `GET /flickr8k/{filename}` (StaticFiles FastAPI).

### Détails d'implémentation backend

**Chargement en deux niveaux** pour éviter de bloquer le démarrage du serveur :
- `_load_data()` : charge les index FAISS + le CSV (`captions.txt`) en ~1 s. Appelé par tous les endpoints.
- `_load_model()` : charge le modèle CLIP ViT-B/32 (poids OpenAI via `open_clip`). Appelé uniquement par les endpoints nécessitant l'encodage texte. ~10 s à la première requête, puis mis en cache.

**Reconstruction des vecteurs image** : pour `image-to-text`, l'embedding de l'image est reconstruit directement depuis l'index FAISS (`index_images.reconstruct(idx)`) sans re-encoder depuis le fichier image — plus rapide et sans dépendance à la présence du fichier.

**Ground truth pour l'évaluation** :
- *Texte → Image* : si le texte est une caption Flickr8K connue, l'image pertinente est celle à laquelle elle appartient (1 pertinent / requête). Texte libre → 0 pertinent, métriques nulles.
- *Image → Texte* : les 5 captions annotées de l'image dans `captions.txt` constituent le ground truth (5 pertinents / requête).

**Compatibilité des embeddings** : `open_clip` avec `pretrained="openai"` charge les poids officiels OpenAI ViT-B/32, produisant des embeddings bit-à-bit identiques à `clip.load("ViT-B/32")` utilisé dans le notebook de génération des index.

### Fichiers créés / modifiés
| Fichier | Changement |
|---------|-----------|
| `frontend/src/components/CLIPPage.jsx` | **Créé** — page unique avec les 3 sections (TextToImage, ImageToText, Evaluation) + navigateur d'images partagé + affichage des métriques |
| `frontend/src/components/CLIPPage.module.css` | **Créé** — styles dark theme cohérents avec le reste de l'application |
| `frontend/src/api.js` | **Modifié** — ajout de `getClipImages`, `clipTextToImage`, `clipImageToText`, `clipEvaluate` |
| `frontend/src/App.jsx` | **Modifié** — bouton "CLIP" dans la navbar, rendu conditionnel de `CLIPPage` |
| `app/clip_searcher.py` | **Créé** — `CLIPSearcher` singleton : chargement FAISS + CSV, encodage texte CLIP, recherche kNN, métriques d'évaluation |
| `app/main.py` | **Modifié** — 4 endpoints `/api/clip/*`, mount `/flickr8k` pour les images Flickr8K |
