# Informations sur les données du projet MIR

## Vue d'ensemble

| Dataset | Rôle | Emplacement |
|---|---|---|
| **Car Dataset** | Images de voitures pour la recherche par image | `dataset/` |
| **Flickr8K** | Images naturelles + légendes pour la recherche multimodale (CLIP) | `Flickr8K/` |
| **FAISS Indexes** | Index vectoriels pré-calculés (CLIP ViT-B/32) | `indexes_faiss/` |

---

## 1. Car Dataset (`dataset/`)

### Statistiques globales

| Métrique | Valeur |
|---|---|
| Fichiers totaux | 5 012 |
| Images valides | 5 000 |
| Fichiers malformés | 12 (doublons `Ford_GT` avec `(1)` dans le nom) |
| Nombre de classes | 46 |
| Nombre de marques | 5 |
| Taille totale | ~524 MB |
| Taille moyenne par image | ~107 KB |

### Format des noms de fichiers

```
{split_id}_{batch_id}_{Marque}_{Modele}_{index}.jpg
ex : 0_0_BMW_Serie3Berline_1.jpg
```

La classe est reconstruite comme `Marque_Modele` (tokens `[2:-1]` en séparant sur `_`).

### Répartition par marque

| Marque | Images | Classes |
|---|---|---|
| BMW | 1 000 | 9 |
| Ford | 1 000 | 8 |
| Hyundai | 1 000 | 10 |
| Opel | 1 000 | 10 |
| Volkswagen | 1 000 | 9 |
| **Total** | **5 000** | **46** |

### Répartition par classe

| Classe | Images | Classe | Images |
|---|---|---|---|
| BMW_X3 | 162 | Volkswagen_Golf | 100 |
| Ford_Kuga | 158 | Volkswagen_GolfVariant | 100 |
| Ford_GT | 157 | Volkswagen_Passat | 100 |
| Hyundai_i10 | 148 | Volkswagen_Polo | 100 |
| Ford_Puma | 140 | Volkswagen_Sharan | 100 |
| BMW_i8 | 139 | Volkswagen_T-cross | 100 |
| Ford_S-max | 138 | Volkswagen_T-Roc | 100 |
| Opel_crosslandX | 136 | Volkswagen_Tiguan | 100 |
| Hyundai_i20 | 124 | Volkswagen_Touareg | 100 |
| Hyundai_kona | 117 | Volkswagen_up | 100 |
| Hyundai_i30break | 111 | Opel_astra | 100 |
| Ford_Explorer | 107 | Opel_corsa | 100 |
| BMW_Serie3Berline | 100 | Opel_GrandlandX | 100 |
| BMW_Serie3Touring | 100 | Opel_Insignatourer | 100 |
| BMW_Serie5 | 100 | Opel_vivarofourgon | 100 |
| BMW_X2 | 100 | Opel_zafiralife | 100 |
| BMW_X4 | 100 | Hyundai_i30 | 100 |
| BMW_X5 | 100 | Hyundai_i30fastback | 100 |
| Ford_Fiesta | 100 | Hyundai_Newtucson | 100 |
| Ford_Focus | 100 | Hyundai_Nexo | 100 |
| Ford_Galaxy | 100 | Hyundai_Santafe | 100 |
| BMW_Serie5Touring | 99 | Opel_movano | 94 |
| Opel_Insigna | 91 | Opel_astrabreak | 79 |

**Min :** 79 images (`Opel_astrabreak`) — **Max :** 162 images (`BMW_X3`) — **Moyenne :** 108,7 images/classe

### Fichiers malformés (à ignorer/nettoyer)

12 fichiers `Ford_GT` avec parenthèses dans le nom (`Ford_GT_11103(1).jpg` → `Ford_GT_11114(1).jpg`). Ce sont des doublons. Le parser de classe les ignore automatiquement.

---

## 2. Flickr8K (`Flickr8K/`)

### Statistiques globales

| Métrique | Valeur |
|---|---|
| Images | 8 091 |
| Légendes (captions) | 40 455 |
| Légendes par image | 5 (exactement) |
| Langue | Anglais |
| Format images | JPEG |

### Structure

```
Flickr8K/
├── Images/          # 8 091 fichiers .jpg
└── captions.txt     # CSV : colonnes image, caption
```

### Exemple de captions

```
image,caption
1000268201_693b08cb0e.jpg, A child in a pink dress is climbing up a set of stairs in an entry way.
1000268201_693b08cb0e.jpg, A girl going into a wooden building.
1000268201_693b08cb0e.jpg, A little girl climbing into a wooden playhouse.
1000268201_693b08cb0e.jpg, A little girl climbing the stairs to her playhouse.
1000268201_693b08cb0e.jpg, A little girl in a pink dress going into a wooden cabin.
```

### Usage dans le projet

Utilisé pour la **recherche multimodale CLIP** :
- Requête texte → images les plus proches (image retrieval)
- Requête image → images les plus proches (image-to-image)
- Requête image → captions les plus proches (image-to-text)

---

## 3. FAISS Indexes (`indexes_faiss/`)

### Fichiers

| Fichier | Taille | Contenu |
|---|---|---|
| `index_images.faiss` | 15,8 MB | 8 091 vecteurs — un par image Flickr8K |
| `index_captions.faiss` | 79,0 MB | 40 455 vecteurs — un par caption Flickr8K |

### Spécifications techniques

| Propriété | Valeur |
|---|---|
| Modèle source | **CLIP ViT-B/32** (OpenAI via `open_clip`) |
| Dimension des embeddings | 512 |
| Type d'index FAISS | `IndexFlatIP` (Inner Product = cosine sim sur vecteurs normalisés L2) |
| Normalisation | L2 (vecteurs unitaires) |

### Correspondance avec les données

- `index_images.faiss[i]` ↔ `df['image'].unique()[i]` (image Flickr8K)
- `index_captions.faiss[i]` ↔ `df.iloc[i]['caption']` (caption Flickr8K)

---

## 4. Résumé des ressources

| Ressource | Taille | Remarque |
|---|---|---|
| `dataset/` (voitures) | ~524 MB | 5 000 images, 46 classes |
| `Flickr8K/` (images + captions) | ~1 GB | 8 091 images, 40 455 captions |
| `indexes_faiss/` (CLIP) | ~95 MB | IndexFlatIP, dim 512 |
| **Total** | **~1,6 GB** | |

---

## 5. Notes et limitations

- Le Car Dataset est **déséquilibré** : entre 79 et 162 images/classe. Pour l'évaluation (mAP, Recall@K), les classes petites ont moins de positifs, ce qui peut biaiser les métriques globales.
- Flickr8K est un dataset **généraliste** (scènes naturelles, personnes, animaux) — pas spécialisé voitures. Les index CLIP sont entraînés dessus.
- Les **indexes ALIGN** ne sont pas encore générés (notebook `MIR_ALIGN_indexing.ipynb` prévu à cet effet).
- Les 12 fichiers malformés dans `dataset/` sont des doublons `Ford_GT` — la classe `Ford_GT` a officiellement 157 images valides (et non 169 si on les compte).
