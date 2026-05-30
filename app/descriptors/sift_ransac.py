"""
SIFT + RANSAC descriptor — à implémenter.

Approche prévue :
  1. Détecter les keypoints SIFT sur chaque image (cv2.SIFT_create)
  2. Matcher les descripteurs entre la requête et la base (BFMatcher + knn k=2)
  3. Filtrer les faux positifs avec le ratio test de Lowe (ratio < 0.75)
  4. Affiner les correspondances avec RANSAC (cv2.findHomography)
  5. Score de similarité = nombre d'inliers RANSAC normalisé

Note : cette approche est par nature asymétrique (requête vs image)
et ne se prête pas à un index pré-calculé statique — elle nécessite
une recherche en ligne candidat-par-candidat ou un pré-filtrage
(e.g. avec SIFT BOW/VLAD) avant d'appliquer RANSAC.
"""

import numpy as np


def extract(image: np.ndarray) -> np.ndarray:
    raise NotImplementedError("SIFT-RANSAC n'est pas encore implémenté.")
