"""
On-demand PCA reduction for high-dimensional descriptors.
Fitted once from the existing index and cached on disk.
Applied transparently at search time only for distribution-based measures.
"""
import os
import numpy as np

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "indexes")

# Descriptors that benefit from PCA for Jensen/Chi-square
PCA_TARGETS: set = set()  # à remplir si un futur descripteur haute-dim en a besoin
PCA_DIM = 256

_pca_cache: dict = {}


def _pca_path(name: str) -> str:
    return os.path.join(INDEX_DIR, f"{name}_pca{PCA_DIM}.npz")


def fit_and_save(name: str, matrix: np.ndarray):
    """Fit PCA on the descriptor matrix and save components to disk."""
    n, d = matrix.shape
    k = min(PCA_DIM, d, n)

    mean = matrix.mean(axis=0)
    centered = matrix - mean
    # Use SVD on a random subsample if large
    if n > 2000:
        idx = np.random.choice(n, 2000, replace=False)
        centered_s = centered[idx]
    else:
        centered_s = centered

    _, _, Vt = np.linalg.svd(centered_s, full_matrices=False)
    components = Vt[:k]  # (k, d)

    np.savez_compressed(_pca_path(name), mean=mean, components=components)
    return mean, components


def load_or_fit(name: str, matrix: np.ndarray):
    if name in _pca_cache:
        return _pca_cache[name]
    path = _pca_path(name)
    if os.path.exists(path):
        data = np.load(path)
        mean, components = data["mean"], data["components"]
    else:
        mean, components = fit_and_save(name, matrix)
    _pca_cache[name] = (mean, components)
    return mean, components


def reduce(name: str, matrix: np.ndarray) -> np.ndarray:
    """Project matrix rows into PCA space."""
    mean, components = load_or_fit(name, matrix)
    return (matrix - mean).dot(components.T)


def reduce_vector(name: str, vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Project a single vector into PCA space (uses same components as matrix)."""
    mean, components = load_or_fit(name, matrix)
    return (vec - mean).dot(components.T)
