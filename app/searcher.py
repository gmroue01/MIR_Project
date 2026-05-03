"""
Core search engine. Loads pre-computed indexes and performs nearest-neighbour
search with concatenated descriptors and a chosen similarity measure.
"""
import os
import json
import numpy as np
from typing import List, Tuple

from app.similarity.measures import MEASURES, BATCH_MEASURES
from app.pca_reducer import PCA_TARGETS, reduce, reduce_vector

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "indexes")

# Distribution-based measures that benefit from PCA on high-dim descriptors
_DISTRIBUTION_MEASURES = {"jensen", "chi_square"}

_cache: dict = {}
_prepared_cache: dict = {}  # (descriptor_name, measure) -> prepared matrix


def _load(name: str) -> np.ndarray:
    if name not in _cache:
        path = os.path.join(INDEX_DIR, f"{name}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index not found: {path}. Run indexer first.")
        _cache[name] = np.load(path)["features"]
    return _cache[name]


def _load_meta() -> Tuple[np.ndarray, np.ndarray]:
    if "_meta" not in _cache:
        f_path = os.path.join(INDEX_DIR, "filenames.npy")
        c_path = os.path.join(INDEX_DIR, "classes.npy")
        if not os.path.exists(f_path) or not os.path.exists(c_path):
            return np.array([]), np.array([])
        _cache["_meta"] = (np.load(f_path), np.load(c_path))
    return _cache["_meta"]


def _normalize_l2(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _prepare_descriptor(name: str, measure: str) -> np.ndarray:
    """Load descriptor matrix, apply PCA for distribution measures on large descriptors.
    Result is cached in memory for the lifetime of the server process."""
    cache_key = (name, measure if (name in PCA_TARGETS and measure in _DISTRIBUTION_MEASURES) else "_base")
    if cache_key in _prepared_cache:
        return _prepared_cache[cache_key]

    mat = _load(name).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    if name in PCA_TARGETS and measure in _DISTRIBUTION_MEASURES:
        mat = reduce(name, mat)

    _prepared_cache[cache_key] = mat
    return mat


def build_combined_features(descriptor_names: List[str], measure: str = "euclidean") -> np.ndarray:
    """Concatenate (L2-normalized, optionally PCA-reduced) feature matrices."""
    parts = [_prepare_descriptor(name, measure) for name in descriptor_names]
    return np.concatenate(parts, axis=1)


def get_query_vector(query_idx: int, descriptor_names: List[str], measure: str = "euclidean") -> np.ndarray:
    """Return the combined feature vector for a query image.
    Reuses the already-prepared (normalized + optional PCA) matrix from cache."""
    parts = []
    for name in descriptor_names:
        prepared = _prepare_descriptor(name, measure)
        parts.append(prepared[query_idx])
    return np.concatenate(parts)


def search(
    query_idx: int,
    descriptor_names: List[str],
    measure: str,
    top_k: int = 50,
) -> List[dict]:
    """
    Returns top_k results (excluding the query itself) as a list of dicts:
    {filename, class, rank, distance}
    """
    filenames, classes = _load_meta()
    batch_fn = BATCH_MEASURES[measure]

    db_matrix = build_combined_features(descriptor_names, measure)
    query_vec = get_query_vector(query_idx, descriptor_names, measure)

    distances = batch_fn(query_vec, db_matrix).astype(np.float32)
    distances[query_idx] = np.inf
    ranked = np.argsort(distances)

    results = []
    for rank, idx in enumerate(ranked[:top_k], start=1):
        results.append({
            "filename": str(filenames[idx]),
            "class": str(classes[idx]),
            "rank": rank,
            "distance": float(distances[idx]),
            "index": int(idx),
        })
    return results


def get_indexing_metrics() -> dict:
    path = os.path.join(INDEX_DIR, "metrics.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_all_images() -> List[dict]:
    filenames, classes = _load_meta()
    return [
        {"filename": str(filenames[i]), "class": str(classes[i]), "index": i}
        for i in range(len(filenames))
    ]
