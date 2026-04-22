import cv2
import numpy as np

_orb = cv2.ORB_create(nfeatures=500)
DESCRIPTOR_DIM = 32


def extract(image: np.ndarray) -> np.ndarray:
    """
    ORB: returns mean-pooled float vector (32-dim) for general similarity measures.
    Raw binary descriptors stored separately by the indexer for Hamming distance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = _orb.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(DESCRIPTOR_DIM, dtype=np.float32)
    feat = descriptors.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat


def extract_binary(image: np.ndarray) -> np.ndarray:
    """Raw ORB binary descriptors (N x 32 uint8) for Hamming distance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = _orb.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros((1, DESCRIPTOR_DIM), dtype=np.uint8)
    return descriptors
