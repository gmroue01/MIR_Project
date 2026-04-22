import cv2
import numpy as np


def extract(image: np.ndarray) -> np.ndarray:
    """HOG descriptor via OpenCV on grayscale 128x128 image, L2-normalized."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))

    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    feat = hog.compute(resized).flatten().astype(np.float32)
    norm = np.linalg.norm(feat)
    return feat / norm if norm > 0 else feat
