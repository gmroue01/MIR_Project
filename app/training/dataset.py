"""
Car dataset for metric learning.
Returns (image_tensor, class_index) pairs. Online triplet mining is done in the loss.
"""
import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def get_class_label(filename: str) -> str:
    """Extract class from filename: brandId_modelId_Brand_Model_index.jpg → brandId_modelId_Brand_Model"""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    return "_".join(parts[:4]) if len(parts) >= 4 else base


def build_splits(dataset_dir: str, val_ratio: float = 0.2, seed: int = 42):
    """Stratified 80/20 split. Returns (train_paths, val_paths)."""
    all_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))
    class_to_paths: dict = {}
    for p in all_paths:
        cls = get_class_label(p)
        class_to_paths.setdefault(cls, []).append(p)

    rng = np.random.default_rng(seed)
    train_paths, val_paths = [], []
    for cls, paths in class_to_paths.items():
        paths = sorted(paths)
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_ratio))
        val_paths.extend(paths[:n_val])
        train_paths.extend(paths[n_val:])

    return train_paths, val_paths


TRAIN_TRANSFORMS = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# DinoV2 uses 518x518
TRAIN_TRANSFORMS_518 = T.Compose([
    T.Resize((518, 518)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS_518 = T.Compose([
    T.Resize((518, 518)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class CarDataset(Dataset):
    def __init__(self, paths: list, transform=None):
        self.paths = paths
        self.transform = transform

        # Build class → int label mapping
        classes = sorted(set(get_class_label(p) for p in paths))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.labels = [self.class_to_idx[get_class_label(p)] for p in paths]
        self.num_classes = len(classes)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
