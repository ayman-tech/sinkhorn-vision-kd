"""
CIFAR-10 and CIFAR-100 data loading with standard augmentation.

Augmentation pipeline (following standard practice for CIFAR):
  Train: RandomCrop(32, padding=4) -> RandomHorizontalFlip -> Normalize
  Test:  Normalize only

Optionally splits a fraction of the training set as a validation set
for bilevel optimization of the learnable cost matrix C.
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as T
from typing import Tuple, Optional

# Per-dataset normalization constants (computed over training sets)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Fine-grained class names for CIFAR-100 (used in cost matrix visualization)
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_cifar_loaders(
    dataset: str = "cifar100",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_fraction: float = 0.0,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Create CIFAR train/val/test data loaders with standard augmentation.

    Args:
        dataset: Either "cifar10" or "cifar100".
        data_dir: Root directory for dataset download/storage.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        val_fraction: Fraction of training data to reserve for validation
            (used for bilevel optimization of cost matrix C). If 0, no
            validation loader is created.
        seed: Random seed for reproducible train/val split.

    Returns:
        (train_loader, val_loader, test_loader) where val_loader is None
        if val_fraction == 0.
    """
    if dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        Dataset = torchvision.datasets.CIFAR10
    elif dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        Dataset = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'cifar10' or 'cifar100'.")

    # Standard CIFAR augmentation
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_dataset = Dataset(root=data_dir, train=True, download=True,
                            transform=train_transform)
    test_dataset = Dataset(root=data_dir, train=False, download=True,
                           transform=test_transform)

    # Optionally split training data for bilevel optimization
    val_loader = None
    if val_fraction > 0:
        total = len(train_dataset)
        val_size = int(total * val_fraction)
        train_size = total - val_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size], generator=generator
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_class_names(dataset: str) -> list:
    """Return human-readable class names for a dataset."""
    if dataset == "cifar10":
        return CIFAR10_CLASSES
    elif dataset == "cifar100":
        return CIFAR100_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
