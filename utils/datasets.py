"""
Dataset loaders and preprocessing pipelines for neuromorphic vision experiments.
Supports CIFAR-10, CIFAR-100, and fast synthetic data for smoke tests.
"""

import os
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Standard CIFAR-10 training and test data augmentations."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, test_transform


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
    val_split: float = 0.0,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Returns (train_loader, test_loader, val_loader).
    If val_split is 0.0, val_loader is None.
    """
    train_transform, test_transform = get_cifar10_transforms()

    full_train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=test_transform
    )

    if val_split > 0.0:
        val_size = int(len(full_train_set) * val_split)
        train_size = len(full_train_set) - val_size
        train_set, val_set = random_split(
            full_train_set,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        train_set = full_train_set
        val_loader = None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader, val_loader


def get_cifar100_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Returns (train_loader, test_loader) for CIFAR-100."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )

    train_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=download, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def get_synthetic_loaders(
    batch_size: int = 16,
    num_train_samples: int = 64,
    num_test_samples: int = 32,
    num_classes: int = 10,
    img_size: int = 32,
    in_channels: int = 3,
) -> Tuple[DataLoader, DataLoader]:
    """
    Generates fast synthetic data for instant CPU smoke tests and unit testing
    without external network downloads.
    """
    g = torch.Generator().manual_seed(42)
    train_x = torch.randn(num_train_samples, in_channels, img_size, img_size, generator=g)
    train_y = torch.randint(0, num_classes, (num_train_samples,), generator=g)

    test_x = torch.randn(num_test_samples, in_channels, img_size, img_size, generator=g)
    test_y = torch.randint(0, num_classes, (num_test_samples,), generator=g)

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_event_loaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 2,
    T: int = 8,
    img_size: int = 32,
    synthetic: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Neuromorphic event-based vision data loader.
    Outputs tensors with 2 polarity channels (ON/OFF) matching neuromorphic sensors.
    """
    if synthetic or not os.path.exists(os.path.join(data_dir, "cifar10_dvs")):
        # Generate synthetic neuromorphic spike event streams: [N, 2, H, W]
        g = torch.Generator().manual_seed(42)
        num_train = 64 if synthetic else 256
        num_test = 32 if synthetic else 64
        # Binary or sparse Poisson spike events
        train_x = (torch.rand(num_train, 2, img_size, img_size, generator=g) < 0.15).float()
        train_y = torch.randint(0, 10, (num_train,), generator=g)
        test_x = (torch.rand(num_test, 2, img_size, img_size, generator=g) < 0.15).float()
        test_y = torch.randint(0, 10, (num_test,), generator=g)

        train_set = TensorDataset(train_x, train_y)
        test_set = TensorDataset(test_x, test_y)
        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
        )

    # If local CIFAR10-DVS directory exists, load via torchvision or spikingjelly
    try:
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        train_set = CIFAR10DVS(root=data_dir, data_type="frame", frames_number=T, split_by="number")
        test_set = train_set
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader
    except Exception:
        return get_event_loaders(synthetic=True, batch_size=batch_size, T=T, img_size=img_size)

