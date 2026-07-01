import torch
from torch.utils.data import DataLoader, Subset

from .config import (
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

from .dataset import NucleiDataset
from .transforms import (
    get_train_transform,
    get_val_transform,
)

def get_dataloaders():

    # Dataset size
    dataset = NucleiDataset()
    dataset_size = len(dataset)

    # Compute split sizes
    train_size = int(TRAIN_SPLIT * dataset_size)
    val_size = int(VAL_SPLIT * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Reproducible shuffle
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    indices = torch.randperm(
        dataset_size,
        generator=generator
    ).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Different transforms for each subset
    train_dataset = Subset(
        NucleiDataset(transform=get_train_transform()),
        train_indices
    )

    val_dataset = Subset(
        NucleiDataset(transform=get_val_transform()),
        val_indices
    )

    test_dataset = Subset(
        NucleiDataset(transform=get_val_transform()),
        test_indices
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader