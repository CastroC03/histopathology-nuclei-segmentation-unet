from .dataloader import get_dataloaders


def main():

    train_loader, val_loader, test_loader = get_dataloaders()

    print("=" * 50)

    print(f"Train batches      : {len(train_loader)}")
    print(f"Validation batches : {len(val_loader)}")
    print(f"Test batches       : {len(test_loader)}")

    print("=" * 50)

    images, masks = next(iter(train_loader))

    print(f"Image type  : {type(images)}")
    print(f"Mask type   : {type(masks)}")

    print(f"Image shape : {images.shape}")
    print(f"Mask shape  : {masks.shape}")

    print(f"Image dtype : {images.dtype}")
    print(f"Mask dtype  : {masks.dtype}")


if __name__ == "__main__":
    main()