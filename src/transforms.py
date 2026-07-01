import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform():

    return A.Compose([
        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.5),

        A.Rotate(limit=30, p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2(transpose_mask=True)
    ])


def get_val_transform():

    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2(transpose_mask=True)
    ])