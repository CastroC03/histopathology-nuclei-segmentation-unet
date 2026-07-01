import torch

from .metrics import (
    dice_score,
    iou_score,
)


def main():

    preds = torch.randn(2, 1, 512, 512)

    targets = torch.randint(
        0,
        2,
        (2, 1, 512, 512)
    ).float()

    dice = dice_score(preds, targets)
    iou = iou_score(preds, targets)

    print("=" * 50)

    print(f"Dice Score : {dice:.4f}")
    print(f"IoU Score  : {iou:.4f}")


if __name__ == "__main__":
    main()