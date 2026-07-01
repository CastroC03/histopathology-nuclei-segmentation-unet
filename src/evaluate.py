import torch
import segmentation_models_pytorch as smp

from .config import DEVICE, MODELS_DIR
from .dataloader import get_dataloaders
from .model import build_model
from .metrics import dice_score, iou_score


def evaluate():

    _, _, test_loader = get_dataloaders()

    model = build_model().to(DEVICE)

    checkpoint = torch.load(
        MODELS_DIR / "best_model.pth",
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint)

    model.eval()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    dice_loss = smp.losses.DiceLoss(
        mode="binary",
        from_logits=True,
    )

    test_loss = 0.0
    dice = 0.0
    iou = 0.0

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = (
                bce_loss(outputs, masks)
                + dice_loss(outputs, masks)
            )

            test_loss += loss.item()

            dice += dice_score(outputs, masks)
            iou += iou_score(outputs, masks)

    test_loss /= len(test_loader)
    dice /= len(test_loader)
    iou /= len(test_loader)

    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Dice Score: {dice:.4f}")
    print(f"IoU Score : {iou:.4f}")
    print("=" * 50)

    return {
        "test_loss": test_loss,
        "dice": dice,
        "iou": iou,
    }