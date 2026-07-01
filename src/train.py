import torch
import segmentation_models_pytorch as smp

from tqdm import tqdm

from .config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
)

from .dataloader import get_dataloaders
from .model import build_model
from .metrics import (
    dice_score,
    iou_score,
)

from .utils import (
    create_directories,
    save_checkpoint,
    save_training_history,
    save_report,
    Timer,
)

from .visualization import (
    plot_losses,
    plot_metric,
)


def train():

    create_directories()

    timer = Timer()
    timer.start()

    train_loader, val_loader, _ = get_dataloaders()

    model = build_model().to(DEVICE)

    bce_loss = torch.nn.BCEWithLogitsLoss()

    dice_loss = smp.losses.DiceLoss(
        mode="binary",
        from_logits=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "dice": [],
        "iou": [],
        "learning_rate": [],
    }

    best_dice = 0.0
    best_epoch = 0

    patience = 10
    patience_counter = 0

    print("\nTraining started...\n")

    for epoch in range(NUM_EPOCHS):

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

        ###########################################
        # TRAIN
        ###########################################

        model.train()

        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc="Training",
            leave=False,
        )

        for images, masks in train_bar:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = (
                bce_loss(outputs, masks)
                + dice_loss(outputs, masks)
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss /= len(train_loader)

        ###########################################
        # VALIDATION
        ###########################################

        model.eval()

        val_loss = 0.0

        dice = 0.0
        iou = 0.0

        with torch.no_grad():

            val_bar = tqdm(
                val_loader,
                desc="Validation",
                leave=False,
            )

            for images, masks in val_bar:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)

                loss = (
                    bce_loss(outputs, masks)
                    + dice_loss(outputs, masks)
                )

                val_loss += loss.item()

                dice += dice_score(
                    outputs,
                    masks,
                )

                iou += iou_score(
                    outputs,
                    masks,
                )

        val_loss /= len(val_loader)
        dice /= len(val_loader)
        iou /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["dice"].append(dice)
        history["iou"].append(iou)
        history["learning_rate"].append(
            optimizer.param_groups[0]["lr"]
        )

        scheduler.step(dice)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Dice: {dice:.4f} | "
            f"IoU: {iou:.4f}"
        )
        
        ###########################################
        # SAVE BEST MODEL
        ###########################################

        if dice > best_dice:

            best_dice = dice
            best_epoch = epoch + 1

            save_checkpoint(model)

            patience_counter = 0

            print("✓ Best model updated.")

        else:
            patience_counter += 1

        ###########################################
        # EARLY STOPPING
        ###########################################

        if patience_counter >= patience:

            print("\nEarly stopping triggered.")

            break

    ###########################################
    # TRAINING FINISHED
    ###########################################

    training_time = timer.stop()

    save_training_history(history)

    plot_losses(
        history["train_loss"],
        history["val_loss"],
    )

    plot_metric(
        history["dice"],
        ylabel="Dice Score",
        filename="dice_score.png",
    )

    plot_metric(
        history["iou"],
        ylabel="IoU",
        filename="iou.png",
    )

    save_report(
        best_epoch=best_epoch,
        best_dice=best_dice,
        training_time=training_time,
    )

    print("\n" + "=" * 50)
    print("Training completed successfully.")
    print("=" * 50)

    print(f"Best Epoch      : {best_epoch}")
    print(f"Best Dice Score : {best_dice:.4f}")
    print(f"Training Time   : {training_time:.2f} s")

    print("=" * 50)