import matplotlib.pyplot as plt

from .config import RESULTS_DIR


def plot_metric(values,
                ylabel,
                filename):

    plt.figure(figsize=(8,5))

    plt.plot(
        values,
        linewidth=2
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / filename,
        dpi=300
    )

    plt.close()


def plot_losses(train_loss,
                val_loss):

    plt.figure(figsize=(8,5))

    plt.plot(
        train_loss,
        label="Train"
    )

    plt.plot(
        val_loss,
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "loss.png",
        dpi=300
    )

    plt.close()