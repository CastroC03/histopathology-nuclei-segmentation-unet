from pathlib import Path
import time
import pandas as pd
import torch

from .config import MODELS_DIR, RESULTS_DIR


def create_directories():

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, filename="best_model.pth"):

    torch.save(
        model.state_dict(),
        MODELS_DIR / filename
    )


def save_training_history(history):

    df = pd.DataFrame(history)

    df.to_csv(
        RESULTS_DIR / "training_history.csv",
        index=False
    )


def save_report(best_epoch,
                best_dice,
                training_time):

    report = f"""
Training Report
==============================

Best Epoch      : {best_epoch}
Best Dice Score : {best_dice:.4f}
Training Time   : {training_time:.2f} seconds
"""

    with open(
        RESULTS_DIR / "report.txt",
        "w"
    ) as f:
        f.write(report)


class Timer:

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        return time.time() - self.start_time