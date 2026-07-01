import torch
from pathlib import Path

# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# DATASET

IMAGE_SIZE = (512, 512)
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
RANDOM_SEED = 42

# TRAINING

BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 5e-5
NUM_WORKERS = 8

# MODEL

ENCODER = "resnet50"
ENCODER_WEIGHTS = "imagenet"
NUM_CLASSES = 1

# HARDWARE

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PIN_MEMORY = DEVICE.type == "cuda"