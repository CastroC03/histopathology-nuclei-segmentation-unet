from pathlib import Path

import cv2
import numpy as np
import torch

from .config import DEVICE, MODELS_DIR
from .model import build_model
from .transforms import get_val_transform


INPUT_DIR = Path("to_predict/input")
OUTPUT_DIR = Path("to_predict/output")


def predict():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model().to(DEVICE)

    checkpoint = torch.load(
        MODELS_DIR / "best_model.pth",
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint)

    model.eval()

    transform = get_val_transform()

    image_paths = sorted(INPUT_DIR.glob("*"))

    with torch.no_grad():

        for image_path in image_paths:

            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            augmented = transform(
                image=image,
                mask=np.zeros(image.shape[:2], dtype=np.float32)
            )

            image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

            output = model(image_tensor)

            prediction = torch.sigmoid(output)
            prediction = (prediction > 0.5).float()

            mask = prediction.squeeze().cpu().numpy()
            mask = (mask * 255).astype(np.uint8)

            output_path = OUTPUT_DIR / f"{image_path.stem}.png"

            cv2.imwrite(str(output_path), mask)

            print(f"Saved: {output_path}")

    print("\nInference completed successfully.")