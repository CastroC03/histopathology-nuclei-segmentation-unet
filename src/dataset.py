from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.config import IMAGE_DIR, MASK_DIR

class NucleiDataset(Dataset):
    
    def __init__(self, transform=None):
        self.image_dir = IMAGE_DIR
        self.mask_dir = MASK_DIR
        self.transform = transform
        self.images = sorted(self.image_dir.glob("*"))
        self.masks = sorted(self.mask_dir.glob("*"))

        if len(self.images) != len(self.masks):
            raise ValueError(
                "The number of images and masks does not match."
            )
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32) / 255.0

        if self.transform:
            augmented = self.transform(
                image=image,
                mask=mask
            )
            image = augmented["image"]
            mask = augmented["mask"]
    
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask