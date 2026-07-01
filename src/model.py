import segmentation_models_pytorch as smp

from .config import (
    ENCODER,
    ENCODER_WEIGHTS,
    NUM_CLASSES,
)

def build_model():

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )

    return model