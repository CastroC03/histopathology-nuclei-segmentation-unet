from .model import build_model

import torch


def main():

    model = build_model()

    x = torch.randn(8, 3, 512, 512)

    y = model(x)

    print("=" * 50)

    print(model.__class__.__name__)

    print("=" * 50)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {y.shape}")

    print(f"Output dtype : {y.dtype}")


if __name__ == "__main__":
    main()