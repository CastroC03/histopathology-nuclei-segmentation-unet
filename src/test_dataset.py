import numpy as np
import matplotlib.pyplot as plt

from .dataset import NucleiDataset


def main():
    dataset = NucleiDataset()

    # Número de imágenes
    print("=" * 50)
    print(f"Images detected: {len(dataset)}")
    print("=" * 50)

    # Primera muestra
    image, mask = dataset[0]

    # Información básica
    print(f"Image shape: {image.shape}")
    print(f"Image dtype : {image.dtype}")

    print(f"Mask shape : {mask.shape}")
    print(f"Mask dtype : {mask.dtype}")

    # Valores únicos de la máscara
    unique_values = np.unique(mask)

    print(f"Mask unique values: {unique_values}")

    # Comprobar si es binaria
    if np.all(np.isin(unique_values, [0.0, 1.0])):
        print("✓ Mask is binary (0 and 1).")
    else:
        print("✗ Mask is NOT binary.")

    # Mostrar imagen y máscara
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()