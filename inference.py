from src.predict import predict


def main():
    """
    Runs inference on all images in inference/input.
    """

    print("=" * 60)
    print("Biomedical Image Segmentation - Inference")
    print("=" * 60)

    predict()

    print("\n" + "=" * 60)
    print("Inference completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()