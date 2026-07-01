from src.train import train
from src.evaluate import evaluate


def main():
    """
    Training and evaluation pipeline.
    """

    print("=" * 60)
    print("Biomedical Image Segmentation Pipeline")
    print("=" * 60)

    print("\n[1/2] Training model...")
    train()

    print("\n[2/2] Evaluating model...")
    evaluate()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()