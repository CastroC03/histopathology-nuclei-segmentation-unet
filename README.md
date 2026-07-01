# Biomedical Image Segmentation using U-Net

## Overview

This project implements a complete Deep Learning pipeline for biomedical image segmentation using the **U-Net** architecture with a **ResNet34 encoder**. The objective is to accurately segment cell nuclei in histopathology images using the **NuInsSeg** dataset.

The project includes:

* Dataset loading
* Data augmentation
* Model training
* Model evaluation
* Automatic metric logging
* Prediction on unseen images
* Automatic generation of figures and reports

---

## Dataset

**Dataset:** NuInsSeg (Nucleus Instance Segmentation)

The dataset contains RGB histopathology images and their corresponding binary segmentation masks.

Directory structure:

```
data/
├── images/
└── masks/
```

---

## Model

* Architecture: U-Net
* Encoder: ResNet34
* Encoder weights: ImageNet
* Framework: segmentation-models-pytorch

---

## Project Structure

```
ML/
│
├── data/
│   ├── images/
│   └── masks/
│
├── inference/
│   ├── input/
│   └── output/
│
├── models/
│
├── notebooks/
│
├── results/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── dataloader.py
│   ├── transforms.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── utils.py
│   └── __init__.py
│
├── main.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

To train the model and evaluate its performance:

```bash
python main.py
```

During training, the pipeline automatically:

* Trains the model
* Computes Dice Score and IoU
* Saves the best model
* Generates training plots
* Creates a training report

---

## Inference

To predict binary masks for new images:

1. Copy your images into:

```
inference/input/
```

2. Run:

```bash
python inference.py
```

3. The predicted binary masks will be saved in:

```
inference/output/
```

using the same filenames as the input images.

---

## Training Outputs

After training, the following files are automatically generated:

```
models/
└── best_model.pth
```

```
results/
├── loss.png
├── dice_score.png
├── iou.png
├── training_history.csv
├── report.txt
└── predictions/
```

---

## Evaluation Metrics

The model is evaluated using:

* Binary Cross Entropy + Dice Loss
* Dice Score
* Intersection over Union (IoU)

---

## Main Libraries

* PyTorch
* Torchvision
* OpenCV
* Albumentations
* segmentation-models-pytorch
* NumPy
* Matplotlib
* Pandas
* Scikit-learn
* Scikit-image
* Pillow
* tqdm

---

## Author

**Rubens Castro**

Biomedical Engineering Student | Universidad de Ingeniería y Tecnología (UTEC)

Biomedical Image Segmentation using Deep Learning

GitHub: CastroC03 (https://github.com/CastroC03)