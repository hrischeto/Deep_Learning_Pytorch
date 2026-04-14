# Image Classification — Playing Card Recognition

## Purpose

This notebook benchmarks three deep learning approaches for multi-class image classification, using playing card recognition as the problem domain. The goal is to compare a from-scratch CNN against two transfer learning strategies — frozen feature extraction and full fine-tuning — and evaluate their trade-offs on the same held-out test set.

---

## Technologies

- **PyTorch** — model definition, training loop, loss and optimization
- **torchvision** — pretrained model weights, image transforms, and augmentation
- **pandas / NumPy** — dataset loading and metadata handling
- **matplotlib** — training curve visualization and results comparison
- **Google Colab** — execution environment with GPU acceleration
- **Kaggle API** — programmatic dataset download

---

## Dataset

[Cards Image Dataset Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) — 53 classes of playing card images, split into train, validation, and test sets. A Kaggle API token (`kaggle.json`) is required to download it.

---

## Methods

**Data augmentation** is applied during training (random flips, rotation, and color jitter) to improve generalization. Validation and test images receive only normalization, using ImageNet statistics.

**Three model architectures** are trained and compared:

| Model | Strategy | Description |
|---|---|---|
| Baseline CNN | From scratch | Custom 4-block convolutional network with BatchNorm, trained entirely on this dataset |
| ResNet-18 (Frozen) | Transfer learning | ImageNet-pretrained backbone with all convolutional weights frozen; only the classifier head is trained |
| EfficientNet-B0 (Fine-tuned) | Fine-tuning | ImageNet-pretrained weights with the full network updated end-to-end at a low learning rate |

All models are trained with AdamW and a cosine annealing learning rate schedule. Loss is computed with cross-entropy.

---

## Evaluation

Each model is assessed on the held-out test set by loss and top-1 accuracy (Correct / Total). Training and validation curves are plotted across epochs for all three models, and final test accuracies are compared in a summary bar chart.
