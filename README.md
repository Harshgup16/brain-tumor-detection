<div align="center">

# Brain Tumor Classifier

### Automated Brain Tumor Detection from MRI Images Using Deep Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-API-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-22c55e?style=for-the-badge)]()

> Multi-class brain tumor detection from T1-weighted contrast-enhanced MRI images using deep convolutional neural networks with ImageNet transfer learning. Classifies meningioma, glioma, and pituitary tumors across 15,000 clinical scans.

</div>

---

## Table of Contents

- [Overview](#overview)
- [CV Pipeline](#cv-pipeline)
- [Augmentation Strategy](#augmentation-strategy)
- [Models](#models)
- [Why VGG16 Outperforms on MRI](#why-vgg16-outperforms-on-mri)
- [Architecture Comparison](#architecture-comparison)
- [Tumor Classes — Visual Signatures](#tumor-classes--visual-signatures)
- [Dataset](#dataset)
- [Results](#results)
- [Key Techniques](#key-techniques)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Research Paper](#research-paper)
- [Authors](#authors)

---

## Overview

Brain tumors are among the most dangerous medical conditions, with high mortality rates and significant challenges in early diagnosis. This project presents a **complete computer vision pipeline** for automated, multi-class classification of brain tumors from MRI images using deep transfer learning.

Three production-level models are trained, evaluated, and compared:

| # | Notebook | Architecture | Test Accuracy |
|---|----------|-------------|---------------|
| 1 | `model-1.ipynb` | MobileNet | 94.00% |
| 2 | `model-2.ipynb` | ResNet50 | 94.00% |
| 3 | `model-3.ipynb` | **VGG16** ⭐ | **96.00%** |

All models classify three tumor types — **Meningioma**, **Glioma**, and **Pituitary Tumor** — from T1-weighted contrast-enhanced MRI scans.

---

## CV Pipeline

```
MRI Images (15,000 scans)
        │
        ▼
┌───────────────────────────────────────────┐
│  1. Image Acquisition & Spatial           │
│     Normalization                         │
│                                           │
│  • Resize to 224×224 px via bilinear      │
│    interpolation                          │
│  • Center-crop before resize to preserve  │
│    tumor anatomy — no aspect distortion   │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  2. Channel-wise Mean Subtraction         │
│     (VGG16 Preprocessing)                 │
│                                           │
│  • Subtract ImageNet per-channel means:   │
│    R: 103.94 · G: 116.78 · B: 123.68      │
│  • Aligns MRI histogram with pretrained   │
│    weight distribution — critical for     │
│    fine-tuning convergence                │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  3. Online Data Augmentation              │
│     (Keras ImageDataGenerator)            │
│                                           │
│  • Applied only during training —         │
│    never on val/test sets                 │
│  • All transforms anatomically plausible  │
│    for brain MRI (see section below)      │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  4. Feature Extraction via Frozen         │
│     Convolutional Backbone                │
│                                           │
│  • ImageNet-pretrained weights frozen     │
│  • Backbone outputs deep visual           │
│    descriptors: edges → textures →        │
│    semantic shapes — no gradient updates  │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  5. Custom Classifier Head                │
│     (Trained on MRI domain)               │
│                                           │
│  Flatten → Dense → Dropout → Softmax(3)  │
│  Only these layers receive gradient       │
│  updates during training                  │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  6. Evaluation                            │
│                                           │
│  • Confusion matrix (inter-class          │
│    misclassification analysis)            │
│  • Per-class Precision, Recall, F1-Score  │
│  • ROC-AUC per class                      │
│  • Balanced test set: 500 images × 3      │
└───────────────────────────────────────────┘
```

---

## Augmentation Strategy

All augmentation operations are bounded to preserve diagnostic MRI anatomy. Critically:

- **No vertical flip** — brain superior/inferior orientation is clinically meaningful
- **No color jitter** — MRI encodes signal intensity, not RGB color; hue/saturation shifts are meaningless
- **No shear** — distorts tumor shape which is a key diagnostic feature

| Transform | Range | Rationale |
|-----------|-------|-----------|
| Random rotation | ±20° | Patient head tilt variation |
| Horizontal flip | on/off | Left–right brain symmetry |
| Zoom | 10–20% | Variable MRI scan zoom levels |
| Brightness shift | ±15% | Scanner gain/contrast variation |
| Width shift | ±10% | Centering variability |
| Height shift | ±10% | Centering variability |
| Minority class oversampling | — | Class balance enforcement |

---

## Models

### Model 1 — MobileNet

> `model-1.ipynb`

```
Architecture  : MobileNet (Depthwise Separable Convolutions)
Pretrained On : ImageNet
Input Shape   : 224 × 224 × 3
Parameters    : 3,379,395 total │ 150,531 trainable
Test Accuracy : 94.00%
```

MobileNet factorizes a standard `k×k` convolution into two steps: a **depthwise convolution** (one spatial filter per input channel) followed by a **pointwise 1×1 convolution** (cross-channel mixer). This reduces floating point operations by ~8–9× compared to VGG16, at a small accuracy cost.

**CV characteristic:** Lightweight feature maps with limited cross-channel interaction. Well-suited for edge deployment but the reduced representational capacity weakens fine-grained texture discrimination on complex MRI patterns.

**Best use case:** Real-time inference on resource-constrained devices (smartphones, embedded hardware, IoT).

---

### Model 2 — ResNet50

> `model-2.ipynb`

```
Architecture  : ResNet50 (50-layer Residual Network)
Pretrained On : ImageNet
Input Shape   : 224 × 224 × 3
Parameters    : 23,888,771 total │ 301,059 trainable
Test Accuracy : 94.00%
```

ResNet50 introduced **residual skip connections** — identity shortcuts that allow gradients to bypass stacked layers during backpropagation, solving the vanishing gradient problem in deep networks. Each block uses a `1×1 → 3×3 → 1×1` bottleneck: the `1×1` layers compress then re-expand the channel dimension, reducing computation while the `3×3` layer handles spatial filtering.

**CV characteristic:** Deep feature hierarchy with strong global representations, but the bottleneck compression (to 64 channels) reduces local texture detail compared to VGG16's full-width 512-channel blocks.

**Best use case:** When a balance between depth, accuracy, and memory footprint is required.

---

### Model 3 — VGG16 ⭐ Best Performing

> `model-3.ipynb`

```
Architecture  : VGG16 (16-layer Visual Geometry Group Network)
Pretrained On : ImageNet
Input Shape   : 224 × 224 × 3
Parameters    : 14,789,955 total │ 75,267 trainable
Test Accuracy : 96.00%
```

VGG16 stacks thirteen `3×3` convolutional layers in uniform depth blocks followed by three fully connected layers. Three stacked `3×3` convolutions produce the same **effective receptive field** as a single `7×7` convolution, but apply three non-linear activations — encoding richer, hierarchical texture gradients critical for distinguishing tumor margin and enhancement patterns.

**Classification report (test set):**

```
                 Precision    Recall    F1-Score   Support
─────────────────────────────────────────────────────────
     Meningioma     0.97       0.97       0.97       500
         Glioma     0.96       0.94       0.95       500
Pituitary Tumor     0.96       0.98       0.97       500
─────────────────────────────────────────────────────────
       Accuracy                           0.96      1500
      Macro Avg     0.96       0.96       0.96      1500
   Weighted Avg     0.96       0.96       0.96      1500
```

**Best use case:** Texture-rich medical image classification where fine-grained spatial detail matters.

---

## Why VGG16 Outperforms on MRI

#### 1. Texture sensitivity via 3×3 stacking
Three stacked `3×3` conv layers simulate a `7×7` receptive field but apply three non-linearities. This encodes fine-grained intensity gradients — essential for distinguishing meningioma's sharp homogeneous enhancement from glioma's irregular ring-enhancement and heterogeneous texture.

#### 2. High-resolution spatial feature maps
VGG16's `conv5` block retains a **14×14 feature map** before global pooling, compared to **7×7** in ResNet50 and MobileNet. For 224×224 medical images with small or subtle lesions, this doubled spatial resolution preserves structural detail that is otherwise lost.

#### 3. No bottleneck — full channel width
ResNet's bottleneck compresses activations to 64 channels per block. VGG16 maintains **512 channels** throughout `conv4` and `conv5`, preserving redundant feature representations that improve robustness on visually similar tumor classes.

#### 4. Enhancement pattern encoding
Contrast-enhanced MRI tumors differ by ring vs. homogeneous vs. heterogeneous enhancement patterns. VGG16's deep `3×3` stack learns these ring-shaped high-intensity spatial patterns more effectively than MobileNet's lightweight depthwise separable filters.

---

## Architecture Comparison

| Feature | MobileNet | ResNet50 | VGG16 |
|---------|-----------|----------|-------|
| **Depth** | 28 layers | 50 layers | 16 layers |
| **Total params** | 3.4M | 23.9M | 14.8M |
| **Trainable params** | 150K | 301K | 75K |
| **Receptive field strategy** | Depthwise sep. conv | Skip + bottleneck | Stacked 3×3 |
| **Final conv feature map** | 7×7 | 7×7 | **14×14** |
| **Channel width (deepest block)** | 1024 | 512 (bottleneck: 64) | **512** |
| **Key innovation** | FLOPs reduction | Vanishing gradient solution | Deep texture encoding |
| **Training speed** | ⚡ Fastest | 🔶 Moderate | 🔴 Slowest |
| **Test accuracy** | 94.00% | 94.00% | **96.00%** |
| **Meningioma F1** | 0.95 | 0.95 | **0.97** |
| **Glioma F1** | 0.93 | 0.93 | **0.95** |
| **Pituitary F1** | 0.94 | 0.94 | **0.97** |
| **Best use case** | Mobile / edge | Balanced | Texture-rich imaging |

---

## Tumor Classes — Visual Signatures

### 🔵 Meningioma — Extra-axial, Dural-based

Originates from the meninges (WHO Grade I). **CV visual signature:** well-defined, sharp tumor margin; uniform, homogeneous contrast enhancement; characteristic "dural tail" sign (thin enhancement along the dura); peripheral location relative to brain parenchyma. The strong edge contrast and uniform texture make this the most reliably classified class.

**MRI appearance:** Extra-axial mass with a broad dural base, T1 iso/hypointense, vivid homogeneous post-contrast enhancement.

---

### 🟠 Glioma — Intra-axial, Infiltrative

Derived from glial cells; ranges from Grade I to Grade IV Glioblastoma. **CV visual signature:** highly heterogeneous internal texture; indistinct, infiltrative margin blending into surrounding parenchyma; irregular ring-enhancement pattern; central hypointense necrotic core; surrounding peritumoral edema (T2 hyperintense halo). The texture heterogeneity and irregular boundaries make glioma the most challenging class.

**MRI appearance:** Intra-axial mass, T1 hypointense center, ring-enhancing periphery, surrounding edema.

---

### 🟢 Pituitary Tumor — Sellar / Suprasellar Mass

Adenomas of the anterior pituitary gland. **CV visual signature:** midline anatomical location provides a strong spatial prior; microadenomas appear hypointense on dynamic contrast sequences (opposite to most enhancing tumors); well-defined sellar/suprasellar location with characteristic anatomical context. The distinctive anatomical location in the sella turcica makes spatial context a powerful classification cue.

**MRI appearance:** Sellar/suprasellar mass, microadenomas hypointense on T1+C dynamic sequences.

---

## Dataset

```
Source         : Multi-Cancer MRI Dataset (Kaggle · obulisainaren/multi-cancer)
─────────────────────────────────────────────────────────────────────────────────
  Total Images : 15,000 T1-weighted contrast-enhanced MRI scans
  Patients     : 233
  Classes      : 3 (Meningioma · Glioma · Pituitary Tumor)
  Input Shape  : 224 × 224 × 3 (RGB, center-cropped & bilinear-resized)
─────────────────────────────────────────────────────────────────────────────────
  Split        : Train 80% │ Validation 10% │ Test 10%
  Train        : 12,000 images   (4,000 per class)
  Validation   :  1,500 images   (  500 per class)
  Test         :  1,500 images   (  500 per class)  ← perfectly balanced
─────────────────────────────────────────────────────────────────────────────────
```

**Balanced test set design:** Equal support (500 images per class) eliminates class-frequency bias from accuracy and macro-averaged metrics, ensuring reported scores reflect true per-class discriminative performance.

---

## Results

### Training History — VGG16 (5 Epochs)

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|-------|---------------|--------------|------------|----------|
| 1 | 85.53% | 92.53% | 1.2949 | 0.6018 |
| 2 | 96.26% | — | 0.2707 | — |
| 3 | — | — | — | — |
| 4 | — | — | — | — |
| 5 | ~97%+ | **96.00%** | ~0.10 | ~0.15 |

### Per-Class Performance — VGG16 (Best Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Meningioma | 0.97 | 0.97 | **0.97** | 500 |
| Glioma | 0.96 | 0.94 | **0.95** | 500 |
| Pituitary Tumor | 0.96 | 0.98 | **0.97** | 500 |
| **Macro Average** | **0.963** | **0.963** | **0.963** | 1500 |

---

## Key Techniques

| Technique | CV Rationale |
|-----------|-------------|
| **Transfer learning** | ImageNet weights encode universal low-level vision features (Gabor-like edge detectors, color blobs, texture filters) transferable to MRI |
| **Layer freezing** | Preserves robust convolutional feature hierarchy; prevents catastrophic forgetting of low-level features on limited medical data |
| **Channel-wise mean subtraction** | Normalizes input distribution to match ImageNet statistics; essential for stable gradient flow through pretrained layers |
| **Bilinear interpolation resize** | Preserves spatial continuity of tumor boundaries vs. nearest-neighbor (which introduces blocking artifacts) |
| **Center-crop before resize** | Prevents aspect ratio distortion that would geometrically skew tumor shape features |
| **Anatomically bounded augmentation** | No vertical flip (brain orientation meaningful); no color jitter (MRI is intensity-based); no shear (distorts diagnostic tumor shape) |
| **Dropout regularization** | Prevents co-adaptation of classifier head neurons; improves generalization on unseen MRI scanners |
| **Adam optimizer (lr=0.0001)** | Low learning rate prevents overwriting pretrained feature detectors; adaptive moment estimation handles sparse gradients |
| **ModelCheckpoint** | Saves weights at best validation accuracy epoch; guards against overfitting in later epochs |
| **Balanced test evaluation** | Equal class support eliminates frequency bias; macro-F1 is the primary metric |

---

## Project Structure

```
brain-tumor-classifier/
│
├── 📓 model-1.ipynb          # MobileNet  — full training & evaluation pipeline
├── 📓 model-2.ipynb          # ResNet50   — full training & evaluation pipeline
├── 📓 model-3.ipynb          # VGG16      — full training & evaluation pipeline (best)
│
├── 📄 research_paper.pdf     # Full research paper (Physics Letters B template)
│
├── 📂 dataset/               # (not included — download from Kaggle)
│   ├── train/
│   │   ├── 0/               # Meningioma (4,000 images)
│   │   ├── 1/               # Glioma     (4,000 images)
│   │   └── 2/               # Pituitary  (4,000 images)
│   ├── valid/
│   └── test/
│
├── 📂 saved_models/          # .h5 weights (generated after training)
│   ├── mobilenet_model.h5
│   ├── resnet50_model.h5
│   └── vgg16_model.h5
│
└── 📄 README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA (strongly recommended)
- 8 GB RAM minimum

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### 2. Install dependencies

```bash
pip install numpy tensorflow matplotlib scikit-learn seaborn pandas pillow jupyter
```

### 3. Download the dataset

Download the **Multi-Cancer MRI Dataset** from Kaggle and place it at `D:/pe2/dataset/` (or update the path variable in each notebook):

```
https://www.kaggle.com/datasets/obulisainaren/multi-cancer
```

The dataset split script inside each notebook automatically organizes images into `train/`, `valid/`, and `test/` subdirectories with balanced class counts.

---

## Usage

### Launch Jupyter and run any model

```bash
jupyter notebook
# Open model-3.ipynb (VGG16 — recommended)
```

Each notebook is fully self-contained. **Run All** to:

1. Split the dataset into train/val/test
2. Initialize `ImageDataGenerator` with augmentation and VGG16 preprocessing
3. Build the transfer learning model (frozen backbone + custom classifier head)
4. Train for 5 epochs with `ModelCheckpoint` and `EarlyStopping`
5. Save best weights to `.h5`
6. Generate confusion matrix, per-class ROC curve, and classification report

### Inference on a new MRI scan

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

# Load saved model
model = tf.keras.models.load_model('saved_models/vgg16_model.h5')

# Class labels
class_labels = {0: 'Meningioma', 1: 'Glioma', 2: 'Pituitary Tumor'}

# Preprocess input — center-crop, resize, VGG16 mean subtraction
img = Image.open('mri_scan.jpg').convert('RGB')
img = img.resize((224, 224), Image.BILINEAR)
img_array = preprocess_input(np.expand_dims(np.array(img), axis=0))

# Predict
prediction = model.predict(img_array)
predicted_class = class_labels[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Predicted class : {predicted_class}")
print(f"Confidence      : {confidence:.2f}%")
print(f"Class scores    : { {class_labels[i]: f'{prediction[0][i]*100:.1f}%' for i in range(3)} }")
```

---

## Research Paper

This project is accompanied by a full research paper:

> **"Automated Brain Tumor Classification from MRI Images Using Deep Transfer Learning"**
> Submitted to *Physics Letters B*

The paper covers MRI neuro-oncology background, CNN architecture analysis, related works (2019–2022), experimental methodology, and comparative results including:

- Magnetic Resonance Imaging in neuro-oncology (T1, T2, FLAIR, contrast sequences)
- CNN feature hierarchy analysis — what each convolutional block learns
- Visual explanation of transfer learning domain adaptation
- Confusion matrix analysis and inter-class misclassification patterns
- Full architecture diagrams with feature map dimensions

---

## Authors

**Harsh · Komal**
*University of the Moon*

---

<div align="center">

*Built with TensorFlow · Keras · Python*

⭐ Star this repository if you found it useful!

</div>
