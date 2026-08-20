# 🧠 NeuroScan AI
### *Tri-Model Deep Learning Ensemble for Brain Tumor MRI Classification*

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.0+-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-000000?style=for-the-badge)

**A clinical-grade, multi-architecture deep learning decision support system combining VGG-16, MobileNet, and ResNet-50.**

---

## 🌟 Key Highlights

- 🤝 **Tri-Model Multi-CNN Ensemble**: Coordinates VGG-16, MobileNet, and ResNet-50 into a unified soft-voting fusion pipeline to minimize classification variance.
- ⚡ **Real-Time Multi-Model Comparison**: Simultaneously inspects individual predictions and confidence scores for each backbone architecture side-by-side.
- 🎨 **Minimalist Monochrome UI**: Clean, high-contrast black-and-white theme designed for clear radiological review.
- 🔒 **100% Local & Privacy-Preserving**: All inference runs entirely on-device with zero cloud telemetry.

---

## 🎯 Supported Pathological Classes

| Index | Pathology | Severity Profile | Histological Origin | Clinical Management Path |
|:---:|:---|:---:|:---|:---|
| **0** | **Glioma** | 🔴 High | Glial cells (astrocytes, oligodendrocytes, ependymal) | Surgical resection, radiotherapy, adjuvant temozolomide chemotherapy |
| **1** | **Meningioma** | 🟡 Low–Medium | Arachnoid cap cells of protective meninges | Active surveillance, neurosurgical resection, stereotactic radiosurgery |
| **2** | **Pituitary Tumor** | 🟢 Low (Benign) | Anterior pituitary gland within the sella turcica | Dopamine agonists / somatostatin analogs, transsphenoidal surgery |

---

## 🏗️ Model Architecture & Ensemble Pipeline

### Constituent Models

- **VGG-16** (`final_brain_tumor_model_main.h5`): ~14.7M parameters — Deep sequential convolutional filters extracting fine-grained spatial feature hierarchies.
- **MobileNet** (`final_brain_tumor_model_main_mobilenet.h5`): ~3.2M parameters — Depthwise separable convolutions offering lightweight, regularized representations.
- **ResNet-50** (`final_brain_tumor_model_main_resnet50.h5`): ~23.5M parameters — Deep residual skip connections mitigating gradient vanishing across complex textures.

### Ensemble Fusion Strategy
The soft-voting ensemble computes the mean probability distribution across all 3 models:

$$\hat{P} = \frac{P_{\text{VGG}} + P_{\text{MobileNet}} + P_{\text{ResNet}}}{3}$$

The final prediction is selected by maximum likelihood:

$$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} \hat{P}_c$$

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Harshgup16/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application

- **Option A (One-Click Windows Launcher)**:
  ```powershell
  .\run_app.bat
  ```

- **Option B (Standard Streamlit)**:
  ```bash
  streamlit run app.py
  ```

- **Option C (Conda Environment)**:
  ```bash
  conda activate brain_tumor
  streamlit run app.py
  ```

Open **http://localhost:8501** in your browser to start detecting brain tumors.

---

## 📁 Repository Structure

```
brain-tumor-detection/
├── app.py                                       # Main Streamlit ensemble web application
├── final_brain_tumor_model_main.h5              # Trained VGG-16 model weights
├── final_brain_tumor_model_main_mobilenet.h5    # Trained MobileNet model weights
├── final_brain_tumor_model_main_resnet50.h5     # Trained ResNet-50 model weights
├── run_app.bat                                  # One-click Windows CMD launcher
├── run_app.ps1                                  # PowerShell launcher script
├── requirements.txt                             # Python package dependencies
├── model-1.ipynb                                # MobileNet training notebook
├── model-2.ipynb                                # ResNet-50 training notebook
├── model-3.ipynb                                # VGG-16 training notebook
├── .streamlit/
│   └── config.toml                              # Streamlit theme configuration
└── README.md                                    # Project documentation
```

---

## 🛡️ Medical Disclaimer

> **Research and Educational Prototype Only**: This application is developed strictly for academic evaluation, algorithm benchmarking, and educational demonstrations. It is not an FDA/CE-cleared medical device and should never replace professional radiological evaluation, histological biopsy, or physician clinical judgment.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.
