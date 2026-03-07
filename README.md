# 🦴 Bone Fracture Detection
### Deep Learning Medical Imaging Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![Accuracy](https://img.shields.io/badge/Accuracy-97.37%25-brightgreen)
![HuggingFace](https://img.shields.io/badge/🤗-Live%20Demo-yellow)

> AI-powered bone fracture detection from X-ray images using ResNet50 transfer learning. Achieves 97.37% test accuracy with OOD detection, Grad-CAM explainability, and automated PDF report generation.

**🔗 Live Demo:** [huggingface.co/spaces/Nainika0205/BoneFractureDetection](https://huggingface.co/spaces/Nainika0205/BoneFractureDetection)

---

## Overview

Given an X-ray image, the system:
1. Verifies the image is a valid bone X-ray (OOD detection)
2. Classifies as **Fracture** or **Normal**
3. Assigns severity (Mild / Moderate / Severe)
4. Detects fracture location using Grad-CAM
5. Generates a downloadable PDF diagnostic report

---

## Features

| Feature | Description |
|---------|-------------|
| **Fracture Detection** | Binary classification: Fracture / Normal |
| **OOD Detection** | Mahalanobis Distance rejects non-bone images |
| **Severity Scoring** | Mild / Moderate / Severe |
| **Location Detection** | Anatomical region via Grad-CAM heatmap |
| **Clinical Recommendations** | Next steps tailored to severity |
| **PDF Report** | Professional report with patient info and Grad-CAM |
| **REST API** | FastAPI endpoint for programmatic access |

---

## Model Architecture

- **Base Model:** ResNet50 (pretrained on ImageNet)
- **Trainable Layers:** Layer3, Layer4, FC
- **Custom FC:** Dropout(0.4) -> Linear(2048 -> 2)
- **Optimizer:** Adam (lr=0.0001, weight_decay=1e-4)
- **Loss:** CrossEntropyLoss with class weights [0.0597, 0.9403]
- **Early Stopping:** patience=5

---

## Dataset

- **Source:** [Kaggle - Bone Fracture Dataset](https://www.kaggle.com/datasets/orvile/bone-fracture-dataset)
- **Total Images:** 2127 (Fracture: 2000, Normal: 127)
- **Split:** Train 1900 / Val 380 / Test 380
- **Class Imbalance Fix:** Augmentation + WeightedRandomSampler

---

## Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **97.37%** |
| **ROC AUC** | 0.9970 |
| Fracture F1 | 0.98 |
| Normal F1 | 0.94 |

### OOD Detection
- Bone X-ray acceptance rate: 98.8%
- Threshold: 494.47 (99th percentile)
- Brain MRI, random images: REJECTED

---

## Installation
```
git clone https://github.com/nainika-2504/BoneFractureDetection
cd BoneFractureDetection
pip install -r requirements.txt
python app/gradio_app.py
```

---

## Project Structure
```
BoneFractureDetection/
├── app/
│   ├── app.py
│   └── gradio_app.py
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── gradcam_results.png
├── requirements.txt
└── README.md
```

Model files hosted on Hugging Face due to GitHub 100MB limit.

---

## Tech Stack

Python | PyTorch | ResNet50 | Grad-CAM | Gradio | FastAPI | ReportLab | Hugging Face Spaces

---

## Disclaimer

For educational purposes only. Not a substitute for professional medical advice.

---

## Author
Nainika M 
