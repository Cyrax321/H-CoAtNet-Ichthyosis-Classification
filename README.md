<div align="center">

---

### ⚠️ REVIEW-ONLY NOTICE

**This repository is provided solely for peer review and reproducibility purposes**  
**associated with the submitted manuscript.**

Reuse, redistribution, modification, or deployment of any code, data, or results  
contained herein is **strictly prohibited** without explicit written permission from the authors.

© 2026 The Authors. All Rights Reserved.

---

</div>

---

# **H-CoAtNet**

## **Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification**

## **Official Research Codebase**

This repository contains the **reference implementation** of **H-CoAtNet**, a hierarchically enhanced hybrid convolution–transformer framework for **multi-class Ichthyosis subtype classification** from dermatological images.

The release supports **methodological verification, benchmarking, and reproducibility** for rare disease medical image analysis.

---

## 📄 **Associated Paper**

**Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet **

Athul Joe Joseph Palliparambil, Anandhu P Shaji, Rajeev Rajan
*(Under Review, 2025)*

---

## 🔧 **Repository Structure and Execution Context (Critical)**

After cloning, note that the **actual project root** is the inner `H-CoAtNet/` directory.

```bash
git clone https://github.com/Cyrax321/H-CoAtNet-Ichthyosis.git
cd H-CoAtNet-Ichthyosis
cd H-CoAtNet
```

All commands **must be executed from this directory**.
Running commands from the outer directory will result in missing file or module errors.

```
H-CoAtNet/
├── README.md
├── requirements.txt
├── proposed_method/
│   └── train_h_coatnet.py
└── baselines/
    ├── train_cnn.py
    ├── train_efficientnet.py
    ├── train_vit.py
    ├── train_swin.py
    ├── train_coatnet.py
    └── train_gft.py
```

---

## 1. **Environment Setup**

### Dependencies

```bash
pip install -r requirements.txt
```

**Core Requirements**

* Python ≥ 3.9
* PyTorch
* timm
* torchvision
* scikit-learn
* numpy, pandas, matplotlib
* roboflow

Tested on **macOS (Apple Silicon)** and **Linux** environments.

---

## 2. **Problem Overview**

Ichthyosis represents a heterogeneous group of rare genetic skin disorders characterized by abnormal keratinization and severe scaling. Automated classification is challenging due to:

* Extreme **class imbalance**
* **Subtle morphological differences** between subtypes
* **Limited annotated medical datasets**

H-CoAtNet addresses these challenges through **hybrid convolution–transformer modeling** with hierarchical feature refinement.

---

## 3. **Method Overview**

H-CoAtNet integrates three core architectural components:

* Convolutional stem for localized texture and scale modeling
* Transformer blocks for global contextual dependency learning
* Hierarchical squeeze-excitation with progressive token selection

This design balances **inductive bias**, **global reasoning**, and **computational efficiency**, optimized for rare disease image classification.

---

## 4. **Dataset Description**

The dataset used in this study contains **1,580 dermatological images** distributed across **five diagnostic categories**:

* Harlequin Ichthyosis (HI)
* Ichthyosis Vulgaris (IV)
* Lamellar Ichthyosis (LI)
* Netherton Syndrome (NS)
* Healthy Skin

Images are resized to **224 × 224**, normalized using ImageNet statistics, and split using **stratified 70/15/15 train–validation–test partitions**.

> The dataset reflects clinically realistic prevalence while ensuring sufficient representation of rare subtypes.

---

##  5. **Dataset Access and API Configuration (Required Before Running Code)**

To ensure **controlled access, versioning, and reproducibility**, the dataset is hosted using **Roboflow**.

### 📎 Dataset Project Link

 **Roboflow Dataset Page**
[https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj](https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj)

---

## **How to Access the Data**

To obtain your **Roboflow API key**, follow these steps:

1. Click the **dataset project link** above.
2. Navigate to **Dataset** in the left sidebar.
3. Click **Download Dataset**.
4. Select **Download Dataset (Get a code snippet or ZIP file)**.
5. Ensure **Show download code** is enabled.
6. Choose:
   **“Custom train this dataset using the provided code snippet in a notebook.”**
7. Copy **only the API key string** from the snippet.

### Example API key format

```python
api_key="xxxxxxxxxxxxxxxxxxx"
```

---

## 🧩 **Adding the API Key to the Code (Mandatory)**

Before executing **any training script**, the API key must be added.

### Example: H-CoAtNet training script

Open:

```
proposed_method/train_h_coatnet.py
```

Add the configuration block near the top of the file:

```python
#  Configuration
API_KEY = "PASTE_YOUR_KEY_HERE"

```

Then initialize the dataset:

```python
from roboflow import Roboflow

rf = Roboflow(api_key=API_KEY)
```

**Important notes**

* Use the **same dataset version** for all baseline and proposed models.

---

## 6. **Training and Execution**

### Proposed Method (H-CoAtNet)

```bash
python proposed_method/train_h_coatnet.py
```

### Baseline Models

```bash
python baselines/train_cnn.py
python baselines/train_efficientnet.py
python baselines/train_vit.py
python baselines/train_swin.py
python baselines/train_coatnet.py
python baselines/train_gft.py
```

All models use **identical dataset splits, preprocessing, and evaluation protocols**.

---

## 7. **Experimental Protocol (Reproducibility)**

* Optimizer: Adam
* Epochs: 30
* Dropout: 0.2
* Weight decay enabled
* No external pretraining (trained from scratch)
* Fixed random seeds

### Hardware

* Apple MacBook Pro (M3 Pro, 18 GB RAM)
* Google Colab (verification only)

No TPU-specific optimizations are used.

---

## 8. **Evaluation Metrics**

* Accuracy
* Macro-averaged Precision, Recall, F1-score
* Weighted F1-score

Macro-averaged metrics are emphasized due to **class imbalance** inherent in rare disease datasets.

---

## 9. **Results Summary**

| Model                | Accuracy   | Macro F1   | Weighted F1 |
| -------------------- | ---------- | ---------- | ----------- |
| **H-CoAtNet (Ours)** | **90.51%** | **0.8605** | **0.9024**  |
| Swin Transformer     | 82.91%     | 0.7477     | 0.8150      |
| GFT                  | 82.28%     | 0.7701     | 0.8221      |
| CoAtNet              | 74.68%     | 0.6517     | 0.7463      |
| Vision Transformer   | 72.15%     | 0.6310     | 0.7103      |
| CNN                  | 69.62%     | 0.6085     | 0.6889      |
| EfficientNet-B0      | 66.46%     | 0.5938     | 0.6675      |

---

## 10. **Ethical Considerations**

* No patient-identifiable data is used
* Images are anonymized and sourced from publicly available materials
* Intended strictly as a **clinical decision-support system**, not a standalone diagnostic tool

---

## 11. **Contact**

**Anandhu P. Shaji**
Email: [reach.anandhu.me@gmail.com](mailto:reach.anandhu.me@gmail.com)

---

