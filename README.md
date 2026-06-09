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
## **Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification(H-CoAtNet)**

## **Official Research Codebase**

This repository contains the **reference implementation** of **H-CoAtNet**, a hierarchically enhanced hybrid convolution–transformer framework for **multi-class Ichthyosis subtype classification** from dermatological images.

The release supports **methodological verification, benchmarking, and reproducibility** for rare disease medical image analysis.

---

## 📄 **Associated Paper**

**Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet **

Athul Joe Joseph Palliparambil, Anandhu P Shaji, Rajeev Rajan
*(Under Review)*

---

## 🔧 **Repository Structure and Execution Context (Critical)**

After cloning, note that the **actual project root** is the inner `H-CoAtNet/` directory.

```bash
git clone https://github.com/Cyrax321/H-CoAtNet-Ichthyosis-Classification.git
cd H-CoAtNet-Ichthyosis-Classification/H-CoAtNet
```

All commands **must be executed from this directory**.
Running commands from the outer directory will result in missing file or module errors.

```
H-CoAtNet/
├── requirements.txt
├── proposed/
│   └── train_h_coatnet.py              # Proposed model — H-CoAtNet
├── baselines/
│   ├── pretrained/                     # Fine-tuned from ImageNet weights
│   │   ├── train_efficientnet_b0.py    # EfficientNet-B0
│   │   ├── train_swin_t.py             # Swin Transformer Tiny
│   │   ├── train_vit_b16.py            # ViT-B/16
│   │   ├── train_coatnet.py            # CoAtNet
│   │   ├── train_gft.py                # GFT
│   │   ├── train_biomedclip.py         # BiomedCLIP (foundation model)
│   │   └── train_dinov2.py             # DINOv2 (foundation model)
│   └── scratch/                        # Trained from random initialisation
│       ├── train_cnn.py
│       ├── train_efficientnet_b0.py
│       ├── train_swin_t.py
│       └── train_vit.py
└── evaluation/
    ├── crossval.py                      # 5-fold cross-validation + McNemar
    ├── ablation.py                      # Ablation study (4 conditions)
    └── gradcam.py                       # Grad-CAM visualizations
```

All commands must be run from inside the `H-CoAtNet/` directory.lassification is challenging due to:

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
proposed/train_h_coatnet.py
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
python proposed/train_h_coatnet.py
```

### Pretrained Baselines (Primary Comparisons)

```bash
python baselines/pretrained/train_efficientnet_b0.py
python baselines/pretrained/train_swin_t.py
python baselines/pretrained/train_vit_b16.py
python baselines/pretrained/train_coatnet.py
python baselines/pretrained/train_gft.py
python baselines/pretrained/train_biomedclip.py    # pip install open_clip_torch
python baselines/pretrained/train_dinov2.py         # pip install transformers
```

### From-Scratch Baselines

```bash
python baselines/scratch/train_cnn.py
python baselines/scratch/train_efficientnet_b0.py
python baselines/scratch/train_swin_t.py
python baselines/scratch/train_vit.py
```
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
```

---

## 4. **Ablation Study**

```bash
python evaluation/ablation.py --condition full
python evaluation/ablation.py --condition no_hse
python evaluation/ablation.py --condition no_transformer
python evaluation/ablation.py --condition baseline
```

Results are appended to `ablation_results.csv` after each run.

---

## 5. **Cross-Validation and Statistical Tests**

```bash
python evaluation/crossval.py
```

Outputs:
- `crossval_summary.txt` — mean ± std
- `crossval_results.csv` — per-fold breakdown
- `mcnemar_results.csv` — χ² and p-values vs all baselines
- `fold_{k}_cm.png` — confusion matrix per fold (300 DPI)

---

## 6. **Grad-CAM Interpretability**

```bash
python evaluation/gradcam.py --checkpoint best_h_coatnet.pth
```

Outputs:
- `gradcam/<ClassName>_sample<N>.png` — individual overlays at 300 DPI
- `gradcam/gradcam_grid.png` — publication-quality 5×3 grid

---

## 7. **Hyperparameter Table**

| Parameter | H-CoAtNet | Pretrained Baselines |
|---|---|---|
| Backbone | ConvNeXt-Tiny (pretrained=True) | Model-specific (pretrained=True) |
| Input resolution | 224 × 224 | 224 × 224 |
| Batch size | 24 | 24 (ViT: 16) |
| Epochs | 30 | 30 |
| Optimiser | AdamW | AdamW (layer-wise LR) |
| Backbone LR | 5e-5 | 1e-5 |
| Head LR | 5e-5 | 1e-4 |
| Weight decay | 0.01 | 0.01 |
| LR schedule | CosineAnnealing (T_max=30) | CosineAnnealing |
| Loss | CrossEntropy (label_smoothing=0.1, class_weights=True) | Same |
| Dropout | 0.2 | 0.2 |
| Random seed | 42 | 42 |
| ViT blocks | 2 (dim=192, heads=6) | — |
| HSE stages | 2 (keep 75%, then 50% tokens) | — |
| Augmentation | RandomResizedCrop, HFlip, Rotation(15°), TrivialAugmentWide, RandomErasing(p=0.2) | Same |

---

## 8. **Results Summary**

> **Note:** From-scratch baselines are retained for reference. Pretrained baselines are the primary comparison.

| Model | Training | Accuracy | Macro F1 |
|---|---|---|---|
| **H-CoAtNet (Ours)** | Pretrained backbone | **To be updated after CV** | — |
| EfficientNet-B0 | Pretrained (ImageNet) | — | — |
| Swin-T | Pretrained (ImageNet) | — | — |
| ViT-B/16 | Pretrained (ImageNet-21k) | — | — |
| CoAtNet | Pretrained | — | — |
| GFT | Pretrained | — | — |
| EfficientNet-B0 | From scratch | 66.46% | 0.5938 |
| Swin Transformer | From scratch | 82.91% | 0.7477 |
| ViT | From scratch | 72.15% | 0.6310 |

*Update this table with 5-fold CV mean ± std results from `crossval_summary.txt`.*

---

## 9. **Ethical Considerations**

* No patient-identifiable data is used
* Images are sourced from publicly available materials
* Intended strictly as a clinical decision-support tool, not a standalone diagnostic system

---

## 10. **Contact**

**Anandhu P. Shaji** — [reach.anandhu.me@gmail.com](mailto:reach.anandhu.me@gmail.com)

---

## ⚖️ Legal Notice & Copyright

```
Copyright © 2026 The Authors. All Rights Reserved.

This repository, titled “Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification (H-CoAtNet)”, and all associated materials — including but not limited to source code, experimental pipelines, benchmark datasets, execution logs, research documentation, and the accompanying manuscript — are provided solely for the purposes of peer review, validation, and reproducibility assessment in connection with the submitted work:

“Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet”
Athul Joe Joseph Palliparambil, Anandhu P Shaji, Rajeev Rajan (Under Review, 2025)

This repository constitutes the official research codebase for a hierarchically enhanced hybrid convolution–transformer framework designed for multi-class Ichthyosis subtype classification from dermatological imagery.
This is NOT an open-source release.

Restrictions

 • The following actions are strictly prohibited without prior explicit written consent from the authors:
 • Reuse of any portion of the codebase in external projects, systems, or products
 • Redistribution of this repository or its contents, in full or in part
 • Modification, adaptation, translation, or creation of derivative works
 • Deployment of the framework or its components in production or clinical systems
 • Citation or reference to unpublished results prior to formal publication
 • Utilization of repository contents for training machine learning models

Legal Notice
Unauthorized use may constitute copyright infringement, intellectual property violation, and/or misappropriation of unpublished academic material.
```

> **Permitted use:** Reviewers assigned by the programme committee may read, compile, and run the code solely for the purpose of evaluating the submitted manuscript. No other use is permitted.

---

<div align="center">


