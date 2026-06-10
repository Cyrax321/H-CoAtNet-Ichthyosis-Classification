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
## **Wavelet-enhanced Convolutional Attention Network for Ichthyosis Classification (WaveCoAtNet)**

## **Official Research Codebase**

This repository contains the **reference implementation** of **WaveCoAtNet**, a wavelet-enhanced convolutional attention network with prototype-anchored token selection for **multi-class Ichthyosis subtype classification** from dermatological images.

The release supports **methodological verification, benchmarking, and reproducibility** for rare disease medical image analysis.

---

## 📄 **Associated Paper**

**Wavelet-enhanced Classification of Ichthyosis Variants in Dermatological Images Using WaveCoAtNet**

Athul Joe Joseph Palliparambil, Anandhu P Shaji, Rajeev Rajan
*(Under Review)*

---

## 🔧 **Repository Structure**

After cloning, the **actual project root** is the inner `H-CoAtNet/` directory.

```bash
git clone https://github.com/Cyrax321/H-CoAtNet-Ichthyosis-Classification.git
cd H-CoAtNet-Ichthyosis-Classification/H-CoAtNet
```

All commands **must be executed from this directory**.

```
H-CoAtNet/
├── requirements.txt
├── proposed/
│   └── train_h_coatnet.py              # WaveCoAtNet (proposed)
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
    ├── ablation.py                      # Ablation study (8 conditions)
    ├── gradcam.py                       # Grad-CAM visualizations
    └── generate_visualizations.py       # 13 publication-quality figures
```

---

## 1. **Problem Statement**

Automated ichthyosis subtype classification is challenging due to:

* Extreme **class imbalance** across rare subtypes
* **Subtle morphological differences** (e.g., plate-like fissures vs. fish-scale patterns)
* **Limited annotated medical datasets** for rare dermatological conditions

WaveCoAtNet addresses these through three novel architectural contributions combining frequency-aware feature fusion, prototype-guided token selection, and contrastive representation learning.

---

## 2. **Method Overview — Novel Contributions**

WaveCoAtNet integrates three independently ablatable modules:

### Wavelet-Guided Frequency-Decomposed Cross-Attention (WG-FDCA)

Decomposes early CNN features via 2D Haar DWT into structure (LL sub-band) and texture (LH+HL+HH) frequency streams. Dual-stream cross-attention with a learnable per-token frequency gate dynamically balances structure vs. texture based on image content — plate-like fissures in Harlequin Ichthyosis (low-freq) vs. fish-scale patterns in Ichthyosis Vulgaris (high-freq).

### Prototype-Anchored Dynamic Token Selection (PA-DTS)

Selects diagnostically relevant tokens by scoring them against learnable class prototypes (updated via EMA). Token importance combines three signals: prototype affinity, affinity entropy (retains ambiguous boundary tokens), and channel attention. An adaptive keep-ratio predictor adjusts pruning aggressiveness per image.

### Supervised Contrastive Token Regularization (SCTR)

Auxiliary SupCon loss on mean-pooled token embeddings. Forces same-class representations to cluster and different-class to separate, improving inter-class discriminability for rare subtypes like Netherton Syndrome.

### Architecture Flow

```
Input (224×224) → ConvNeXt Stem → Stage 1 (96ch, 56×56)
                                → Stage 2 (192ch, 28×28)
    Stage 1 → Haar DWT → LL / (LH+HL+HH) → WG-FDCA ← Stage 2 queries
    → Positional Embedding → ViT Blocks (2×, dim=192, heads=6)
    → Reshape → Stage 3 (384ch, 14×14) → Stage 4 (768ch, 7×7)
    → Flatten → PA-DTS (adaptive token selection)
    → Mean Pool → LayerNorm → Classifier
    → SCTR (training only): CE + 0.1 × SupCon(T=0.07)
```

---

## 3. **Dataset Description**

The dataset contains **1,580 dermatological images** across **five diagnostic categories**:

* Harlequin Ichthyosis (HI)
* Ichthyosis Vulgaris (IV)
* Lamellar Ichthyosis (LI)
* Netherton Syndrome (NS)
* Healthy Skin

Images are resized to **224 × 224**, normalized using ImageNet statistics, and split using **stratified 70/15/15 train–validation–test partitions**.

> The dataset reflects clinically realistic prevalence while ensuring sufficient representation of rare subtypes.

---

## 4. **Dataset Access and API Configuration**

The dataset is hosted on **Roboflow** for controlled access and versioning.

**Roboflow Dataset Page:**
[https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj](https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj)

### How to Access

1. Click the dataset link above
2. Navigate to **Dataset** → **Download Dataset**
3. Enable **Show download code** and copy the API key

### Adding the API Key

Open `proposed/train_h_coatnet.py` and set:

```python
API_KEY = "PASTE_YOUR_KEY_HERE"
```

Use the **same dataset version** for all models.

---

## 5. **Training and Execution**

### Proposed Method (WaveCoAtNet)

```bash
python proposed/train_h_coatnet.py
```

### Pretrained Baselines

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

## 6. **Ablation Study (8 Conditions)**

```bash
python evaluation/ablation.py --condition full
python evaluation/ablation.py --condition no_wgfdca
python evaluation/ablation.py --condition no_transformer
python evaluation/ablation.py --condition no_padts
python evaluation/ablation.py --condition no_sctr
python evaluation/ablation.py --condition fixed_pruning
python evaluation/ablation.py --condition no_prototypes
python evaluation/ablation.py --condition baseline
```

| Condition | Description |
|---|---|
| `full` | WaveCoAtNet (all modules) |
| `no_wgfdca` | Plain cross-attention (no wavelet decomposition) |
| `no_transformer` | No ViT blocks, no cross-attention |
| `no_padts` | Global average pooling (no token selection) |
| `no_sctr` | CE loss only (no contrastive regularization) |
| `fixed_pruning` | Fixed SE pruning (75% → 50%) |
| `no_prototypes` | SE-based selection without class prototypes |
| `baseline` | Plain ConvNeXt-Tiny fine-tuned |

Results append to `ablation_results.csv`.

---

## 7. **Cross-Validation and Statistical Tests**

```bash
python evaluation/crossval.py
```

Outputs:
- `crossval_summary.txt` — mean ± std
- `crossval_results.csv` — per-fold breakdown
- `mcnemar_results.csv` — χ² and p-values vs all baselines
- `fold_{k}_cm.png` — confusion matrix per fold (300 DPI)

---

## 8. **Grad-CAM Interpretability**

```bash
python evaluation/gradcam.py --checkpoint best_h_coatnet.pth
```

Outputs:
- `gradcam/<ClassName>_sample<N>.png` — individual overlays at 300 DPI
- `gradcam/gradcam_grid.png` — publication-quality 5×3 grid

---

## 9. **Publication Figure Generation**

```bash
python evaluation/generate_visualizations.py
```

Generates 13 publication-quality figures (300 DPI) in the `figures/` directory:

| Figure | Description |
|---|---|
| ROC curves | Per-class and macro-averaged |
| Precision-Recall curves | Per-class |
| Model comparison bar chart | Accuracy + Macro F1 + Weighted F1 |
| Confusion matrix comparison | Side-by-side all models |
| t-SNE embeddings | Feature space visualisation |
| Dataset sample grid | Representative images per class |
| Class distribution | Bar chart with counts |
| Per-class F1 heatmap | Models × classes |
| McNemar p-value heatmap | Statistical significance matrix |
| Model efficiency bubble | Accuracy vs. parameters vs. speed |
| Ablation bar chart | Module contribution analysis |
| Failure analysis grid | Misclassified samples |
| Comprehensive results table | All metrics as image |

---

## 10. **Hyperparameter Table**

| Parameter | WaveCoAtNet | Pretrained Baselines |
|---|---|---|
| Backbone | ConvNeXt-Tiny (pretrained=True) | Model-specific (pretrained=True) |
| Input resolution | 224 × 224 | 224 × 224 |
| Batch size | 24 | 24 (ViT: 16) |
| Epochs | 30 | 30 |
| Optimiser | AdamW | AdamW (layer-wise LR) |
| Learning rate | 5e-5 | 1e-5 (backbone), 1e-4 (head) |
| Weight decay | 0.01 | 0.01 |
| LR schedule | CosineAnnealing (T_max=30) | CosineAnnealing |
| Loss | CE(label_smoothing=0.1, class_weights) + 0.1×SupCon(T=0.07) | CE(label_smoothing=0.1) |
| Dropout | 0.2 | 0.2 |
| Random seed | 42 | 42 |
| ViT blocks | 2 (dim=192, heads=6) | — |
| WG-FDCA | Haar DWT, 4-head dual-stream cross-attn | — |
| PA-DTS | 5 prototypes, EMA=0.999, keep 30–80% | — |
| SCTR weight | λ=0.1, T=0.07, proj_dim=128 | — |
| Augmentation | RandomResizedCrop, HFlip, Rotation(15°), TrivialAugmentWide, RandomErasing(p=0.2) | Same |

---

## 11. **Results Summary**

> **Note:** From-scratch baselines are retained for reference. Pretrained baselines are the primary comparison.

| Model | Training | Accuracy | Macro F1 |
|---|---|---|---|
| **WaveCoAtNet (Ours)** | Pretrained backbone | **To be updated after CV** | — |
| EfficientNet-B0 | Pretrained (ImageNet) | — | — |
| Swin-T | Pretrained (ImageNet) | — | — |
| ViT-B/16 | Pretrained (ImageNet-21k) | — | — |
| CoAtNet | Pretrained | — | — |
| GFT | Pretrained | — | — |
| BiomedCLIP | Foundation model | — | — |
| DINOv2 | Foundation model | — | — |
| EfficientNet-B0 | From scratch | 66.46% | 0.5938 |
| Swin Transformer | From scratch | 82.91% | 0.7477 |
| ViT | From scratch | 72.15% | 0.6310 |

*Update this table with 5-fold CV mean ± std results from `crossval_summary.txt`.*

---

## 12. **Experimental Protocol (Reproducibility)**

* Optimizer: AdamW (lr=5e-5, weight_decay=0.01)
* LR schedule: CosineAnnealingLR
* Epochs: 30
* Dropout: 0.2
* Loss: CrossEntropy + SupCon (λ=0.1)
* Fixed random seed: 42
* All results reproducible with `torch.backends.cudnn.deterministic = True`

### Hardware

* Apple MacBook Pro (M3 Pro, 18 GB RAM)
* Google Colab T4 GPU (verification and reproducibility)

---

## 13. **Ethical Considerations**

* No patient-identifiable data is used
* Images are sourced from publicly available materials
* Intended strictly as a clinical decision-support tool, not a standalone diagnostic system

---

## 14. **Contact**

**Anandhu P. Shaji** — [reach.anandhu.me@gmail.com](mailto:reach.anandhu.me@gmail.com)

---

## ⚖️ Legal Notice & Copyright

```
Copyright © 2026 The Authors. All Rights Reserved.

This repository, titled "Wavelet Frequency-decomposed Prototype-anchored Learning for Ichthyosis
Classification (WaveCoAtNet)", and all associated materials — including but not
limited to source code, experimental pipelines, benchmark datasets, execution logs,
research documentation, and the accompanying manuscript — are provided solely for
the purposes of peer review, validation, and reproducibility assessment.

"Wavelet Frequency-decomposed Prototype-anchored Classification of Ichthyosis Variants
in Dermatological Images Using WaveCoAtNet"
Athul Joe Joseph Palliparambil, Anandhu P Shaji, Rajeev Rajan (Under Review, 2025)

Unauthorized reproduction, redistribution, modification, or commercial deployment
of any content within this repository is strictly prohibited.
```
