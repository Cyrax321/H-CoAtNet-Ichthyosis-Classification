# WaveCoAtNet — Google Colab Execution Guide

> **Run every experiment from inside `H-CoAtNet/`.**
> All commands assume that directory as the working directory.

---

## Setup (Run these cells first, in order)

### Cell 1 — Mount Drive & Clone Repo

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/HCoAtNet_experiments', exist_ok=True)
%cd /content/drive/MyDrive/HCoAtNet_experiments
```

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments
git clone https://github.com/Cyrax321/H-CoAtNet-Ichthyosis-Classification.git
cd H-CoAtNet-Ichthyosis-Classification/H-CoAtNet
pip install -q -r requirements.txt
```

### Cell 2 — Verify GPU

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    raise RuntimeError("GPU required! Runtime → Change runtime type → T4 GPU")
```

---

## Phase 1: Train WaveCoAtNet (Proposed Method)

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis-Classification/H-CoAtNet
python proposed/train_h_coatnet.py 2>&1 | tee logs_h_coatnet.txt
```

**Output files:**
- `best_h_coatnet.pth` — best checkpoint
- `h_coatnet_y_true.npy`, `h_coatnet_y_pred.npy` — test predictions
- `confusion_matrix_h_coatnet.png`
- `h_coatnet_loss_curves.png`, `h_coatnet_acc_curves.png`

---

## Phase 2: Train All Baselines

### Pretrained Baselines (ImageNet fine-tuned)

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis-Classification/H-CoAtNet

echo "=== EfficientNet-B0 (Pretrained) ==="
python baselines/pretrained/train_efficientnet_b0.py 2>&1 | tee logs_efficientnet_pretrained.txt

echo "=== Swin-T (Pretrained) ==="
python baselines/pretrained/train_swin_t.py 2>&1 | tee logs_swin_pretrained.txt

echo "=== ViT-B/16 (Pretrained) ==="
python baselines/pretrained/train_vit_b16.py 2>&1 | tee logs_vit_pretrained.txt

echo "=== CoAtNet ==="
python baselines/pretrained/train_coatnet.py 2>&1 | tee logs_coatnet.txt

echo "=== GFT ==="
python baselines/pretrained/train_gft.py 2>&1 | tee logs_gft.txt
```

### Foundation Model Baselines

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis-Classification/H-CoAtNet

echo "=== BiomedCLIP ==="
python baselines/pretrained/train_biomedclip.py 2>&1 | tee logs_biomedclip.txt

echo "=== DINOv2 ==="
python baselines/pretrained/train_dinov2.py 2>&1 | tee logs_dinov2.txt
```

### From-Scratch Baselines

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis-Classification/H-CoAtNet

echo "=== CNN (Scratch) ==="
python baselines/scratch/train_cnn.py 2>&1 | tee logs_cnn_scratch.txt

echo "=== EfficientNet-B0 (Scratch) ==="
python baselines/scratch/train_efficientnet_b0.py 2>&1 | tee logs_efficientnet_scratch.txt

echo "=== Swin-T (Scratch) ==="
python baselines/scratch/train_swin_t.py 2>&1 | tee logs_swin_scratch.txt

echo "=== ViT (Scratch) ==="
python baselines/scratch/train_vit.py 2>&1 | tee logs_vit_scratch.txt
```

---

## Phase 3: Ablation Study (8 conditions)

Run **after** WaveCoAtNet training is complete.

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis-Classification/H-CoAtNet

for cond in full no_wgfdca no_transformer no_padts no_sctr fixed_pruning no_prototypes baseline; do
    echo "=== Ablation: $cond ==="
    python evaluation/ablation.py --condition $cond 2>&1 | tee ablation_${cond}.txt
done
```

**Output:** `ablation_results.csv` — paste directly into your paper's ablation table.

| Condition | What it tests |
|---|---|
| `full` | WaveCoAtNet (all modules) |
| `no_wgfdca` | Removes wavelet decomposition (plain cross-attention) |
| `no_transformer` | No ViT blocks, no cross-attention |
| `no_padts` | Global average pooling instead of PA-DTS |
| `no_sctr` | No contrastive loss (CE only) |
| `fixed_pruning` | Fixed SE pruning instead of adaptive prototype-based |
| `no_prototypes` | SE-based selection without class prototypes |
| `baseline` | Plain ConvNeXt-Tiny fine-tuned |

---

## Phase 4: Cross-Validation + McNemar's Test

Run **after** all models are trained (needs all `*_y_pred.npy` files).

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis-Classification/H-CoAtNet
python evaluation/crossval.py 2>&1 | tee logs_crossval.txt
```

**Output files:**
- `crossval_summary.txt` — mean ± std (copy into Results section)
- `crossval_results.csv` — per-fold metrics
- `mcnemar_results.csv` — p-values for statistical significance
- `fold_1_cm.png` through `fold_5_cm.png`

---

## Phase 5: Grad-CAM Visualizations

Run **after** WaveCoAtNet training is complete.

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis-Classification/H-CoAtNet
python evaluation/gradcam.py --checkpoint best_h_coatnet.pth 2>&1 | tee logs_gradcam.txt
```

**Output:** `gradcam/gradcam_grid.png` — publication-quality figure.

---

## Phase 6: Generate All Publication Figures

Run **after** all models are trained (needs all `*_y_pred.npy` files).

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis-Classification/H-CoAtNet
python evaluation/generate_visualizations.py 2>&1 | tee logs_visualizations.txt
```

**Output figures (in `figures/` directory):**

| Figure | Filename | Paper section |
|---|---|---|
| ROC curves (per-class) | `roc_curves_all.png` | Results |
| Precision-Recall curves | `pr_curves_all.png` | Results |
| Model comparison bar chart | `model_comparison_bar.png` | Results |
| Confusion matrix comparison | `confusion_matrix_comparison.png` | Results |
| t-SNE embeddings | `tsne_embeddings.png` | Results |
| Dataset sample grid | `dataset_samples.png` | Dataset |
| Class distribution | `class_distribution.png` | Dataset |
| Per-class F1 heatmap | `per_class_f1_heatmap.png` | Results |
| McNemar p-value heatmap | `statistical_significance.png` | Results |
| Model efficiency bubble | `model_efficiency_bubble.png` | Discussion |
| Ablation bar chart | `ablation_comparison_bar.png` | Ablation |
| Failure analysis grid | `failure_analysis.png` | Discussion |
| Comprehensive results table | `comprehensive_results_table.png` | Results |

**CSV exports:** `comprehensive_results.csv`, `mcnemar_pvalues.csv`

---

## Estimated Training Times (T4 GPU)

| Script | Time |
|---|---|
| `train_h_coatnet.py` (proposed) | ~40–55 min |
| Pretrained baselines (5 models) | ~2–3 hours total |
| Foundation models (2 models) | ~1–1.5 hours total |
| From-scratch baselines (4 models) | ~2–3 hours total |
| Ablation study (8 conditions) | ~4–6 hours total |
| Cross-validation (5 folds × 30 epochs) | ~3–4 hours |
| Grad-CAM + visualizations | ~5–10 min |

**Total: ~13–18 hours. Use Colab Pro or split across sessions. All checkpoints save to Drive.**

---

## Troubleshooting

### Prevent session timeout

```javascript
// Run in browser console (F12) to keep session alive
function ClickConnect(){
    console.log("Keeping alive");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
```

### Check completion status

```python
import os
base = '/content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis-Classification/H-CoAtNet'
checkpoints = [
    'best_h_coatnet.pth',
    'best_efficientnet_pretrained.pth',
    'best_swin_pretrained.pth',
    'best_vit_pretrained.pth',
    'best_coatnet_baseline.pth',
    'best_gft_model.pth',
]
predictions = [f for f in os.listdir(base) if f.endswith('_y_pred.npy')]
print(f"Predictions found: {len(predictions)}")
for p in sorted(predictions):
    print(f"  OK  {p}")
for ckpt in checkpoints:
    path = os.path.join(base, ckpt)
    status = f"OK ({os.path.getsize(path)/1e6:.1f} MB)" if os.path.exists(path) else "MISSING"
    print(f"{ckpt}: {status}")
```

### OOM on ViT-B/16

Reduce `BATCH_SIZE` from 16 to 8 in the training script.

---

## Paper Writing Checklist (After All Runs Complete)

1. **Results Table** — Copy from `figures/comprehensive_results.csv`
2. **Cross-Validation** — Copy mean ± std from `crossval_summary.txt`
3. **Statistical Significance** — Copy p-values from `mcnemar_results.csv`
4. **Ablation Table** — Copy from `ablation_results.csv`
5. **Figures** — Use from `figures/` and `gradcam/` directories (all 300 DPI)
6. **Architecture Diagram** — Create manually showing: ConvNeXt Stem → Stages 1-2 → Haar DWT → WG-FDCA → ViT → Stages 3-4 → PA-DTS → SCTR → Classifier
