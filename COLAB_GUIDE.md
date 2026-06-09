# H-CoAtNet — Google Colab Execution Guide

## Setup (Run these cells first, in order)

---

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
git clone https://github.com/Cyrax321/H-CoAtNet-Ichthyosis-Classification.git
cd H-CoAtNet-Ichthyosis-Classification/H-CoAtNet
pip install -q -r requirements.txt
```

---

### Cell 2 — Set API Key (Do this BEFORE any training cell)

```python
import os
os.environ["ROBOFLOW_API_KEY"] = "gXuxxWEMFJ8nK73o7pN7"  # ← your key
# After setting this, rotate the key on Roboflow dashboard
```

---

### Cell 3 — Verify GPU

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
# You need a GPU (Runtime → Change runtime type → T4 GPU)
```

---

## Running Experiments (in recommended order)

---

### Train H-CoAtNet (Proposed Method)

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis/H-CoAtNet
python proposed_method/train_h_coatnet.py 2>&1 | tee logs_h_coatnet.txt
```

**Expected output files:**
- `best_h_coatnet.pth` — best checkpoint
- `h_coatnet_y_true.npy`, `h_coatnet_y_pred.npy` — predictions for stats tests
- `confusion_matrix_h_coatnet.png`
- `h_coatnet_loss_curves.png`, `h_coatnet_acc_curves.png`

---

### Train Pretrained Baselines (PRIMARY comparisons — run these, not the scratch versions)

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis/H-CoAtNet
python baselines/train_efficientnet_pretrained.py 2>&1 | tee logs_efficientnet_pretrained.txt
```

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis/H-CoAtNet
python baselines/train_swin_pretrained.py 2>&1 | tee logs_swin_pretrained.txt
```

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis/H-CoAtNet
python baselines/train_vit_pretrained.py 2>&1 | tee logs_vit_pretrained.txt
```

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis/H-CoAtNet
python baselines/train_coatnet.py 2>&1 | tee logs_coatnet.txt
```

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis/H-CoAtNet
python baselines/train_gft.py 2>&1 | tee logs_gft.txt
```

---

### Run Ablation Study (4 conditions — do this AFTER H-CoAtNet training)

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis/H-CoAtNet
python proposed_method/train_ablation.py --condition full 2>&1 | tee ablation_full.txt
python proposed_method/train_ablation.py --condition no_hse 2>&1 | tee ablation_no_hse.txt
python proposed_method/train_ablation.py --condition no_transformer 2>&1 | tee ablation_no_transformer.txt
python proposed_method/train_ablation.py --condition baseline 2>&1 | tee ablation_baseline.txt
```

**Output:** `ablation_results.csv` — paste this table directly into your paper.

---

### Run Cross-Validation + McNemar's Test (do AFTER all models are trained)

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis/H-CoAtNet
python proposed_method/evaluate_crossval.py 2>&1 | tee logs_crossval.txt
```

**Output files:**
- `crossval_summary.txt` — copy mean ± std numbers directly into your paper
- `mcnemar_results.csv` — p-values for all baseline comparisons
- `fold_1_cm.png` through `fold_5_cm.png` — per-fold confusion matrices

---

### Generate Grad-CAM Visualizations (do AFTER H-CoAtNet training)

```bash
%%bash
cd /content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis/H-CoAtNet
python proposed_method/generate_gradcam.py --checkpoint best_h_coatnet.pth
```

**Output:** `gradcam/gradcam_grid.png` — ready to use as a figure in your paper.

---

## Estimated Training Times (T4 GPU)

| Script | Estimated Time |
|---|---|
| `train_h_coatnet.py` | ~35–50 min |
| `train_efficientnet_pretrained.py` | ~20–30 min |
| `train_swin_pretrained.py` | ~25–35 min |
| `train_vit_pretrained.py` | ~30–45 min |
| `train_coatnet.py` | ~20–30 min |
| `train_gft.py` | ~25–35 min |
| `train_ablation.py` (all 4) | ~2–3 hours total |
| `evaluate_crossval.py` | ~3–4 hours (5 folds × 30 epochs) |
| `generate_gradcam.py` | ~2–5 min |

**Total: ~8–10 hours. Use Colab Pro or split across sessions (all checkpoints save to Drive).**

---

## Important Notes for Colab

### Prevent session timeout during long runs

```python
# Run this in a browser console (F12) to keep the session alive
function ClickConnect(){
    console.log("Keeping alive");
    document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
```

### Check if a run completed after a disconnect

```python
import os
# Check which checkpoints exist
checkpoints = [
    'best_h_coatnet.pth',
    'best_efficientnet_pretrained.pth',
    'best_swin_pretrained.pth',
    'best_vit_pretrained.pth',
    'best_coatnet_baseline.pth',
    'best_gft_model.pth',
]
base = '/content/drive/MyDrive/HCoAtNet_experiments/H-CoAtNet-Ichthyosis/H-CoAtNet'
for ckpt in checkpoints:
    path = os.path.join(base, ckpt)
    status = f"✅ {os.path.getsize(path)/1e6:.1f} MB" if os.path.exists(path) else "❌ missing"
    print(f"{ckpt}: {status}")
```

### If you get OOM (out-of-memory) on ViT-B/16

```python
# In train_vit_pretrained.py, change BATCH_SIZE from 16 to 8
# Or use gradient checkpointing (add to model):
import torch
model.gradient_checkpointing_enable()  # only works with HuggingFace models
# For timm: reduce batch size is the safest fix
```

---

## Order for Paper Writing After Running

1. Copy accuracy/F1 numbers from log files into Table 8
2. Copy mean ± std from `crossval_summary.txt` into the results section
3. Copy p-values from `mcnemar_results.csv` into a new statistical significance table
4. Copy ablation rows from `ablation_results.csv` into a new ablation table
5. Use `gradcam/gradcam_grid.png` as Figure 7 (interpretability)
6. Use `fold_{k}_cm.png` images for per-fold confusion matrices

---

## What Still Needs to Be Done in the Paper (Not Code)

These require rewriting in the paper itself:

| Section | What to Do |
|---|---|
| Abstract | Fix the accuracy number — use the single CV mean |
| Section 4.2 | Add parameter count + inference time per model (logged by scripts) |
| Related Work | Expand to 1.5 pages with subsections (CNN, hybrid, medical imaging, foundation models) |
| Section 3.3 | Add the hyperparameter table from the README |
| Section 4.3 | Add ablation table with 4 conditions |
| Section 4.4 | Add Grad-CAM figure + discussion of what regions the model attends to |
| Discussion | Add failure case analysis (why Lamellar Ichthyosis drops to 63.64% recall?) |
| Discussion | Add limitations section (single dataset, no prospective clinical validation) |
| Dataset section | Add clinical validation protocol (who validated, how many specialists, Cohen's kappa) |
| References | Replace Ref 41 with Dosovitskiy et al. 2021 ViT paper |
