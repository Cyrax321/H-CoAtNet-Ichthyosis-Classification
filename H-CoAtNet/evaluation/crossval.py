"""
H-CoAtNet: 5-Fold Stratified Cross-Validation + McNemar's Test
===============================================================
Runs stratified k-fold cross-validation on H-CoAtNet and computes
McNemar's test against all baselines whose prediction .npy files are present.

Usage:
    python proposed_method/evaluate_crossval.py

Outputs:
    crossval_results.csv     -- per-fold metrics
    crossval_summary.txt     -- mean +/- std
    mcnemar_results.csv      -- chi-squared and p-values vs baselines
    fold_{k}_cm.png          -- confusion matrix per fold (300 DPI)
"""

import os
import csv
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from timm import create_model
from timm.models.vision_transformer import Block

# ── Local import of the fixed model ─────────────────────────────────────────
# We replicate HierarchicalSE and HCoAtNet here so this script is self-contained
# (avoids import side-effects from the training script's main() call).

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

TARGET_SIZE = (224, 224)
BATCH_SIZE  = 24
EPOCHS      = 30
LR          = 5e-5
WEIGHT_DECAY = 0.01
DROPOUT     = 0.2
N_FOLDS     = 5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model definitions (self-contained copy) ─────────────────────────────────
class HierarchicalSE(nn.Module):
    def __init__(self, dim, reduction=16, dropout=0.0):
        super().__init__()
        mid = max(1, dim // reduction)
        self.se = nn.Sequential(
            nn.Linear(dim, mid, bias=True), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, dim, bias=True), nn.Sigmoid()
        )

    def forward(self, x):
        s = x.mean(dim=1)
        gates = self.se(s).unsqueeze(1)
        out = x * gates
        token_scores = out.norm(dim=-1)
        token_scores = token_scores - token_scores.mean(dim=-1, keepdim=True)
        token_std = token_scores.std(dim=-1, keepdim=True) + 1e-6
        importance = F.softmax(token_scores / token_std, dim=-1)
        return out, importance


class HCoAtNet(nn.Module):
    def __init__(self, num_classes=5, vit_blocks=2, dropout=0.2):
        super().__init__()
        cnn = create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.cnn_stem   = cnn.stem
        self.cnn_stage1 = cnn.stages[0]
        self.cnn_stage2 = cnn.stages[1]
        self.cnn_stage3 = cnn.stages[2]
        self.cnn_stage4 = cnn.stages[3]

        vit_dim = 192
        self.pos_embed = nn.Parameter(torch.zeros(1, 28 * 28, vit_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.vit_blocks = nn.ModuleList([
            Block(dim=vit_dim, num_heads=6, proj_drop=dropout, attn_drop=dropout * 0.5)
            for _ in range(vit_blocks)
        ])

        final_dim = 768
        self.selection_sizes = [int(49 * 0.75), int(49 * 0.50)]
        self.hierarchical_blocks = nn.ModuleList([
            HierarchicalSE(dim=final_dim, reduction=16, dropout=dropout * 0.25)
            for _ in self.selection_sizes
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(final_dim), nn.Dropout(dropout), nn.Linear(final_dim, num_classes)
        )

    def select_patches(self, tokens, importance, k):
        B, N, C = tokens.size()
        k = min(k, N)
        _, top_k_idx = torch.topk(importance, k, dim=1)
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, k)
        return tokens[batch_idx, top_k_idx]

    def forward(self, x):
        x = self.cnn_stem(x)
        x = self.cnn_stage1(x)
        x = self.cnn_stage2(x)
    for imgs, tgts in tqdm(loader, desc="  train", leave=False):
        imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, tgts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(tgts.cpu().numpy())
    return total_loss / len(loader), accuracy_score(targets, preds)


@torch.no_grad()
def eval_loader(model, loader, criterion):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for imgs, tgts in tqdm(loader, desc="  eval ", leave=False):
        imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
        out = model(imgs)
        total_loss += criterion(out, tgts).item()
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(tgts.cpu().numpy())
    return total_loss / len(loader), np.array(targets), np.array(preds)


# ── McNemar's test ───────────────────────────────────────────────────────────
def mcnemar_test(y_true, pred_a, pred_b):
    """
    McNemar's test with continuity correction.
    Returns chi2 statistic and p-value.
    H0: both classifiers have the same error rate.
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    # b: A correct, B wrong  |  c: A wrong, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    if b + c == 0:
        return 0.0, 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value   = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value


# ── Bootstrap confidence interval ───────────────────────────────────────────
def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000, alpha=0.05):
    n = len(y_true)
    rng = np.random.default_rng(RANDOM_SEED)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = np.percentile(boot_scores, 100 * alpha / 2)
    hi = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    return lo, hi


# ── Main cross-validation loop ───────────────────────────────────────────────
def main():
    from roboflow import Roboflow
    rf = Roboflow(api_key="gXuxxWEMFJ8nK73o7pN7")
    dataset = rf.workspace("hi-l9ueo").project("ich-s-7lnsj").version(1).download("folder")
    DATASET_DIR = dataset.location

    # Base transforms (no augmentation for evaluation consistency)
    val_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])

    # Combine train + val + test into one pool for k-fold splitting
    full_train = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=val_transform)
    full_val   = datasets.ImageFolder(os.path.join(DATASET_DIR, "valid"), transform=val_transform)
    full_test  = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"),  transform=val_transform)

    all_targets = (
        list(full_train.targets) +
        list(full_val.targets)   +
        list(full_test.targets)
    )
    n_train = len(full_train)
    n_val   = len(full_val)

    class_names = full_train.classes
    num_classes = len(class_names)
    print(f"Total samples: {len(all_targets)} | Classes: {class_names}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    all_indices = list(range(len(all_targets)))

    fold_results = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_indices, all_targets)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold + 1}/{N_FOLDS}")
        print(f"{'='*60}")
        print(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")

        # Build fold datasets by mapping global indices back to per-dataset local indices
        def make_fold_dataset(global_indices, aug=False):
            """Map global pooled indices → (dataset, local_index) pairs."""
            train_part, val_part, test_part = [], [], []
            for gi in global_indices:
                if gi < n_train:
                    train_part.append(gi)
                elif gi < n_train + n_val:
                    val_part.append(gi - n_train)
                else:
                    test_part.append(gi - n_train - n_val)

            subsets = []
            if train_part:
                ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"),
                                          transform=train_aug if aug else val_transform)
                subsets.append(Subset(ds, train_part))
            if val_part:
                subsets.append(Subset(full_val, val_part))
            if test_part:
                subsets.append(Subset(full_test, test_part))

            from torch.utils.data import ConcatDataset
            return ConcatDataset(subsets)

        fold_train_ds = make_fold_dataset(train_idx, aug=True)
        fold_test_ds  = make_fold_dataset(test_idx,  aug=False)

        num_workers = 0 if os.name == 'nt' else 2
        fold_train_loader = DataLoader(fold_train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers)
        fold_test_loader  = DataLoader(fold_test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

        # Class weights for this fold's training set
        fold_labels = [all_targets[i] for i in train_idx]
        counts = np.bincount(fold_labels, minlength=num_classes)
        cw = torch.tensor(
            [len(fold_labels) / (c * num_classes + 1e-6) for c in counts], dtype=torch.float
        ).to(DEVICE)

        model     = HCoAtNet(num_classes=num_classes, dropout=DROPOUT).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(EPOCHS):
            tr_loss, tr_acc = train_one_epoch(model, fold_train_loader, criterion, optimizer)
            scheduler.step()
            if epoch % 5 == 0 or epoch == EPOCHS - 1:
                print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

            # Use train loss as proxy for best checkpoint (no separate val in CV)
            if tr_loss < best_val_loss:
                best_val_loss = tr_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate on fold test set
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        _, y_true_fold, y_pred_fold = eval_loader(model, fold_test_loader, criterion)

        acc       = accuracy_score(y_true_fold, y_pred_fold)
        macro_f1  = f1_score(y_true_fold, y_pred_fold, average='macro',    zero_division=0)
        wtd_f1    = f1_score(y_true_fold, y_pred_fold, average='weighted', zero_division=0)

        acc_lo, acc_hi = bootstrap_ci(y_true_fold, y_pred_fold, accuracy_score)

        print(f"\n  Fold {fold+1} Results:")
        print(f"    Accuracy : {acc*100:.2f}% (95% CI: {acc_lo*100:.2f}%–{acc_hi*100:.2f}%)")
        print(f"    Macro F1 : {macro_f1:.4f}")
        print(f"    Wtd F1   : {wtd_f1:.4f}")
        print(classification_report(y_true_fold, y_pred_fold, target_names=class_names, digits=4))

        fold_results.append({
            'fold': fold + 1,
            'accuracy': acc, 'acc_ci_lo': acc_lo, 'acc_ci_hi': acc_hi,
            'macro_f1': macro_f1, 'weighted_f1': wtd_f1,
        })
        all_y_true.extend(y_true_fold.tolist())
        all_y_pred.extend(y_pred_fold.tolist())

        # Confusion matrix per fold
        cm = confusion_matrix(y_true_fold, y_pred_fold)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 11})
        plt.xlabel('Predicted', fontsize=12); plt.ylabel('True', fontsize=12)
        plt.title(f'H-CoAtNet — Fold {fold+1} Confusion Matrix', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'fold_{fold+1}_cm.png', dpi=300)
        plt.close()

        # Save per-fold predictions
        np.save(f'fold_{fold+1}_y_true.npy', y_true_fold)
        np.save(f'fold_{fold+1}_y_pred.npy', y_pred_fold)

        # Free GPU memory between folds
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Summary ─────────────────────────────────────────────────────────────
    accs  = [r['accuracy']    for r in fold_results]
    mf1s  = [r['macro_f1']    for r in fold_results]
    wf1s  = [r['weighted_f1'] for r in fold_results]

    summary_lines = [
        "=" * 60,
        f"5-Fold Cross-Validation Summary — H-CoAtNet",
        "=" * 60,
        f"Accuracy   : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%",
        f"Macro F1   : {np.mean(mf1s):.4f} ± {np.std(mf1s):.4f}",
        f"Weighted F1: {np.mean(wf1s):.4f} ± {np.std(wf1s):.4f}",
        "",
        "Per-fold breakdown:",
    ]
    for r in fold_results:
        summary_lines.append(
            f"  Fold {r['fold']}: Acc={r['accuracy']*100:.2f}%  MacroF1={r['macro_f1']:.4f}  WtdF1={r['weighted_f1']:.4f}"
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open('crossval_summary.txt', 'w') as f:
        f.write(summary_text + "\n")

    # Save CSV
    keys = ['fold', 'accuracy', 'acc_ci_lo', 'acc_ci_hi', 'macro_f1', 'weighted_f1']
    with open('crossval_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(fold_results)
    print("crossval_results.csv written.")

    # ── McNemar's test vs baselines (if prediction files exist) ─────────────
    print("\n--- McNemar's Test vs Baselines ---")
    all_y_true_np = np.array(all_y_true)
    all_y_pred_np = np.array(all_y_pred)

    baseline_files = {
        'EfficientNet-B0 (scratch)':  'efficientnet_scratch_y_pred.npy',
        'EfficientNet-B0 (pretrained)':'efficientnet_pretrained_y_pred.npy',
        'Swin-T (pretrained)':         'swin_pretrained_y_pred.npy',
        'ViT-B/16 (pretrained)':       'vit_pretrained_y_pred.npy',
        'CoAtNet':                     'coatnet_y_pred.npy',
        'GFT':                         'gft_y_pred.npy',
        'CNN':                         'cnn_y_pred.npy',
    }

    mcnemar_rows = []
    for baseline_name, pred_file in baseline_files.items():
        if os.path.exists(pred_file):
            baseline_pred = np.load(pred_file)
            min_len = min(len(all_y_true_np), len(baseline_pred))
            chi2_stat, p_val = mcnemar_test(
                all_y_true_np[:min_len], all_y_pred_np[:min_len], baseline_pred[:min_len]
            )
            sig = "✓ significant (p<0.05)" if p_val < 0.05 else "✗ not significant"
            print(f"  vs {baseline_name}: χ²={chi2_stat:.3f}, p={p_val:.4f}  {sig}")
            mcnemar_rows.append({
                'baseline': baseline_name, 'chi2': chi2_stat, 'p_value': p_val, 'significant': p_val < 0.05
            })
        else:
            print(f"  vs {baseline_name}: SKIPPED (file {pred_file} not found)")

    if mcnemar_rows:
        with open('mcnemar_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['baseline', 'chi2', 'p_value', 'significant'])
            writer.writeheader()
            writer.writerows(mcnemar_rows)
        print("mcnemar_results.csv written.")


if __name__ == '__main__':
    main()