"""
H-CoAtNet: Ablation Study
=========================
Trains four model conditions to isolate the contribution of each architectural component.
All conditions use identical hyperparameters, seed, and data split.

Usage:
    python proposed_method/train_ablation.py --condition full
    python proposed_method/train_ablation.py --condition no_hse
    python proposed_method/train_ablation.py --condition no_transformer
    python proposed_method/train_ablation.py --condition baseline

Outputs:
    ablation_results.csv          -- one row per condition
    ablation_{condition}_cm.png   -- confusion matrix at 300 DPI
"""

import os
import csv
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from timm import create_model
from timm.models.vision_transformer import Block

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

TARGET_SIZE  = (224, 224)
BATCH_SIZE   = 24
EPOCHS       = 30
LR           = 5e-5
WEIGHT_DECAY = 0.01
DROPOUT      = 0.2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV  = "ablation_results.csv"


# ── Model building blocks ────────────────────────────────────────────────────
class HierarchicalSE(nn.Module):
    """Dual-stage Hierarchical Squeeze-Excitation with token importance scoring."""
    def __init__(self, dim, reduction=16, dropout=0.0):
        super().__init__()
        mid = max(1, dim // reduction)
        self.se = nn.Sequential(
            nn.Linear(dim, mid, bias=True), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, dim, bias=True), nn.Sigmoid()
        )

    def forward(self, x):
        gates = self.se(x.mean(dim=1)).unsqueeze(1)
        out = x * gates
        token_scores = out.norm(dim=-1)
        token_scores = token_scores - token_scores.mean(dim=-1, keepdim=True)
        importance = F.softmax(token_scores / (token_scores.std(dim=-1, keepdim=True) + 1e-6), dim=-1)
        return out, importance


def select_patches(tokens, importance, k):
    B, N, C = tokens.size()
    k = min(k, N)
    _, top_k_idx = torch.topk(importance, k, dim=1)
    batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, k)
    return tokens[batch_idx, top_k_idx]


# ── Ablation model factory ───────────────────────────────────────────────────
def build_model(condition: str, num_classes: int) -> nn.Module:
    """
    Returns a model for the given ablation condition.

    condition:
      'full'           — H-CoAtNet (all components)
      'no_hse'         — ViT blocks only, HSE replaced with global avg pool
      'no_transformer' — ConvNeXt stages only, ViT blocks skipped
      'baseline'       — Plain ConvNeXt-Tiny fine-tuned (CoAtNet baseline)
    """
    assert condition in ('full', 'no_hse', 'no_transformer', 'baseline'), \
        f"Unknown condition: {condition}"

    class AblationModel(nn.Module):
        def __init__(self, condition, num_classes):
            super().__init__()
            self.condition = condition
            cnn = create_model('convnext_tiny', pretrained=True, num_classes=0)
            self.cnn_stem   = cnn.stem
            self.cnn_stage1 = cnn.stages[0]
            self.cnn_stage2 = cnn.stages[1]
            self.cnn_stage3 = cnn.stages[2]
            self.cnn_stage4 = cnn.stages[3]

            # ViT blocks (used in 'full' and 'no_hse')
            if condition in ('full', 'no_hse'):
                vit_dim = 192
                self.pos_embed = nn.Parameter(torch.zeros(1, 28 * 28, vit_dim))
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
                self.vit_blocks = nn.ModuleList([
                    Block(dim=vit_dim, num_heads=6, proj_drop=DROPOUT, attn_drop=DROPOUT * 0.5)
                    for _ in range(2)
                ])

            # HSE blocks (used in 'full' and 'no_transformer')
            if condition in ('full', 'no_transformer'):
                final_dim = 768
                self.selection_sizes = [int(49 * 0.75), int(49 * 0.50)]
                self.hse_blocks = nn.ModuleList([
                    HierarchicalSE(dim=final_dim, reduction=16, dropout=DROPOUT * 0.25)
                    for _ in self.selection_sizes
                ])

            final_dim = 768
            self.classifier = nn.Sequential(
                nn.LayerNorm(final_dim),
                nn.Dropout(DROPOUT),
                nn.Linear(final_dim, num_classes)
            )

            # For 'baseline': use timm's built-in head
            if condition == 'baseline':
                self.model = create_model('convnext_tiny', pretrained=True, num_classes=num_classes)

        def forward(self, x):
            if self.condition == 'baseline':
                return self.model(x)

            # Shared CNN stem + stage1 + stage2
            x = self.cnn_stem(x)
            x = self.cnn_stage1(x)     # (B, 96, 56, 56)
            x = self.cnn_stage2(x)     # (B, 192, 28, 28)

            # Optional transformer stage
            if self.condition in ('full', 'no_hse'):
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2) + self.pos_embed
                for blk in self.vit_blocks:
                    x = blk(x)
                x = x.transpose(1, 2).reshape(B, C, H, W)

            x = self.cnn_stage3(x)     # (B, 384, 14, 14)
            x = self.cnn_stage4(x)     # (B, 768, 7, 7)
            x = x.flatten(2).transpose(1, 2)  # (B, 49, 768)

            # Optional HSE stage
            if self.condition in ('full', 'no_transformer'):
                current = x
                for hse, k in zip(self.hse_blocks, self.selection_sizes):
                    tokens_attn, importance = hse(current)
                    current = select_patches(tokens_attn, importance, k)
                x = current.mean(dim=1)
            else:
                # no_hse: simple global average pooling
                x = x.mean(dim=1)

            return self.classifier(x)

    return AblationModel(condition, num_classes).to(DEVICE)


# ── Training & evaluation ────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, preds, targets = 0.0, [], []
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
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for imgs, tgts in tqdm(loader, desc="  eval ", leave=False):
        imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
        out = model(imgs)
        total_loss += criterion(out, tgts).item()
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(tgts.cpu().numpy())
    y_true = np.array(targets)
    y_pred = np.array(preds)
    acc    = accuracy_score(y_true, y_pred)
    return total_loss / len(loader), acc, y_true, y_pred


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="H-CoAtNet Ablation Study")
    parser.add_argument(
        '--condition',
        choices=['full', 'no_hse', 'no_transformer', 'baseline'],
        default='full',
        help="Ablation condition to train"
    )
    args = parser.parse_args()
    condition = args.condition

    condition_labels = {
        'full':           'H-CoAtNet (Full)',
        'no_hse':         'H-CoAtNet w/o HSE',
        'no_transformer': 'H-CoAtNet w/o Transformer',
        'baseline':       'CoAtNet Baseline (ConvNeXt-Tiny)',
    }
    print(f"=== Ablation Condition: {condition_labels[condition]} ===")
    print(f"Device: {DEVICE} | Seed: {RANDOM_SEED}")

    from roboflow import Roboflow
    rf = Roboflow(api_key="gXuxxWEMFJ8nK73o7pN7")
    dataset = rf.workspace("hi-l9ueo").project("ich-s-7lnsj").version(1).download("folder")
    DATASET_DIR = dataset.location

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR, "valid"), transform=val_transform)
    test_ds  = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"),  transform=val_transform)

    num_workers = 0 if os.name == 'nt' else 2
    g = torch.Generator(); g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    class_names = train_ds.classes
    num_classes = len(class_names)
    counts = np.bincount(train_ds.targets)
    cw = torch.tensor(
        [len(train_ds) / (c * num_classes + 1e-6) for c in counts], dtype=torch.float
    ).to(DEVICE)

    model     = build_model(condition, num_classes)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    ckpt_path = f"ablation_{condition}_best.pth"
    t_start = time.time()

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Train {tr_acc:.4f} | Val {vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), ckpt_path)

    total_time = time.time() - t_start

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    _, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)
    macro_f1 = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    wtd_f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Count trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n--- Ablation Results: {condition_labels[condition]} ---")
    print(f"  Test Accuracy  : {test_acc*100:.2f}%")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  Weighted F1    : {wtd_f1:.4f}")
    print(f"  Parameters     : {n_params:,}")
    print(f"  Training time  : {total_time:.1f}s")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Save predictions
    np.save(f'ablation_{condition}_y_true.npy', y_true)
    np.save(f'ablation_{condition}_y_pred.npy', y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 11})
    plt.title(f'Ablation: {condition_labels[condition]}', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12); plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'ablation_{condition}_cm.png', dpi=300)
    plt.close()

    # Append row to ablation_results.csv
    row = {
        'condition': condition_labels[condition],
        'test_accuracy': round(test_acc * 100, 2),
        'macro_f1': round(macro_f1, 4),
        'weighted_f1': round(wtd_f1, 4),
        'n_params': n_params,
        'train_time_s': round(total_time, 1),
    }
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nResults appended to {RESULTS_CSV}")


if __name__ == '__main__':
    main()