"""
H-CoAtNet: Ablation Study
=========================
Trains six model conditions to isolate the contribution of each novel module.
All conditions use identical hyperparameters, seed, and data split.

Conditions:
  full             -- H-CoAtNet (all novel components: MSCAF + ViT + DCSHSE + LATP)
  no_mscaf         -- Single-scale input to ViT (removes cross-attention fusion)
  no_transformer   -- CNN + DCSHSE only (removes ViT blocks and MSCAF)
  no_dcshse        -- MSCAF + ViT, but uses global avg pool instead of DCSHSE+LATP
  fixed_pruning    -- MSCAF + ViT + old HSE with fixed 75/50% pruning ratios
  baseline         -- Plain ConvNeXt-Tiny fine-tuned

Usage:
    python evaluation/ablation.py --condition full
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


# ── Novel modules (mirrored from train_h_coatnet.py) ────────────────────────

class MultiScaleCrossAttentionFusion(nn.Module):
    def __init__(self, dim_low=96, dim_high=192, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_high // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_low = nn.Sequential(
            nn.Conv2d(dim_low, dim_high, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_high), nn.GELU())
        self.downsample_low = nn.AdaptiveAvgPool2d(28)
        self.q_proj = nn.Linear(dim_high, dim_high, bias=False)
        self.k_proj = nn.Linear(dim_high, dim_high, bias=False)
        self.v_proj = nn.Linear(dim_high, dim_high, bias=False)
        self.out_proj = nn.Linear(dim_high, dim_high)
        self.attn_drop = nn.Dropout(dropout * 0.5)
        self.proj_drop = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim_high)
        self.norm_kv = nn.LayerNorm(dim_high)
        self.ffn = nn.Sequential(
            nn.Linear(dim_high, dim_high * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_high * 2, dim_high), nn.Dropout(dropout))
        self.norm_ffn = nn.LayerNorm(dim_high)

    def forward(self, feat_low, feat_high):
        B = feat_low.shape[0]
        kv_feat = self.downsample_low(self.proj_low(feat_low))
        kv_tokens = kv_feat.flatten(2).transpose(1, 2)
        q_tokens = feat_high.flatten(2).transpose(1, 2)
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)
        Q = self.q_proj(q).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        out = self.proj_drop(self.out_proj(out))
        fused = q_tokens + out
        fused = fused + self.ffn(self.norm_ffn(fused))
        return fused


class DualPathChannelSpatialHSE(nn.Module):
    def __init__(self, dim, reduction=16, dropout=0.0):
        super().__init__()
        mid = max(1, dim // reduction)
        self.channel_se = nn.Sequential(
            nn.Linear(dim, mid, bias=True), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, dim, bias=True), nn.Sigmoid())
        self.spatial_mlp = nn.Sequential(
            nn.Linear(dim * 2, mid, bias=True), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, 1, bias=True))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_normed = self.norm(x)
        global_ctx = x_normed.mean(dim=1)
        channel_gates = self.channel_se(global_ctx).unsqueeze(1)
        out = x * channel_gates
        ch_scores = out.norm(dim=-1)
        ch_scores = (ch_scores - ch_scores.mean(dim=-1, keepdim=True)) / (ch_scores.std(dim=-1, keepdim=True) + 1e-6)
        B, N, C = x.shape
        sp_input = torch.cat([x_normed, global_ctx.unsqueeze(1).expand(-1, N, -1)], dim=-1)
        sp_scores = self.spatial_mlp(sp_input).squeeze(-1)
        sp_scores = (sp_scores - sp_scores.mean(dim=-1, keepdim=True)) / (sp_scores.std(dim=-1, keepdim=True) + 1e-6)
        alpha = torch.sigmoid(self.alpha)
        importance = F.softmax(alpha * ch_scores + (1 - alpha) * sp_scores, dim=-1)
        return out, importance


class AdaptiveTokenPruning(nn.Module):
    def __init__(self, dim, min_keep=0.3, max_keep=0.9):
        super().__init__()
        self.min_keep = min_keep
        self.max_keep = max_keep
        self.threshold_predictor = nn.Sequential(
            nn.Linear(dim + 3, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, tokens, importance):
        B, N, C = tokens.size()
        global_feat = tokens.mean(dim=1)
        imp_stats = torch.stack([importance.mean(dim=1), importance.std(dim=1),
                                  importance.max(dim=1).values], dim=-1)
        keep_ratio = self.threshold_predictor(torch.cat([global_feat, imp_stats], dim=-1)).squeeze(-1)
        keep_ratio = self.min_keep + keep_ratio * (self.max_keep - self.min_keep)
        k_val = torch.clamp((keep_ratio * N).long(), min=max(1, int(self.min_keep * N)),
                             max=int(self.max_keep * N))[0].item()
        _, top_k_idx = torch.topk(importance, k_val, dim=1)
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, k_val)
        return tokens[batch_idx, top_k_idx]


class HierarchicalSE(nn.Module):
    """Old-style fixed-ratio SE for the 'fixed_pruning' ablation condition."""
    def __init__(self, dim, reduction=16, dropout=0.0):
        super().__init__()
        mid = max(1, dim // reduction)
        self.se = nn.Sequential(
            nn.Linear(dim, mid, bias=True), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, dim, bias=True), nn.Sigmoid())

    def forward(self, x):
        gates = self.se(x.mean(dim=1)).unsqueeze(1)
        out = x * gates
        scores = out.norm(dim=-1)
        scores = (scores - scores.mean(dim=-1, keepdim=True)) / (scores.std(dim=-1, keepdim=True) + 1e-6)
        importance = F.softmax(scores, dim=-1)
        return out, importance


def select_patches_fixed(tokens, importance, k):
    B, N, C = tokens.size()
    k = min(k, N)
    _, top_k_idx = torch.topk(importance, k, dim=1)
    batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, k)
    return tokens[batch_idx, top_k_idx]


# ── Ablation model factory ───────────────────────────────────────────────────
def build_model(condition: str, num_classes: int) -> nn.Module:
    """
    Returns a model for the given ablation condition.
    """
    VALID = ('full', 'no_mscaf', 'no_transformer', 'no_dcshse', 'fixed_pruning', 'baseline')
    assert condition in VALID, f"Unknown condition: {condition}. Choose from {VALID}"

    class AblationModel(nn.Module):
        def __init__(self, condition, num_classes):
            super().__init__()
            self.condition = condition

            if condition == 'baseline':
                self.model = create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
                return

            cnn = create_model('convnext_tiny', pretrained=True, num_classes=0)
            self.cnn_stem   = cnn.stem
            self.cnn_stage1 = cnn.stages[0]
            self.cnn_stage2 = cnn.stages[1]
            self.cnn_stage3 = cnn.stages[2]
            self.cnn_stage4 = cnn.stages[3]

            has_mscaf = condition in ('full', 'no_dcshse', 'fixed_pruning')
            has_vit = condition != 'no_transformer'
            has_dcshse = condition in ('full', 'no_mscaf', 'no_transformer')
            has_fixed_hse = condition == 'fixed_pruning'

            if has_mscaf:
                self.mscaf = MultiScaleCrossAttentionFusion(
                    dim_low=96, dim_high=192, num_heads=4, dropout=DROPOUT)

            if has_vit:
                vit_dim = 192
                self.pos_embed = nn.Parameter(torch.zeros(1, 28 * 28, vit_dim))
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
                self.vit_blocks = nn.ModuleList([
                    Block(dim=vit_dim, num_heads=6, proj_drop=DROPOUT, attn_drop=DROPOUT * 0.5)
                    for _ in range(2)])

            final_dim = 768
            if has_dcshse:
                self.dcshse_blocks = nn.ModuleList([
                    DualPathChannelSpatialHSE(dim=final_dim, reduction=16, dropout=DROPOUT * 0.25)
                    for _ in range(2)])
                self.latp_blocks = nn.ModuleList([
                    AdaptiveTokenPruning(dim=final_dim, min_keep=0.4, max_keep=0.85),
                    AdaptiveTokenPruning(dim=final_dim, min_keep=0.25, max_keep=0.7)])

            if has_fixed_hse:
                self.selection_sizes = [int(49 * 0.75), int(49 * 0.50)]
                self.hse_blocks = nn.ModuleList([
                    HierarchicalSE(dim=final_dim, reduction=16, dropout=DROPOUT * 0.25)
                    for _ in self.selection_sizes])

            self.classifier = nn.Sequential(
                nn.LayerNorm(final_dim), nn.Dropout(DROPOUT), nn.Linear(final_dim, num_classes))

        def forward(self, x):
            if self.condition == 'baseline':
                return self.model(x)

            has_mscaf = self.condition in ('full', 'no_dcshse', 'fixed_pruning')
            has_vit = self.condition != 'no_transformer'
            has_dcshse = self.condition in ('full', 'no_mscaf', 'no_transformer')
            has_fixed_hse = self.condition == 'fixed_pruning'

            x = self.cnn_stem(x)
            feat_s1 = self.cnn_stage1(x)
            feat_s2 = self.cnn_stage2(feat_s1)

            if has_mscaf and has_vit:
                tokens = self.mscaf(feat_s1, feat_s2)
                tokens = tokens + self.pos_embed
                for blk in self.vit_blocks:
                    tokens = blk(tokens)
                B = tokens.shape[0]
                x = tokens.transpose(1, 2).reshape(B, 192, 28, 28)
            elif has_vit:
                B, C, H, W = feat_s2.shape
                tokens = feat_s2.flatten(2).transpose(1, 2)
                tokens = tokens + self.pos_embed
                for blk in self.vit_blocks:
                    tokens = blk(tokens)
                x = tokens.transpose(1, 2).reshape(B, C, H, W)
            else:
                x = feat_s2

            x = self.cnn_stage3(x)
            x = self.cnn_stage4(x)
            x = x.flatten(2).transpose(1, 2)

            if has_dcshse:
                current = x
                for dcshse, latp in zip(self.dcshse_blocks, self.latp_blocks):
                    tokens_attn, importance = dcshse(current)
                    current = latp(tokens_attn, importance)
                x = current.mean(dim=1)
            elif has_fixed_hse:
                current = x
                for hse, k in zip(self.hse_blocks, self.selection_sizes):
                    tokens_attn, importance = hse(current)
                    current = select_patches_fixed(tokens_attn, importance, k)
                x = current.mean(dim=1)
            else:
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
    return total_loss / len(loader), accuracy_score(y_true, y_pred), y_true, y_pred


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="H-CoAtNet Ablation Study")
    parser.add_argument(
        '--condition',
        choices=['full', 'no_mscaf', 'no_transformer', 'no_dcshse', 'fixed_pruning', 'baseline'],
        default='full',
    )
    args = parser.parse_args()
    condition = args.condition

    labels = {
        'full':           'H-CoAtNet (Full)',
        'no_mscaf':       'H-CoAtNet w/o MSCAF',
        'no_transformer': 'H-CoAtNet w/o Transformer',
        'no_dcshse':      'H-CoAtNet w/o DCSHSE (GAP)',
        'fixed_pruning':  'H-CoAtNet w/ Fixed Pruning',
        'baseline':       'ConvNeXt-Tiny Baseline',
    }
    print(f"=== Ablation: {labels[condition]} ===")
    print(f"Device: {DEVICE} | Seed: {RANDOM_SEED}")

    from roboflow import Roboflow
    rf = Roboflow(api_key="gXuxxWEMFJ8nK73o7pN7")
    dataset = rf.workspace("hi-l9ueo").project("ich-s-7lnsj").version(1).download("folder")
    DATASET_DIR = dataset.location

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.TrivialAugmentWide(), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))])
    val_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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
        [len(train_ds) / (c * num_classes + 1e-6) for c in counts], dtype=torch.float).to(DEVICE)

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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n--- Ablation Results: {labels[condition]} ---")
    print(f"  Test Accuracy  : {test_acc*100:.2f}%")
    print(f"  Macro F1       : {macro_f1:.4f}")
    print(f"  Weighted F1    : {wtd_f1:.4f}")
    print(f"  Parameters     : {n_params:,}")
    print(f"  Training time  : {total_time:.1f}s")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    np.save(f'ablation_{condition}_y_true.npy', y_true)
    np.save(f'ablation_{condition}_y_pred.npy', y_pred)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 11})
    plt.title(f'Ablation: {labels[condition]}', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12); plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'ablation_{condition}_cm.png', dpi=300)
    plt.close()

    row = {
        'condition': labels[condition],
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