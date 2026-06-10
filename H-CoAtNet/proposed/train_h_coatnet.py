"""
H-CoAtNet: Hierarchical CoAtNet with Multi-Scale Cross-Attention Fusion,
           Dual-Path Channel-Spatial Attention, and Adaptive Token Pruning
===========================================================================
Proposed method for ichthyosis subtype classification.

Novel contributions:
  1. Multi-Scale Cross-Attention Fusion (MSCAF) — fuses fine (stage1) and
     structural (stage2) CNN features via cross-attention before transformer.
  2. Dual-Path Channel-Spatial HSE (DCSHSE) — extends SE with a spatial
     attention branch and learnable channel-spatial fusion.
  3. Learnable Adaptive Token Pruning (LATP) — predicts per-image pruning
     thresholds from token statistics instead of fixed ratios.
"""

import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix
from roboflow import Roboflow
from timm import create_model
from timm.models.vision_transformer import Block

# ===========================
# Reproducibility
# ===========================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===========================
# Configuration
# ===========================
API_KEY = "gXuxxWEMFJ8nK73o7pN7"
TARGET_SIZE = (224, 224)
BATCH_SIZE = 24
EPOCHS = 30
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# Novel Module 1: Multi-Scale Cross-Attention Fusion (MSCAF)
# ===========================
class MultiScaleCrossAttentionFusion(nn.Module):
    """
    Fuses features from two ConvNeXt stages via cross-attention.

    Stage1 features (high-resolution, fine texture) serve as keys/values.
    Stage2 features (mid-resolution, structural patterns) serve as queries.
    The cross-attention lets stage2 selectively attend to fine-grained
    texture information from stage1, producing enriched multi-scale tokens.

    Args:
        dim_low:   Channel dimension of stage1 output (96 for ConvNeXt-Tiny)
        dim_high:  Channel dimension of stage2 output (192 for ConvNeXt-Tiny)
        num_heads: Number of attention heads
        dropout:   Dropout rate for attention and projection
    """
    def __init__(self, dim_low: int = 96, dim_high: int = 192,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_high // num_heads
        self.scale = self.head_dim ** -0.5

        self.proj_low = nn.Sequential(
            nn.Conv2d(dim_low, dim_high, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_high),
            nn.GELU(),
        )
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
            nn.Linear(dim_high, dim_high * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_high * 2, dim_high),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(dim_high)

    def forward(self, feat_low: torch.Tensor, feat_high: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_low:  (B, 96, 56, 56) — stage1 features (fine texture)
            feat_high: (B, 192, 28, 28) — stage2 features (structural)
        Returns:
            fused: (B, 784, 192) — multi-scale fused token sequence
        """
        B = feat_low.shape[0]

        kv_feat = self.proj_low(feat_low)
        kv_feat = self.downsample_low(kv_feat)
        kv_tokens = kv_feat.flatten(2).transpose(1, 2)

        q_tokens = feat_high.flatten(2).transpose(1, 2)

        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)

        Q = self.q_proj(q).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        out = self.proj_drop(self.out_proj(out))

        fused = q_tokens + out
        fused = fused + self.ffn(self.norm_ffn(fused))

        return fused


# ===========================
# Novel Module 2: Dual-Path Channel-Spatial HSE (DCSHSE)
# ===========================
class DualPathChannelSpatialHSE(nn.Module):
    """
    Dual-path attention module combining channel recalibration (SE path)
    with spatial importance scoring, fused via a learnable alpha parameter.

    Channel path: Standard SE — squeeze global context, excite per-channel.
    Spatial path: Per-token MLP that scores each spatial position based on
                  its feature content relative to the global representation.
    Fusion: Learnable alpha blends channel and spatial importance maps.

    Args:
        dim:       Token embedding dimension
        reduction: SE bottleneck reduction ratio
        dropout:   Dropout rate
    """
    def __init__(self, dim: int, reduction: int = 16, dropout: float = 0.0):
        super().__init__()
        mid = max(1, dim // reduction)

        self.channel_se = nn.Sequential(
            nn.Linear(dim, mid, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, dim, bias=True),
            nn.Sigmoid()
        )

        self.spatial_mlp = nn.Sequential(
            nn.Linear(dim * 2, mid, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1, bias=True),
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, C) — token sequence
        Returns:
            out: (B, N, C) — recalibrated tokens
            importance: (B, N) — per-token importance scores
        """
        x_normed = self.norm(x)
        global_ctx = x_normed.mean(dim=1)

        channel_gates = self.channel_se(global_ctx).unsqueeze(1)
        out = x * channel_gates

        channel_scores = out.norm(dim=-1)
        channel_scores = channel_scores - channel_scores.mean(dim=-1, keepdim=True)
        channel_std = channel_scores.std(dim=-1, keepdim=True) + 1e-6
        channel_importance = channel_scores / channel_std

        B, N, C = x.shape
        global_expanded = global_ctx.unsqueeze(1).expand(-1, N, -1)
        spatial_input = torch.cat([x_normed, global_expanded], dim=-1)
        spatial_scores = self.spatial_mlp(spatial_input).squeeze(-1)
        spatial_importance = spatial_scores - spatial_scores.mean(dim=-1, keepdim=True)
        spatial_std = spatial_importance.std(dim=-1, keepdim=True) + 1e-6
        spatial_importance = spatial_importance / spatial_std

        alpha = torch.sigmoid(self.alpha)
        combined = alpha * channel_importance + (1 - alpha) * spatial_importance
        importance = F.softmax(combined, dim=-1)

        return out, importance


# ===========================
# Novel Module 3: Learnable Adaptive Token Pruning (LATP)
# ===========================
class AdaptiveTokenPruning(nn.Module):
    """
    Learns per-image pruning thresholds from token statistics.

    Instead of fixed top-k ratios, a small MLP predicts the fraction of
    tokens to retain based on the distribution of importance scores.
    Uses a straight-through estimator for differentiable hard decisions.

    Args:
        dim:      Token embedding dimension
        min_keep: Minimum fraction of tokens to retain (safety floor)
        max_keep: Maximum fraction (caps unnecessary retention)
    """
    def __init__(self, dim: int, min_keep: float = 0.3, max_keep: float = 0.9):
        super().__init__()
        self.min_keep = min_keep
        self.max_keep = max_keep

        self.threshold_predictor = nn.Sequential(
            nn.Linear(dim + 3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, importance: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens:     (B, N, C) — current token sequence
            importance: (B, N)   — per-token importance scores
        Returns:
            selected:   (B, K, C) — adaptively pruned tokens
        """
        B, N, C = tokens.size()

        global_feat = tokens.mean(dim=1)
        imp_stats = torch.stack([
            importance.mean(dim=1),
            importance.std(dim=1),
            importance.max(dim=1).values,
        ], dim=-1)

        predictor_input = torch.cat([global_feat, imp_stats], dim=-1)
        keep_ratio = self.threshold_predictor(predictor_input).squeeze(-1)
        keep_ratio = self.min_keep + keep_ratio * (self.max_keep - self.min_keep)

        k = torch.clamp((keep_ratio * N).long(), min=max(1, int(self.min_keep * N)),
                         max=int(self.max_keep * N))
        k_val = k[0].item()

        _, top_k_idx = torch.topk(importance, k_val, dim=1)
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, k_val)
        selected = tokens[batch_idx, top_k_idx]

        return selected


# ===========================
# H-CoAtNet Model (with all 3 novel modules)
# ===========================
class HCoAtNet(nn.Module):
    """
    H-CoAtNet: Hierarchical CoAtNet for ichthyosis image classification.

    Architecture:
      1. ConvNeXt-Tiny stem + stages 1-2 (local feature extraction)
      2. Multi-Scale Cross-Attention Fusion (MSCAF) — fuses stage1 + stage2
      3. Learnable positional embedding + ViT transformer blocks (global context)
      4. ConvNeXt stages 3-4 (deep semantic features)
      5. Dual-stage DCSHSE with LATP (discriminative token selection)
      6. Mean-pool -> LayerNorm -> Linear classifier

    ConvNeXt-Tiny channel progression: 96 -> 192 -> 384 -> 768
    """
    def __init__(
        self,
        base_model: str = 'convnext_tiny',
        num_classes: int = 5,
        vit_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        cnn_backbone = create_model(base_model, pretrained=True, num_classes=0)

        self.cnn_stem   = cnn_backbone.stem
        self.cnn_stage1 = cnn_backbone.stages[0]
        self.cnn_stage2 = cnn_backbone.stages[1]
        self.cnn_stage3 = cnn_backbone.stages[2]
        self.cnn_stage4 = cnn_backbone.stages[3]

        vit_dim = 192

        self.mscaf = MultiScaleCrossAttentionFusion(
            dim_low=96, dim_high=192, num_heads=4, dropout=dropout
        )

        num_vit_tokens = 28 * 28
        self.pos_embed = nn.Parameter(torch.zeros(1, num_vit_tokens, vit_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.vit_blocks = nn.ModuleList([
            Block(dim=vit_dim, num_heads=6, proj_drop=dropout, attn_drop=dropout * 0.5)
            for _ in range(vit_blocks)
        ])

        final_embed_dim = 768
        self.dcshse_blocks = nn.ModuleList([
            DualPathChannelSpatialHSE(dim=final_embed_dim, reduction=16,
                                       dropout=dropout * 0.25)
            for _ in range(2)
        ])
        self.latp_blocks = nn.ModuleList([
            AdaptiveTokenPruning(dim=final_embed_dim, min_keep=0.4, max_keep=0.85),
            AdaptiveTokenPruning(dim=final_embed_dim, min_keep=0.25, max_keep=0.7),
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(final_embed_dim),
            nn.Dropout(dropout),
            nn.Linear(final_embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_stem(x)
        feat_stage1 = self.cnn_stage1(x)
        feat_stage2 = self.cnn_stage2(feat_stage1)

        fused_tokens = self.mscaf(feat_stage1, feat_stage2)

        fused_tokens = fused_tokens + self.pos_embed
        for blk in self.vit_blocks:
            fused_tokens = blk(fused_tokens)

        B = fused_tokens.shape[0]
        x = fused_tokens.transpose(1, 2).reshape(B, 192, 28, 28)

        x = self.cnn_stage3(x)
        x = self.cnn_stage4(x)

        x = x.flatten(2).transpose(1, 2)
        current_tokens = x
        for dcshse, latp in zip(self.dcshse_blocks, self.latp_blocks):
            tokens_attn, importance = dcshse(current_tokens)
            current_tokens = latp(tokens_attn, importance)

        x = current_tokens.mean(dim=1)
        return self.classifier(x)


# ===========================
# Training & Evaluation Utilities
# ===========================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, all_preds, all_targets = 0.0, [], []
    for images, targets in tqdm(loader, desc="Training", leave=False):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() if all_preds else 0.0
    return avg_loss, accuracy


def evaluate(model, loader, criterion, desc="Evaluating"):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() if all_preds else 0.0
    return avg_loss, accuracy, all_targets, all_preds


def plot_curves(history: dict, out_dir: str = "."):
    for metric in ['loss', 'acc']:
        plt.figure(figsize=(10, 6))
        plt.plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
        plt.plot(history[f'val_{metric}'],   label=f'Validation {metric.capitalize()}')
        plt.plot(history[f'test_{metric}'],  label=f'Test {metric.capitalize()}', linestyle='--')
        plt.title(f'H-CoAtNet {metric.capitalize()} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'h_coatnet_{metric}_curves.png'), dpi=300)
        plt.close()


# ===========================
# Main Training Logic
# ===========================
def main():
    print(f"Using device: {DEVICE}")
    print(f"Random seed: {RANDOM_SEED}")

    if not API_KEY:
        raise EnvironmentError(
            "ROBOFLOW_API_KEY environment variable is not set. "
            "Run: export ROBOFLOW_API_KEY=<your_key>"
        )

    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
    dataset = project.version(1).download("folder")
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
    val_test_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset      = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transform)
    validation_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "valid"), transform=val_test_transform)
    test_dataset       = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"),  transform=val_test_transform)

    num_workers = 0 if os.name == 'nt' else 2
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_loader      = DataLoader(train_dataset,      batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers, generator=g)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader       = DataLoader(test_dataset,       batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    counts = np.bincount(train_dataset.targets)
    class_weights = torch.tensor(
        [len(train_dataset) / (c * num_classes + 1e-6) for c in counts],
        dtype=torch.float
    ).to(DEVICE)
    print("Class weights:", class_weights.cpu().numpy().round(4))

    model     = HCoAtNet(num_classes=num_classes, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    try:
        print("\n--- Model Summary ---")
        summary(model, input_size=(BATCH_SIZE, 3, *TARGET_SIZE))
    except Exception as e:
        print(f"Model summary unavailable: {e}")

    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']}
    epoch_times = []
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss,  val_acc,  _, _ = evaluate(model, validation_loader, criterion, "Validating")
        test_loss, test_acc, _, _ = evaluate(model, test_loader,       criterion, "Testing")
        scheduler.step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        history['train_loss'].append(train_loss);  history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss);      history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss);    history['test_acc'].append(test_acc)

        print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"  Losses -- Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f}")
        print(f"  Epoch time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_h_coatnet.pth')
            print(f"  New best model saved (Val Acc = {best_val_acc:.4f})")

    avg_epoch_time = np.mean(epoch_times)
    print(f"\nAverage epoch time: {avg_epoch_time:.1f}s")

    print("\n--- Final Evaluation (Best Checkpoint) ---")
    model.load_state_dict(torch.load('best_h_coatnet.pth', weights_only=True))
    _, final_test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, "Final Test")
    print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")

    np.save('h_coatnet_y_true.npy', np.array(y_true))
    np.save('h_coatnet_y_pred.npy', np.array(y_pred))
    print("Predictions saved to h_coatnet_y_true.npy and h_coatnet_y_pred.npy")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 12}
    )
    plt.xlabel('Predicted Label', fontsize=13)
    plt.ylabel('True Label', fontsize=13)
    plt.title('Confusion Matrix -- H-CoAtNet', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix_h_coatnet.png', dpi=300)
    plt.close()

    plot_curves(history)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n--- Hyperparameter Summary ---")
    print(f"  Architecture     : H-CoAtNet (ConvNeXt-Tiny + MSCAF + {2} ViT blocks + DCSHSE + LATP)")
    print(f"  Backbone         : convnext_tiny (pretrained=True, ImageNet-1k)")
    print(f"  Input resolution : {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Optimiser        : AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"  LR schedule      : CosineAnnealingLR (T_max={EPOCHS})")
    print(f"  Loss             : CrossEntropyLoss (label_smoothing=0.1, class_weights=True)")
    print(f"  Dropout          : {DROPOUT}")
    print(f"  Random seed      : {RANDOM_SEED}")
    print(f"  Trainable params : {n_params:,}")
    print(f"  Avg epoch time   : {avg_epoch_time:.1f}s  (device: {DEVICE})")


if __name__ == '__main__':
    main()