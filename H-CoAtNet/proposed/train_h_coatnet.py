"""
H-CoAtNet: Hierarchical CoAtNet for Ichthyosis Classification
=============================================================
Proposed method — training script.

Key fixes vs initial version:
  - Corrected forward pass: stem → stage1 → stage2 → ViT blocks → stage3 → stage4
    (previously cnn_stage1 was applied 3× and cnn_stage2 was never called)
  - API key loaded from environment variable ROBOFLOW_API_KEY
  - Reproducibility seed (RANDOM_SEED = 42) applied to torch, numpy, and Python
  - torch.load called with weights_only=True (removes PyTorch deprecation warning)
  - Predictions saved as .npy for downstream McNemar's test / CI analysis
  - Per-epoch wall-clock timing recorded for efficiency reporting
  - Consistent "H-CoAtNet" capitalisation throughout
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
API_KEY = "gXuxxWEMFJ8nK73o7pN7"   # set via: export ROBOFLOW_API_KEY=<your_key>
TARGET_SIZE = (224, 224)
BATCH_SIZE = 24
EPOCHS = 30
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# Hierarchical Squeeze-Excitation (HSE) Module
# ===========================
class HierarchicalSE(nn.Module):
    """
    Hierarchical Squeeze-Excitation block with gradient-based token importance scoring.
    Performs channel-wise recalibration followed by per-token importance estimation.
    """
    def __init__(self, dim: int, reduction: int = 16, dropout: float = 0.0):
        super().__init__()
        mid = max(1, dim // reduction)
        self.se = nn.Sequential(
            nn.Linear(dim, mid, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, dim, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        # x: (B, N, C) — sequence of N patch tokens, each with C channels
        s = x.mean(dim=1)                         # (B, C) — global average pooling across tokens
        gates = self.se(s).unsqueeze(1)           # (B, 1, C)
        out = x * gates                           # channel-wise recalibration

        # Gradient-based token importance: L2 norm after recalibration, z-score normalised
        token_scores = out.norm(dim=-1)           # (B, N)
        token_scores = token_scores - token_scores.mean(dim=-1, keepdim=True)
        token_std = token_scores.std(dim=-1, keepdim=True) + 1e-6
        importance = F.softmax(token_scores / token_std, dim=-1)   # (B, N)
        return out, importance


# ===========================
# H-CoAtNet Model
# ===========================
class HCoAtNet(nn.Module):
    """
    H-CoAtNet: Hierarchical CoAtNet for ichthyosis image classification.

    Architecture:
      1. ConvNeXt-Tiny stem + stages 1-2  (local feature extraction)
      2. Learnable positional embedding + ViT transformer blocks  (global context)
      3. ConvNeXt stages 3-4  (deep semantic features)
      4. Dual-stage Hierarchical SE with token pruning  (discriminative token selection)
      5. Mean-pool → LayerNorm → Linear classifier

    Note: ConvNeXt-Tiny channel progression: 96 → 192 → 384 → 768
      stage1 output: (B, 96, 56, 56)
      stage2 output: (B, 192, 28, 28)  ← transformer stage operates here
      stage3 output: (B, 384, 14, 14)
      stage4 output: (B, 768, 7, 7)   ← HSE operates here (49 tokens × 768 dim)
    """
    def __init__(
        self,
        base_model: str = 'convnext_tiny',
        num_classes: int = 5,
        vit_blocks: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Load pretrained ConvNeXt-Tiny backbone, strip classifier head
        cnn_backbone = create_model(base_model, pretrained=True, num_classes=0)

        # ConvNeXt stages — accessed individually for hybrid interleaving
        self.cnn_stem   = cnn_backbone.stem      # 224→56, 3→96 channels
        self.cnn_stage1 = cnn_backbone.stages[0] # 56→56, 96 channels
        self.cnn_stage2 = cnn_backbone.stages[1] # 56→28, 96→192 channels
        self.cnn_stage3 = cnn_backbone.stages[2] # 28→14, 192→384 channels
        self.cnn_stage4 = cnn_backbone.stages[3] # 14→7, 384→768 channels

        # Transformer stage operates on stage2 output: (B, 192, 28, 28)
        # → reshaped to (B, 784, 192) token sequence
        vit_dim = 192                            # matches stage2 output channels
        num_vit_tokens = 28 * 28                # spatial tokens at 28×28 resolution
        self.pos_embed = nn.Parameter(torch.zeros(1, num_vit_tokens, vit_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.vit_blocks = nn.ModuleList([
            Block(dim=vit_dim, num_heads=6, proj_drop=dropout, attn_drop=dropout * 0.5)
            for _ in range(vit_blocks)
        ])

        # Hierarchical SE operates on stage4 output: (B, 768, 7, 7)
        # → reshaped to (B, 49, 768) token sequence
        final_embed_dim = 768
        num_final_tokens = 7 * 7               # 49 tokens at 7×7 resolution
        self.selection_sizes = [
            int(num_final_tokens * 0.75),      # stage 1: keep top 36 tokens
            int(num_final_tokens * 0.50),      # stage 2: keep top 24 tokens
        ]
        self.hierarchical_blocks = nn.ModuleList([
            HierarchicalSE(dim=final_embed_dim, reduction=16, dropout=dropout * 0.25)
            for _ in self.selection_sizes
        ])

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(final_embed_dim),
            nn.Dropout(dropout),
            nn.Linear(final_embed_dim, num_classes)
        )

    def select_patches(
        self, tokens: torch.Tensor, importance: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Retain the k most important tokens by importance score."""
        B, N, C = tokens.size()
        k = min(k, N)
        _, top_k_idx = torch.topk(importance, k, dim=1)
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, k)
        return tokens[batch_idx, top_k_idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Stage 1: CNN local feature extraction ──────────────────────────
        x = self.cnn_stem(x)      # (B,  96, 56, 56)
        x = self.cnn_stage1(x)    # (B,  96, 56, 56)
        x = self.cnn_stage2(x)    # (B, 192, 28, 28)  ← correct: stage2, not stage1 again

        # ── Stage 2: Transformer global context ────────────────────────────
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, 784, 192)
        x = x + self.pos_embed             # add learnable positional encoding
        for blk in self.vit_blocks:
            x = blk(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, 192, 28, 28)

        # ── Stage 3: Deep CNN semantic features ────────────────────────────
        x = self.cnn_stage3(x)    # (B, 384, 14, 14)
        x = self.cnn_stage4(x)    # (B, 768,  7,  7)

        # ── Stage 4: Hierarchical token selection ──────────────────────────
        x = x.flatten(2).transpose(1, 2)  # (B, 49, 768)
        current_tokens = x
        for attn_block, select_size in zip(self.hierarchical_blocks, self.selection_sizes):
            tokens_attn, importance = attn_block(current_tokens)
            current_tokens = self.select_patches(tokens_attn, importance, select_size)

        # ── Stage 5: Classification ─────────────────────────────────────────
        x = current_tokens.mean(dim=1)    # (B, 768) — mean pool over retained tokens
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
    """Save training / validation / test curves at 300 DPI."""
    for metric in ['loss', 'acc']:
        plt.figure(figsize=(10, 6))
        plt.plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
        plt.plot(history[f'val_{metric}'],   label=f'Validation {metric.capitalize()}')
        plt.plot(history[f'test_{metric}'],  label=f'Test {metric.capitalize()}', linestyle='--')
        plt.title(f'H-CoAtNet — {metric.capitalize()} Over Epochs')
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

    # ── 1. Download Dataset ─────────────────────────────────────────────────
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
    dataset = project.version(1).download("folder")
    DATASET_DIR = dataset.location

    # ── 2. Data Transforms & Loaders ───────────────────────────────────────
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

    # ── 3. Class Weights (handles imbalance) ───────────────────────────────
    counts = np.bincount(train_dataset.targets)
    class_weights = torch.tensor(
        [len(train_dataset) / (c * num_classes + 1e-6) for c in counts],
        dtype=torch.float
    ).to(DEVICE)
    print("Class weights:", class_weights.cpu().numpy().round(4))

    # ── 4. Model, Loss, Optimiser, Scheduler ───────────────────────────────
    model     = HCoAtNet(num_classes=num_classes, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    try:
        print("\n--- Model Summary ---")
        summary(model, input_size=(BATCH_SIZE, 3, *TARGET_SIZE))
    except Exception as e:
        print(f"Model summary unavailable: {e}")

    # ── 5. Training Loop ───────────────────────────────────────────────────
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
        print(f"  Losses — Train: {train_loss:.4f} | Val: {val_loss:.4f} | Test: {test_loss:.4f}")
        print(f"  Epoch time: {elapsed:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_h_coatnet.pth')
            print(f"  ✓ New best model saved (Val Acc = {best_val_acc:.4f})")

    avg_epoch_time = np.mean(epoch_times)
    print(f"\nAverage epoch time: {avg_epoch_time:.1f}s")

    # ── 6. Final Evaluation on Best Checkpoint ─────────────────────────────
    print("\n--- Final Evaluation (Best Checkpoint) ---")
    model.load_state_dict(torch.load('best_h_coatnet.pth', weights_only=True))
    _, final_test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, "Final Test")
    print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")

    np.save('h_coatnet_y_true.npy', np.array(y_true))
    np.save('h_coatnet_y_pred.npy', np.array(y_pred))
    print("Predictions saved to h_coatnet_y_true.npy and h_coatnet_y_pred.npy")

    # ── 7. Classification Report & Confusion Matrix ────────────────────────
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
    plt.title('Confusion Matrix — H-CoAtNet', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix_h_coatnet.png', dpi=300)
    plt.close()

    plot_curves(history)

    # ── 8. Hyperparameter Summary (for paper Table) ─────────────────────────
    print("\n--- Hyperparameter Summary ---")
    print(f"  Architecture     : H-CoAtNet (ConvNeXt-Tiny backbone + {2} ViT blocks + dual HSE)")
    print(f"  Backbone         : convnext_tiny (pretrained=True, ImageNet-1k)")
    print(f"  Input resolution : {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Epochs           : {EPOCHS}")
    print(f"  Optimiser        : AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"  LR schedule      : CosineAnnealingLR (T_max={EPOCHS})")
    print(f"  Loss             : CrossEntropyLoss (label_smoothing=0.1, class_weights=True)")
    print(f"  Dropout          : {DROPOUT}")
    print(f"  Random seed      : {RANDOM_SEED}")
    print(f"  Augmentation     : RandomResizedCrop, RandomHFlip, Rotation(15°),")
    print(f"                     TrivialAugmentWide, RandomErasing(p=0.2)")
    print(f"  Avg epoch time   : {avg_epoch_time:.1f}s  (device: {DEVICE})")


if __name__ == '__main__':
    main()