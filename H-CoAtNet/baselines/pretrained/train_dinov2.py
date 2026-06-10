"""
DINOv2 Fine-tuned Baseline
============================
Fine-tunes Facebook DINOv2 (facebook/dinov2-base) on the ichthyosis dataset.
DINOv2 is a self-supervised vision foundation model trained on 142M curated images.

Usage:
    pip install transformers
    python baselines/train_dinov2.py
"""

import os
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

# ── Configuration ────────────────────────────────────────────────────────────
API_KEY      = "gXuxxWEMFJ8nK73o7pN7"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DINOV2_MODEL = "facebook/dinov2-base"   # 86M params, ViT-B/14

TARGET_SIZE  = (224, 224)
BATCH_SIZE   = 24
EPOCHS       = 30
LR_BACKBONE  = 1e-5
LR_HEAD      = 1e-4
WEIGHT_DECAY = 0.01



# ── DINOv2 Classifier ─────────────────────────────────────────────────────────
class DINOv2Classifier(nn.Module):
    """
    DINOv2-Base fine-tuned for classification.
    - Blocks 0–9: frozen (ImageNet knowledge preserved)
    - Blocks 10–11 + LayerNorm: fine-tuned (domain adaptation)
    - Classification head: Linear 768 → num_classes
    """
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(DINOV2_MODEL)

        # Freeze first N-2 blocks
        n_blocks = len(self.backbone.encoder.layer)
        freeze_up_to = n_blocks - 2   # freeze all except last 2 blocks
        print(f"DINOv2: {n_blocks} transformer blocks | Freezing first {freeze_up_to}")

        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.backbone.encoder.layer):
            for param in layer.parameters():
                param.requires_grad = i >= freeze_up_to

        # Always fine-tune LayerNorm
        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

        hidden_size = self.backbone.config.hidden_size   # 768 for ViT-B
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        # Use CLS token (first token of last hidden state)
        cls_token = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)


# ── Train / Eval helpers ─────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, preds, targets = 0.0, [], []
    for imgs, tgts in tqdm(loader, desc="Training", leave=False):
        imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, tgts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(tgts.cpu().numpy())
    return total_loss / len(loader), (np.array(preds) == np.array(targets)).mean()


@torch.no_grad()
def evaluate(model, loader, criterion, desc="Eval"):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for imgs, tgts in tqdm(loader, desc=desc, leave=False):
        imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
        out  = model(imgs)
        total_loss += criterion(out, tgts).item()
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(tgts.cpu().numpy())
    y_true = np.array(targets); y_pred = np.array(preds)
    return total_loss / len(loader), (y_pred == y_true).mean(), y_true, y_pred


def plot_curves(history, prefix="dinov2"):
    for metric in ['loss', 'acc']:
        plt.figure(figsize=(10, 6))
        for split in ['train', 'val', 'test']:
            ls = '--' if split == 'test' else '-'
            plt.plot(history[f'{split}_{metric}'], label=split.capitalize(), linestyle=ls)
        plt.title(f'DINOv2-Base (Fine-tuned) — {metric.capitalize()}')
        plt.xlabel('Epoch'); plt.ylabel(metric.capitalize())
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(f'{prefix}_{metric}_curves.png', dpi=300)
        plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE} | Model: {DINOV2_MODEL} | Seed: {RANDOM_SEED}")

    # Install transformers if missing
    try:
        from transformers import AutoModel
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers'])

    from roboflow import Roboflow
    rf      = Roboflow(api_key=API_KEY)
    dataset = rf.workspace("hi-l9ueo").project("ich-s-7lnsj").version(1).download("folder")
    DATASET_DIR = dataset.location

    # DINOv2 was pretrained with ImageNet normalisation
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
    num_classes  = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    model = DINOv2Classifier(num_classes=num_classes).to(DEVICE)
    n_params_total     = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_params_total:,} | Trainable: {n_params_trainable:,}")

    # Layer-wise LR
    head_ids        = set(id(p) for p in model.classifier.parameters())
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params,               'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    counts = np.bincount(train_ds.targets)
    cw     = torch.tensor([len(train_ds) / (c * num_classes + 1e-6) for c in counts], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    history = {k: [] for k in ['train_loss','train_acc','val_loss','val_acc','test_loss','test_acc']}
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, "Validating")
        te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion, "Testing")
        scheduler.step()
        for k, v in zip(['train_loss','train_acc','val_loss','val_acc','test_loss','test_acc'],
                         [tr_loss, tr_acc, vl_loss, vl_acc, te_loss, te_acc]):
            history[k].append(v)
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train {tr_acc:.4f} | Val {vl_acc:.4f} | Test {te_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), 'best_dinov2.pth')
            print(f"  ✓ Best model saved (Val={best_val_acc:.4f})")

    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load('best_dinov2.pth', weights_only=True))
    _, final_acc, y_true, y_pred = evaluate(model, test_loader, criterion, "Final Test")
    print(f"Test Accuracy: {final_acc*100:.2f}%")
    print(f"Macro F1:      {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    np.save('dinov2_y_true.npy', y_true)
    np.save('dinov2_y_pred.npy', y_pred)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
    plt.title('Confusion Matrix — DINOv2-Base (Fine-tuned)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=13); plt.ylabel('True', fontsize=13)
    plt.tight_layout(); plt.savefig('confusion_matrix_dinov2.png', dpi=300); plt.close()
    plot_curves(history)


if __name__ == '__main__':
    main()