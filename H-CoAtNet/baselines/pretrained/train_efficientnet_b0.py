"""
EfficientNet-B0 Fine-tuned Baseline
=====================================
Fine-tunes EfficientNet-B0 (ImageNet pretrained via timm) on the ichthyosis dataset.
Uses layer-wise learning rate decay and label smoothing.

Usage:
    python baselines/train_efficientnet_pretrained.py
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
import timm
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RANDOM_SEED  = 42
random.seed(RANDOM_SEED);  np.random.seed(RANDOM_SEED);  torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)

TARGET_SIZE  = (224, 224)
BATCH_SIZE   = 24
EPOCHS       = 30
LR_BACKBONE  = 1e-5    # lower LR for pretrained layers
LR_HEAD      = 1e-4    # higher LR for new classifier head
WEIGHT_DECAY = 0.01
DROPOUT      = 0.2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    acc = (np.array(preds) == np.array(targets)).mean()
    return total_loss / len(loader), acc


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
    y_true = np.array(targets)
    y_pred = np.array(preds)
    acc = (y_pred == y_true).mean()
    return total_loss / len(loader), acc, y_true, y_pred


def plot_curves(history, prefix="efficientnet_pretrained"):
    for metric in ['loss', 'acc']:
        plt.figure(figsize=(10, 6))
        plt.plot(history[f'train_{metric}'], label=f'Train')
        plt.plot(history[f'val_{metric}'],   label=f'Validation')
        plt.plot(history[f'test_{metric}'],  label=f'Test', linestyle='--')
        plt.title(f'EfficientNet-B0 (Pretrained) — {metric.capitalize()}')
        plt.xlabel('Epoch'); plt.ylabel(metric.capitalize())
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(f'{prefix}_{metric}_curves.png', dpi=300)
        plt.close()


def main():
    from roboflow import Roboflow
    rf      = Roboflow(api_key="gXuxxWEMFJ8nK73o7pN7")
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
    print(f"Classes: {class_names}")

    # EfficientNet-B0 fine-tuned from ImageNet weights
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes, drop_rate=DROPOUT)
    model = model.to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Layer-wise LR: backbone gets 10× lower LR than the classification head
    head_params     = list(model.classifier.parameters())
    head_param_ids  = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params,     'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    counts = np.bincount(train_ds.targets)
    cw = torch.tensor(
        [len(train_ds) / (c * num_classes + 1e-6) for c in counts], dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)

    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']}
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, "Validating")
        te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion, "Testing")
        scheduler.step()

        for k, v in zip(['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'],
                         [tr_loss, tr_acc, vl_loss, vl_acc, te_loss, te_acc]):
            history[k].append(v)

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train {tr_acc:.4f} | Val {vl_acc:.4f} | Test {te_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), 'best_efficientnet_pretrained.pth')
            print(f"  ✓ Best model saved (Val Acc = {best_val_acc:.4f})")

    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load('best_efficientnet_pretrained.pth', weights_only=True))
    _, final_acc, y_true, y_pred = evaluate(model, test_loader, criterion, "Final Test")
    print(f"Test Accuracy: {final_acc*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    np.save('efficientnet_pretrained_y_true.npy', y_true)
    np.save('efficientnet_pretrained_y_pred.npy', y_pred)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
    plt.title('Confusion Matrix — EfficientNet-B0 (Pretrained)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=13); plt.ylabel('True', fontsize=13)
    plt.tight_layout()
    plt.savefig('confusion_matrix_efficientnet_pretrained.png', dpi=300)
    plt.close()

    plot_curves(history)
    print("\nHyperparameters:")
    print(f"  backbone LR: {LR_BACKBONE} | head LR: {LR_HEAD} | WD: {WEIGHT_DECAY}")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Dropout: {DROPOUT} | Seed: {RANDOM_SEED}")


if __name__ == '__main__':
    main()