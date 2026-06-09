"""
BiomedCLIP Fine-tuned Baseline
================================
Fine-tunes Microsoft BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
on the ichthyosis dataset. BiomedCLIP is pretrained on 15 million biomedical
image-text pairs from PubMed Central.

Usage:
    pip install open_clip_torch
    python baselines/train_biomedclip.py
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

BIOMEDCLIP_MODEL = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


# ── BiomedCLIP Vision Classifier ─────────────────────────────────────────────
class BiomedCLIPClassifier(nn.Module):
    """
    Fine-tunable classification wrapper around BiomedCLIP's vision encoder.
    Uses open_clip to load the pretrained vision transformer backbone,
    then attaches a dropout + linear classification head.
    """
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        try:
            import open_clip
        except ImportError:
            raise ImportError("Run: pip install open_clip_torch")

        # Load BiomedCLIP vision encoder via open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.vision_encoder = self.model.visual
        embed_dim = self.vision_encoder.output_tokens   # 512 for BiomedCLIP ViT-B/16

        # Freeze early layers, fine-tune top layers + head
        for name, param in self.vision_encoder.named_parameters():
            if 'blocks.10' in name or 'blocks.11' in name or 'norm' in name or 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        features = self.vision_encoder(x)
        if isinstance(features, tuple):
            features = features[0]
        return self.classifier(features)


# ── Simpler fallback: use open_clip's encode_image ───────────────────────────
class BiomedCLIPSimple(nn.Module):
    """
    Simpler wrapper that uses open_clip's full model.encode_image().
    More robust across open_clip versions.
    """
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        import open_clip
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        # Determine embedding dimension by a forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat  = self.clip_model.encode_image(dummy)
            embed_dim = feat.shape[-1]

        print(f"BiomedCLIP vision embedding dim: {embed_dim}")

        # Partially unfreeze: last 2 transformer blocks + LayerNorm
        unfrozen_keywords = ['blocks.10', 'blocks.11', 'ln_post', 'proj']
        for name, param in self.clip_model.visual.named_parameters():
            param.requires_grad = any(k in name for k in unfrozen_keywords)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        features = self.clip_model.encode_image(x)
        return self.classifier(features)


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


def plot_curves(history, prefix="biomedclip"):
    for metric in ['loss', 'acc']:
        plt.figure(figsize=(10, 6))
        for split in ['train', 'val', 'test']:
            ls = '--' if split == 'test' else '-'
            plt.plot(history[f'{split}_{metric}'], label=split.capitalize(), linestyle=ls)
        plt.title(f'BiomedCLIP (Fine-tuned) — {metric.capitalize()}')
        plt.xlabel('Epoch'); plt.ylabel(metric.capitalize())
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(f'{prefix}_{metric}_curves.png', dpi=300)
        plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE} | Seed: {RANDOM_SEED}")
    print(f"Loading BiomedCLIP: {BIOMEDCLIP_MODEL}")

    # Install open_clip if missing
    try:
        import open_clip
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'open_clip_torch'])
        import open_clip

    from roboflow import Roboflow
    rf      = Roboflow(api_key=API_KEY)
    dataset = rf.workspace("hi-l9ueo").project("ich-s-7lnsj").version(1).download("folder")
    DATASET_DIR = dataset.location

    # BiomedCLIP uses specific normalisation from its pretraining
    biomedclip_mean = [0.48145466, 0.4578275,  0.40821073]
    biomedclip_std  = [0.26862954, 0.26130258, 0.27577711]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=biomedclip_mean, std=biomedclip_std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=biomedclip_mean, std=biomedclip_std),
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

    model = BiomedCLIPSimple(num_classes=num_classes).to(DEVICE)
    n_params_total    = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_params_total:,} | Trainable: {n_params_trainable:,}")

    # Layer-wise LR: backbone (unfrozen layers) lower LR, head higher LR
    head_ids  = set(id(p) for p in model.classifier.parameters())
    backbone_trainable = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {'params': backbone_trainable,          'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(),'lr': LR_HEAD},
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
            torch.save(model.state_dict(), 'best_biomedclip.pth')
            print(f"  ✓ Best model saved (Val={best_val_acc:.4f})")

    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load('best_biomedclip.pth', weights_only=True))
    _, final_acc, y_true, y_pred = evaluate(model, test_loader, criterion, "Final Test")
    print(f"Test Accuracy: {final_acc*100:.2f}%")
    print(f"Macro F1:      {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    np.save('biomedclip_y_true.npy', y_true)
    np.save('biomedclip_y_pred.npy', y_pred)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
    plt.title('Confusion Matrix — BiomedCLIP (Fine-tuned)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=13); plt.ylabel('True', fontsize=13)
    plt.tight_layout(); plt.savefig('confusion_matrix_biomedclip.png', dpi=300); plt.close()
    plot_curves(history)


if __name__ == '__main__':
    main()