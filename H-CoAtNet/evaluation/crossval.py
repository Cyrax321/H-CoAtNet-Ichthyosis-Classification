"""
WaveCoAtNet: 5-Fold Stratified Cross-Validation + McNemar's Test
===============================================================
Runs stratified k-fold cross-validation on WaveCoAtNet and computes
McNemar's test against all baselines whose prediction .npy files are present.

Usage:
    python evaluation/crossval.py

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold
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

TARGET_SIZE  = (224, 224)
BATCH_SIZE   = 24
EPOCHS       = 30
LR           = 5e-5
WEIGHT_DECAY = 0.01
DROPOUT      = 0.2
N_FOLDS      = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SCTR_WEIGHT  = 0.1
PROTO_MOM    = 0.999

# ── Model definitions (self-contained, matches train_wavecoatnet.py v2) ────────

def haar_dwt_2d(x):
    """2D Haar Discrete Wavelet Transform."""
    x_l = (x[:, :, :, 0::2] + x[:, :, :, 1::2]) * 0.5
    x_h = (x[:, :, :, 0::2] - x[:, :, :, 1::2]) * 0.5
    ll = (x_l[:, :, 0::2, :] + x_l[:, :, 1::2, :]) * 0.5
    lh = (x_l[:, :, 0::2, :] - x_l[:, :, 1::2, :]) * 0.5
    hl = (x_h[:, :, 0::2, :] + x_h[:, :, 1::2, :]) * 0.5
    hh = (x_h[:, :, 0::2, :] - x_h[:, :, 1::2, :]) * 0.5
    return ll, lh, hl, hh


class WaveletFrequencyDecomposedCrossAttention(nn.Module):
    """WG-FDCA: frequency-selective cross-attention via Haar DWT."""
    def __init__(self, dim_low=96, dim_high=192, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_high // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_low_freq = nn.Sequential(
            nn.Conv2d(dim_low, dim_high, 1, bias=False), nn.BatchNorm2d(dim_high), nn.GELU())
        self.proj_high_freq = nn.Sequential(
            nn.Conv2d(dim_low * 3, dim_high, 1, bias=False), nn.BatchNorm2d(dim_high), nn.GELU())
        self.q_proj = nn.Linear(dim_high, dim_high, bias=False)
        self.norm_q = nn.LayerNorm(dim_high)
        self.k_proj_low = nn.Linear(dim_high, dim_high, bias=False)
        self.v_proj_low = nn.Linear(dim_high, dim_high, bias=False)
        self.out_proj_low = nn.Linear(dim_high, dim_high)
        self.norm_kv_low = nn.LayerNorm(dim_high)
        self.k_proj_high = nn.Linear(dim_high, dim_high, bias=False)
        self.v_proj_high = nn.Linear(dim_high, dim_high, bias=False)
        self.out_proj_high = nn.Linear(dim_high, dim_high)
        self.norm_kv_high = nn.LayerNorm(dim_high)
        self.attn_drop = nn.Dropout(dropout * 0.5)
        self.proj_drop = nn.Dropout(dropout)
        self.freq_gate = nn.Sequential(
            nn.Linear(dim_high * 2, dim_high // 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_high // 4, 1), nn.Sigmoid())
        self.ffn = nn.Sequential(
            nn.Linear(dim_high, dim_high * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_high * 2, dim_high), nn.Dropout(dropout))
        self.norm_ffn = nn.LayerNorm(dim_high)

    def _cross_attend(self, q_tokens, kv_tokens, k_proj, v_proj, out_proj, norm_kv):
        B = q_tokens.shape[0]
        kv = norm_kv(kv_tokens)
        Q = self.q_proj(self.norm_q(q_tokens)).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = k_proj(kv).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = v_proj(kv).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = self.attn_drop((Q @ K.transpose(-2, -1) * self.scale).softmax(dim=-1))
        out = (attn @ V).transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        return self.proj_drop(out_proj(out))

    def forward(self, feat_low, feat_high):
        ll, lh, hl, hh = haar_dwt_2d(feat_low)
        low_tokens = self.proj_low_freq(ll).flatten(2).transpose(1, 2)
        high_tokens = self.proj_high_freq(torch.cat([lh, hl, hh], dim=1)).flatten(2).transpose(1, 2)
        q_tokens = feat_high.flatten(2).transpose(1, 2)
        low_out = self._cross_attend(q_tokens, low_tokens, self.k_proj_low, self.v_proj_low, self.out_proj_low, self.norm_kv_low)
        high_out = self._cross_attend(q_tokens, high_tokens, self.k_proj_high, self.v_proj_high, self.out_proj_high, self.norm_kv_high)
        gate = self.freq_gate(torch.cat([low_out, high_out], dim=-1))
        fused = q_tokens + gate * high_out + (1 - gate) * low_out
        return fused + self.ffn(self.norm_ffn(fused))


class PrototypeAnchoredTokenSelection(nn.Module):
    """PA-DTS: token selection via class prototype affinity + entropy + SE."""
    def __init__(self, dim, num_classes=5, min_keep=0.3, max_keep=0.8, dropout=0.0):
        super().__init__()
        self.dim = dim; self.num_classes = num_classes
        self.min_keep = min_keep; self.max_keep = max_keep
        self.register_buffer('prototypes', torch.randn(num_classes, dim) * 0.02)
        mid = max(1, dim // 16)
        self.channel_scorer = nn.Sequential(nn.Linear(dim, mid), nn.GELU(), nn.Dropout(dropout), nn.Linear(mid, 1))
        self.importance_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.5]))
        self.keep_predictor = nn.Sequential(nn.Linear(dim + 3, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape; x_n = self.norm(x)
        p_n = F.normalize(self.prototypes, dim=-1); t_n = F.normalize(x_n, dim=-1)
        sim = t_n @ p_n.T; aff = sim.max(-1).values
        probs = F.softmax(sim / 0.1, -1); ent = -(probs * (probs + 1e-8).log()).sum(-1)
        ch = self.channel_scorer(x_n).squeeze(-1)
        def _zn(s):
            return (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-6)
        w = F.softmax(self.importance_weights, 0)
        imp = F.softmax(w[0]*_zn(aff) + w[1]*_zn(ent) + w[2]*_zn(ch), -1)
        g = self.keep_predictor(torch.cat([x.mean(1), torch.stack([imp.mean(1), imp.std(1), imp.max(1).values], -1)], -1)).squeeze(-1)
        g = self.min_keep + g * (self.max_keep - self.min_keep)
        k = torch.clamp((g*N).long(), min=max(1, int(self.min_keep*N)), max=int(self.max_keep*N))[0].item()
        _, idx = torch.topk(imp, k, dim=1)
        bi = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k)
        return x[bi, idx] * (1 + imp[bi, idx].unsqueeze(-1)), imp

    @torch.no_grad()
    def update_prototypes(self, embeddings, labels, momentum=0.999):
        for c in range(self.num_classes):
            m = labels == c
            if m.sum() > 0:
                self.prototypes[c] = momentum*self.prototypes[c] + (1-momentum)*embeddings[m].mean(0)


class SupervisedContrastiveTokenLoss(nn.Module):
    """SCTR: SupCon on mean-pooled token embeddings."""
    def __init__(self, embed_dim, proj_dim=128, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, proj_dim))

    def forward(self, embeddings, labels):
        B = embeddings.shape[0]
        if B < 2: return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        z = F.normalize(self.projector(embeddings), -1)
        sim = z @ z.T / self.temperature
        leq = labels.unsqueeze(0) == labels.unsqueeze(1)
        sm = ~torch.eye(B, dtype=torch.bool, device=z.device)
        pos = leq & sm; hp = pos.float().sum(1) > 0
        if hp.sum() == 0: return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        sim = sim - sim.max(1, keepdim=True).values.detach()
        lp = sim - torch.log((torch.exp(sim)*sm.float()).sum(1, keepdim=True) + 1e-8)
        loss = -(pos.float()*lp).sum(1) / torch.clamp(pos.float().sum(1), min=1.0)
        return loss[hp].mean()


class WaveCoAtNet(nn.Module):
    """WaveCoAtNet with WG-FDCA + PA-DTS + SCTR."""
    def __init__(self, num_classes=5, vit_blocks=2, dropout=0.2):
        super().__init__()
        cnn = create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.cnn_stem   = cnn.stem
        self.cnn_stage1 = cnn.stages[0]
        self.cnn_stage2 = cnn.stages[1]
        self.cnn_stage3 = cnn.stages[2]
        self.cnn_stage4 = cnn.stages[3]

        vit_dim = 192
        self.wg_fdca = WaveletFrequencyDecomposedCrossAttention(96, 192, 4, dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, 28*28, vit_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.vit_blocks = nn.ModuleList([
            Block(dim=vit_dim, num_heads=6, proj_drop=dropout, attn_drop=dropout*0.5)
            for _ in range(vit_blocks)])

        final_dim = 768
        self.pa_dts = PrototypeAnchoredTokenSelection(final_dim, num_classes, 0.3, 0.8, dropout*0.25)
        self.sctr = SupervisedContrastiveTokenLoss(final_dim, 128, 0.07)
        self.classifier = nn.Sequential(nn.LayerNorm(final_dim), nn.Dropout(dropout), nn.Linear(final_dim, num_classes))

    def forward(self, x, return_embeddings=False):
        x = self.cnn_stem(x)
        s1 = self.cnn_stage1(x); s2 = self.cnn_stage2(s1)
        fused = self.wg_fdca(s1, s2) + self.pos_embed
        for blk in self.vit_blocks: fused = blk(fused)
        B = fused.shape[0]; x = fused.transpose(1, 2).reshape(B, 192, 28, 28)
        x = self.cnn_stage3(x); x = self.cnn_stage4(x)
        x = x.flatten(2).transpose(1, 2)
        selected, _ = self.pa_dts(x)
        embeddings = selected.mean(dim=1)
        logits = self.classifier(embeddings)
        if return_embeddings: return logits, embeddings
        return logits


# ── Training & evaluation ────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, preds, targets = 0.0, [], []
    for imgs, tgts in tqdm(loader, desc="  train", leave=False):
        imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
        optimizer.zero_grad()
        logits, emb = model(imgs, return_embeddings=True)
        ce = criterion(logits, tgts)
        sctr = model.sctr(emb, tgts)
        loss = ce + SCTR_WEIGHT * sctr
        loss.backward()
        optimizer.step()
        model.pa_dts.update_prototypes(emb.detach(), tgts, PROTO_MOM)
        total_loss += loss.item()
        preds.extend(logits.argmax(1).cpu().numpy())
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


def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    if b + c == 0:
        return 0.0, 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000, alpha=0.05):
    n = len(y_true)
    rng = np.random.default_rng(RANDOM_SEED)
    scores = [metric_fn(y_true[rng.integers(0, n, size=n)],
                         y_pred[rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    return np.percentile(scores, 100 * alpha / 2), np.percentile(scores, 100 * (1 - alpha / 2))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    from roboflow import Roboflow
    rf = Roboflow(api_key="gXuxxWEMFJ8nK73o7pN7")
    dataset = rf.workspace("hi-l9ueo").project("ich-s-7lnsj").version(1).download("folder")
    DATASET_DIR = dataset.location

    val_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
        transforms.TrivialAugmentWide(), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))])

    full_train = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=val_transform)
    full_val   = datasets.ImageFolder(os.path.join(DATASET_DIR, "valid"), transform=val_transform)
    full_test  = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"),  transform=val_transform)

    all_targets = list(full_train.targets) + list(full_val.targets) + list(full_test.targets)
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

        def make_fold_dataset(global_indices, aug=False):
            train_part = [gi for gi in global_indices if gi < n_train]
            val_part = [gi - n_train for gi in global_indices if n_train <= gi < n_train + n_val]
            test_part = [gi - n_train - n_val for gi in global_indices if gi >= n_train + n_val]
            subsets = []
            if train_part:
                ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"),
                                          transform=train_aug if aug else val_transform)
                subsets.append(Subset(ds, train_part))
            if val_part:
                subsets.append(Subset(full_val, val_part))
            if test_part:
                subsets.append(Subset(full_test, test_part))
            return ConcatDataset(subsets)

        fold_train_ds = make_fold_dataset(train_idx, aug=True)
        fold_test_ds  = make_fold_dataset(test_idx, aug=False)

        num_workers = 0 if os.name == 'nt' else 2
        fold_train_loader = DataLoader(fold_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
        fold_test_loader  = DataLoader(fold_test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

        fold_labels = [all_targets[i] for i in train_idx]
        counts = np.bincount(fold_labels, minlength=num_classes)
        cw = torch.tensor(
            [len(fold_labels) / (c * num_classes + 1e-6) for c in counts], dtype=torch.float).to(DEVICE)

        model     = WaveCoAtNet(num_classes=num_classes, dropout=DROPOUT).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(EPOCHS):
            tr_loss, tr_acc = train_one_epoch(model, fold_train_loader, criterion, optimizer)
            scheduler.step()
            if epoch % 5 == 0 or epoch == EPOCHS - 1:
                print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Train Acc: {tr_acc:.4f}")
            if tr_loss < best_val_loss:
                best_val_loss = tr_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        _, y_true_fold, y_pred_fold = eval_loader(model, fold_test_loader, criterion)

        acc      = accuracy_score(y_true_fold, y_pred_fold)
        macro_f1 = f1_score(y_true_fold, y_pred_fold, average='macro', zero_division=0)
        wtd_f1   = f1_score(y_true_fold, y_pred_fold, average='weighted', zero_division=0)
        acc_lo, acc_hi = bootstrap_ci(y_true_fold, y_pred_fold, accuracy_score)

        print(f"\n  Fold {fold+1}: Acc={acc*100:.2f}% (CI: {acc_lo*100:.2f}-{acc_hi*100:.2f}%)")
        print(f"    Macro F1={macro_f1:.4f}  Wtd F1={wtd_f1:.4f}")

        fold_results.append({
            'fold': fold + 1, 'accuracy': acc, 'acc_ci_lo': acc_lo, 'acc_ci_hi': acc_hi,
            'macro_f1': macro_f1, 'weighted_f1': wtd_f1})
        all_y_true.extend(y_true_fold.tolist())
        all_y_pred.extend(y_pred_fold.tolist())

        cm = confusion_matrix(y_true_fold, y_pred_fold)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 11})
        plt.title(f'WaveCoAtNet Fold {fold+1} Confusion Matrix', fontsize=13, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12); plt.ylabel('True', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'fold_{fold+1}_cm.png', dpi=300)
        plt.close()

        np.save(f'fold_{fold+1}_y_true.npy', y_true_fold)
        np.save(f'fold_{fold+1}_y_pred.npy', y_pred_fold)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accs = [r['accuracy'] for r in fold_results]
    mf1s = [r['macro_f1'] for r in fold_results]
    wf1s = [r['weighted_f1'] for r in fold_results]

    summary_text = "\n".join([
        "=" * 60,
        "5-Fold Cross-Validation Summary -- WaveCoAtNet",
        "=" * 60,
        f"Accuracy   : {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%",
        f"Macro F1   : {np.mean(mf1s):.4f} +/- {np.std(mf1s):.4f}",
        f"Weighted F1: {np.mean(wf1s):.4f} +/- {np.std(wf1s):.4f}",
    ] + [f"  Fold {r['fold']}: Acc={r['accuracy']*100:.2f}%  F1={r['macro_f1']:.4f}" for r in fold_results])

    print("\n" + summary_text)
    with open('crossval_summary.txt', 'w') as f:
        f.write(summary_text + "\n")

    with open('crossval_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'accuracy', 'acc_ci_lo', 'acc_ci_hi', 'macro_f1', 'weighted_f1'])
        writer.writeheader()
        writer.writerows(fold_results)

    # McNemar's test
    print("\n--- McNemar's Test vs Baselines ---")
    all_yt = np.array(all_y_true)
    all_yp = np.array(all_y_pred)

    baselines = {
        'EfficientNet-B0 (pretrained)': 'efficientnet_pretrained_y_pred.npy',
        'Swin-T (pretrained)':          'swin_pretrained_y_pred.npy',
        'ViT-B/16 (pretrained)':        'vit_pretrained_y_pred.npy',
        'CoAtNet':                       'coatnet_y_pred.npy',
        'GFT':                           'gft_y_pred.npy',
        'BiomedCLIP':                    'biomedclip_y_pred.npy',
        'DINOv2':                        'dinov2_y_pred.npy',
    }

    mcnemar_rows = []
    for name, f in baselines.items():
        if os.path.exists(f):
            bp = np.load(f)
            ml = min(len(all_yt), len(bp))
            chi2_s, pv = mcnemar_test(all_yt[:ml], all_yp[:ml], bp[:ml])
            sig = "significant" if pv < 0.05 else "not significant"
            print(f"  vs {name}: chi2={chi2_s:.3f}, p={pv:.4f} ({sig})")
            mcnemar_rows.append({'baseline': name, 'chi2': chi2_s, 'p_value': pv, 'significant': pv < 0.05})
        else:
            print(f"  vs {name}: SKIPPED ({f} not found)")

    if mcnemar_rows:
        with open('mcnemar_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['baseline', 'chi2', 'p_value', 'significant'])
            writer.writeheader()
            writer.writerows(mcnemar_rows)


if __name__ == '__main__':
    main()