"""
WaveCoAtNet: Grad-CAM Visualization
======================================
Generates class-discriminative heatmaps using Gradient-weighted Class
Activation Mapping (Grad-CAM) on the final ConvNeXt stage of WaveCoAtNet.

Usage:
    python evaluation/gradcam.py --checkpoint best_wavecoatnet.pth

Outputs:
    gradcam/<ClassName>_sample<N>_<correct|wrong>.png  -- overlay at 300 DPI
    gradcam/gradcam_grid.png                            -- publication-quality grid
"""

import os
import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from timm import create_model
from timm.models.vision_transformer import Block

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
SAMPLES_PER_CLASS = 3


# ── Model modules (self-contained, matches train_wavecoatnet.py v2) ────────────

def haar_dwt_2d(x):
    x_l = (x[:, :, :, 0::2] + x[:, :, :, 1::2]) * 0.5
    x_h = (x[:, :, :, 0::2] - x[:, :, :, 1::2]) * 0.5
    ll = (x_l[:, :, 0::2, :] + x_l[:, :, 1::2, :]) * 0.5
    lh = (x_l[:, :, 0::2, :] - x_l[:, :, 1::2, :]) * 0.5
    hl = (x_h[:, :, 0::2, :] + x_h[:, :, 1::2, :]) * 0.5
    hh = (x_h[:, :, 0::2, :] - x_h[:, :, 1::2, :]) * 0.5
    return ll, lh, hl, hh


class WaveletFrequencyDecomposedCrossAttention(nn.Module):
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


class SupervisedContrastiveTokenLoss(nn.Module):
    def __init__(self, embed_dim, proj_dim=128, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, proj_dim))

    def forward(self, embeddings, labels):
        return torch.tensor(0.0)  # Not needed at inference


class WaveCoAtNet(nn.Module):
    """WaveCoAtNet — must match train_wavecoatnet.py exactly for checkpoint loading."""
    def __init__(self, num_classes=5, vit_blocks=2, dropout=0.2):
        super().__init__()
        cnn = create_model('convnext_tiny', pretrained=False, num_classes=0)
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


# ── Grad-CAM ────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)
        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=TARGET_SIZE, mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


def tensor_to_rgb(t):
    img = t.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def apply_overlay(img, cam, alpha=0.4):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(alpha * heatmap + (1 - alpha) * img, 0, 255).astype(np.uint8)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="WaveCoAtNet Grad-CAM")
    parser.add_argument('--checkpoint', default='best_wavecoatnet.pth')
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_CLASS)
    args = parser.parse_args()

    from roboflow import Roboflow
    rf = Roboflow(api_key="gXuxxWEMFJ8nK73o7pN7")
    dataset = rf.workspace("hi-l9ueo").project("ich-s-7lnsj").version(1).download("folder")
    DATASET_DIR = dataset.location

    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())])
    test_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=transform)
    class_names = test_ds.classes
    num_classes = len(class_names)

    model = WaveCoAtNet(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()

    gradcam = GradCAM(model, model.cnn_stage4)
    os.makedirs("gradcam", exist_ok=True)

    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(test_ds.samples):
        class_indices[label].append(idx)

    all_rows = []
    for ci, cls_name in enumerate(class_names):
        indices = class_indices[ci]
        random.shuffle(indices)
        row = []
        for sn, si in enumerate(indices[:args.samples]):
            img_t, tl = test_ds[si]
            inp = img_t.unsqueeze(0).to(DEVICE).requires_grad_(True)
            with torch.enable_grad():
                cam = gradcam.generate(inp, class_idx=ci)
            with torch.no_grad():
                pred = model(inp).argmax(1).item()
            correct = pred == tl
            rgb = tensor_to_rgb(img_t)
            ov = apply_overlay(rgb, cam, 0.4)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(rgb); axes[0].set_title("Original", fontsize=11)
            axes[1].imshow(ov); axes[1].set_title(
                f"Grad-CAM\nPred: {class_names[pred]} ({'correct' if correct else 'wrong'})", fontsize=11)
            for ax in axes: ax.axis('off')
            fig.suptitle(f"{cls_name} - Sample {sn+1}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            fname = f"gradcam/{cls_name}_sample{sn+1}_{'correct' if correct else 'wrong'}.png"
            plt.savefig(fname, dpi=300); plt.close()
            print(f"  Saved: {fname}")
            row.append(ov)

        while len(row) < args.samples:
            row.append(np.zeros((*TARGET_SIZE, 3), dtype=np.uint8))
        all_rows.append((cls_name, row))

    gradcam.remove_hooks()

    fig = plt.figure(figsize=(args.samples * 3.5, num_classes * 3.5))
    gs = gridspec.GridSpec(num_classes, args.samples, figure=fig, hspace=0.35, wspace=0.05)
    for r, (cn, imgs) in enumerate(all_rows):
        for c, ov in enumerate(imgs[:args.samples]):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(ov); ax.axis('off')
            if c == 0: ax.set_ylabel(cn, fontsize=10, fontweight='bold', rotation=90, labelpad=5)
            if r == 0: ax.set_title(f"Sample {c+1}", fontsize=10)
    fig.suptitle("WaveCoAtNet: Grad-CAM Activation Maps", fontsize=13, fontweight='bold', y=1.01)
    plt.savefig("gradcam/gradcam_grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\nPublication grid saved: gradcam/gradcam_grid.png")


if __name__ == '__main__':
    main()