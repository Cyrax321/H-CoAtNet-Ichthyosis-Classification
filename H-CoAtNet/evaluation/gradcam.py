"""
H-CoAtNet: Grad-CAM Visualization
==================================
Generates class-discriminative heatmaps using Gradient-weighted Class
Activation Mapping (Grad-CAM) on the final ConvNeXt stage of H-CoAtNet.

Usage:
    python evaluation/gradcam.py --checkpoint best_h_coatnet.pth

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


# ── Model modules (self-contained, matches train_h_coatnet.py) ──────────────

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


class HCoAtNet(nn.Module):
    def __init__(self, num_classes=5, vit_blocks=2, dropout=0.2):
        super().__init__()
        cnn = create_model('convnext_tiny', pretrained=False, num_classes=0)
        self.cnn_stem   = cnn.stem
        self.cnn_stage1 = cnn.stages[0]
        self.cnn_stage2 = cnn.stages[1]
        self.cnn_stage3 = cnn.stages[2]
        self.cnn_stage4 = cnn.stages[3]

        vit_dim = 192
        self.mscaf = MultiScaleCrossAttentionFusion(
            dim_low=96, dim_high=192, num_heads=4, dropout=dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, 28 * 28, vit_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.vit_blocks = nn.ModuleList([
            Block(dim=vit_dim, num_heads=6, proj_drop=dropout, attn_drop=dropout * 0.5)
            for _ in range(vit_blocks)])

        final_dim = 768
        self.dcshse_blocks = nn.ModuleList([
            DualPathChannelSpatialHSE(dim=final_dim, reduction=16, dropout=dropout * 0.25)
            for _ in range(2)])
        self.latp_blocks = nn.ModuleList([
            AdaptiveTokenPruning(dim=final_dim, min_keep=0.4, max_keep=0.85),
            AdaptiveTokenPruning(dim=final_dim, min_keep=0.25, max_keep=0.7)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(final_dim), nn.Dropout(dropout), nn.Linear(final_dim, num_classes))

    def forward(self, x):
        x = self.cnn_stem(x)
        feat_s1 = self.cnn_stage1(x)
        feat_s2 = self.cnn_stage2(feat_s1)
        fused = self.mscaf(feat_s1, feat_s2)
        fused = fused + self.pos_embed
        for blk in self.vit_blocks:
            fused = blk(fused)
        B = fused.shape[0]
        x = fused.transpose(1, 2).reshape(B, 192, 28, 28)
        x = self.cnn_stage3(x)
        x = self.cnn_stage4(x)
        x = x.flatten(2).transpose(1, 2)
        current = x
        for dcshse, latp in zip(self.dcshse_blocks, self.latp_blocks):
            tokens_attn, importance = dcshse(current)
            current = latp(tokens_attn, importance)
        return self.classifier(current.mean(dim=1))


# ── Grad-CAM ────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.activations = o.detach()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        logits[0, class_idx].backward()
        w = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=TARGET_SIZE, mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


def tensor_to_rgb(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def apply_overlay(rgb, cam, alpha=0.45):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heatmap, alpha, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='best_h_coatnet.pth')
    parser.add_argument('--dataset_dir', default=None)
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_CLASS)
    args = parser.parse_args()

    if args.dataset_dir:
        DATASET_DIR = args.dataset_dir
    else:
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

    model = HCoAtNet(num_classes=num_classes).to(DEVICE)
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
    fig.suptitle("H-CoAtNet: Grad-CAM Activation Maps", fontsize=13, fontweight='bold', y=1.01)
    plt.savefig("gradcam/gradcam_grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\nPublication grid saved: gradcam/gradcam_grid.png")


if __name__ == '__main__':
    main()