"""
H-CoAtNet: Grad-CAM Visualization
==================================
Generates class-discriminative heatmaps using Gradient-weighted Class
Activation Mapping (Grad-CAM) on the final ConvNeXt stage of H-CoAtNet.

Usage:
    python proposed_method/generate_gradcam.py --checkpoint best_h_coatnet.pth

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
SAMPLES_PER_CLASS = 3   # generate 3 examples per class


# ── Model (self-contained, matches train_h_coatnet.py) ──────────────────────
class HierarchicalSE(nn.Module):
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
        scores = out.norm(dim=-1)
        scores = scores - scores.mean(dim=-1, keepdim=True)
        importance = F.softmax(scores / (scores.std(dim=-1, keepdim=True) + 1e-6), dim=-1)
        return out, importance


class HCoAtNet(nn.Module):
    def __init__(self, num_classes=5, vit_blocks=2, dropout=0.2):
        super().__init__()
        cnn = create_model('convnext_tiny', pretrained=False, num_classes=0)
        self.cnn_stem   = cnn.stem
        self.cnn_stage1 = cnn.stages[0]
        self.cnn_stage2 = cnn.stages[1]
        self.cnn_stage3 = cnn.stages[2]
        self.cnn_stage4 = cnn.stages[3]
    """
    Registers forward and backward hooks on `target_layer` to capture
    activations and gradients for Grad-CAM computation.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients   = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Returns a [0,1]-normalised Grad-CAM heatmap (H, W) for `class_idx`.
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score  = logits[0, class_idx]
        score.backward()

        # Global average pool of gradients → weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=TARGET_SIZE, mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Overlay helper ──────────────────────────────────────────────────────────
def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Denormalise an ImageNet-normalised tensor to uint8 RGB (H, W, 3)."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def apply_colormap_overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend RGB image with JET heatmap overlay. Returns uint8 (H, W, 3)."""
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for H-CoAtNet")
    parser.add_argument('--checkpoint', default='best_h_coatnet.pth')
    parser.add_argument('--dataset_dir', default=None,
                        help="Path to dataset root (with test/ subfolder). "
                             "If not provided, downloads from Roboflow.")
    parser.add_argument('--samples', type=int, default=SAMPLES_PER_CLASS)
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────────
    if args.dataset_dir:
        DATASET_DIR = args.dataset_dir
    else:
        from roboflow import Roboflow
        rf = Roboflow(api_key="gXuxxWEMFJ8nK73o7pN7")
        dataset = rf.workspace("hi-l9ueo").project("ich-s-7lnsj").version(1).download("folder")
        DATASET_DIR = dataset.location

    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])
    test_ds     = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=transform)
    class_names = test_ds.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    # ── Load model ───────────────────────────────────────────────────────────
    model = HCoAtNet(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Setup Grad-CAM on cnn_stage4 (final ConvNeXt stage) ──────────────────
    gradcam = GradCAM(model, model.cnn_stage4)

    os.makedirs("gradcam", exist_ok=True)

    # ── Collect samples per class ─────────────────────────────────────────────
    # Group indices by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(test_ds.samples):
        class_indices[label].append(idx)

    all_rows = []   # for publication grid: one row per class, `samples` columns

    for class_idx, cls_name in enumerate(class_names):
        indices = class_indices[class_idx]
        random.shuffle(indices)
        picked_indices = indices[:args.samples]
        row_images = []

        for sample_n, sample_idx in enumerate(picked_indices):
            img_tensor, true_label = test_ds[sample_idx]
            input_tensor = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)

            # Forward + Grad-CAM for the true class
            with torch.enable_grad():
                cam = gradcam.generate(input_tensor, class_idx=class_idx)

            # Get prediction
            with torch.no_grad():
                logits = model(input_tensor)
            pred_label = logits.argmax(1).item()
            pred_name  = class_names[pred_label]
            correct    = pred_label == true_label
            outcome    = "correct" if correct else "wrong"

            rgb  = tensor_to_rgb(img_tensor)
            overlay = apply_colormap_overlay(rgb, cam, alpha=0.4)

            # Save individual image
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(rgb);     axes[0].set_title("Original",  fontsize=11)
            axes[1].imshow(overlay); axes[1].set_title(
                f"Grad-CAM\nPred: {pred_name} ({'✓' if correct else '✗'})", fontsize=11
            )
            for ax in axes:
                ax.axis('off')
            fig.suptitle(f"{cls_name} — Sample {sample_n + 1} ({outcome})",
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            fname = f"gradcam/{cls_name}_sample{sample_n+1}_{outcome}.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"  Saved: {fname}")
            row_images.append(overlay)

        # Pad row if fewer samples than requested
        while len(row_images) < args.samples:
            row_images.append(np.zeros((*TARGET_SIZE, 3), dtype=np.uint8))
        all_rows.append((cls_name, row_images))

    gradcam.remove_hooks()

    # ── Publication-quality grid (classes × samples) ─────────────────────────
    n_rows = num_classes
    n_cols = args.samples
    fig = plt.figure(figsize=(n_cols * 3.5, n_rows * 3.5))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                            hspace=0.35, wspace=0.05)

    for r, (cls_name, row_images) in enumerate(all_rows):
        for c, overlay in enumerate(row_images[:n_cols]):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(overlay)
            ax.axis('off')
            if c == 0:
                ax.set_ylabel(cls_name, fontsize=10, fontweight='bold',
                              rotation=90, labelpad=5)
            if r == 0:
                ax.set_title(f"Sample {c+1}", fontsize=10)

    fig.suptitle("H-CoAtNet — Grad-CAM Activation Maps (cnn_stage4 target layer)",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig("gradcam/gradcam_grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\nPublication grid saved: gradcam/gradcam_grid.png")


if __name__ == '__main__':
    main()