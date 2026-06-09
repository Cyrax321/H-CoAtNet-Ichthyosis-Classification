import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix
from roboflow import Roboflow
from timm import create_model
from timm.models.vision_transformer import Block
 
# === Configuration ===
API_KEY = "gXuxxWEMFJ8nK73o7pN7"
TARGET_SIZE = (224, 224)
BATCH_SIZE = 24
EPOCHS = 30
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

# ===========================
# Helper Module (Unchanged)
# ===========================
class HierarchicalSE(nn.Module):
   def __init__(self, dim, reduction=16, dropout=0.0):
       super().__init__()
       mid = max(1, dim // reduction)
       self.se = nn.Sequential(
           nn.Linear(dim, mid, bias=True),
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(mid, dim, bias=True),
           nn.Sigmoid()
       )
 
   def forward(self, x):
       s = x.mean(dim=1)
       gates = self.se(s).unsqueeze(1)
       out = x * gates
       token_scores = out.norm(dim=-1)
       token_scores = token_scores - token_scores.mean(dim=-1, keepdim=True)
       token_std = token_scores.std(dim=-1, keepdim=True) + 1e-6
       importance = F.softmax(token_scores / token_std, dim=-1)
       return out, importance
 

# =======================================================
# CoAtNet-Style GFT Model (MODIFIED WITH 6 ConvNeXt Blocks)
# =======================================================
class CoAtGFT(nn.Module):
   def __init__(self, base_model='convnext_tiny', num_classes=5, vit_blocks=2):
       super().__init__()
 
       # Load the standard model with num_classes=0 to get access to .stem and .stages
       cnn_backbone = create_model(base_model, pretrained=True, num_classes=0)
 
       # --- Full CNN Backbone ---
       self.cnn_stem   = cnn_backbone.stem
       self.cnn_stage1 = cnn_backbone.stages[0]
       self.cnn_stage2 = cnn_backbone.stages[1]
       self.cnn_stage3 = cnn_backbone.stages[2]
       self.cnn_stage4 = cnn_backbone.stages[3]
 
       # --- Transformer Blocks ---
       vit_dim = 192
       self.pos_embed = nn.Parameter(torch.zeros(1, 28 * 28, vit_dim))
       self.vit_blocks = nn.ModuleList([
           Block(dim=vit_dim, num_heads=6) for _ in range(vit_blocks)
       ])
 
       # --- Hierarchical Selection ---
       final_embed_dim = 768
       num_final_patches = 49
       self.selection_sizes = [
           int(num_final_patches * 0.75),
           int(num_final_patches * 0.5),
       ]
       self.hierarchical_blocks = nn.ModuleList([
           HierarchicalSE(dim=final_embed_dim, reduction=16, dropout=0.05) for _ in self.selection_sizes
       ])
 
       # --- Classifier Head ---
       self.classifier = nn.Sequential(
           nn.LayerNorm(final_embed_dim),
           nn.Linear(final_embed_dim, num_classes)
       )
 
   def select_patches(self, tokens, importance, k):
       B, N, C = tokens.size()
       k = min(k, N)
       _, top_k_idx = torch.topk(importance, k, dim=1)
       batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(-1, k)
       return tokens[batch_idx, top_k_idx]
 
   def forward(self, x):
       # --- Early CNN Stages (Now Deeper) ---
       x = self.cnn_stem(x)
       x = self.cnn_stage1(x)
       x = self.cnn_stage1(x)  # <-- ADDED BLOCK 1
       x = self.cnn_stage1(x)  # <-- ADDED BLOCK 2
       x = self.cnn_stage2(x)
 
       # --- Transformer Stage ---
       B, C, H, W = x.shape
       x = x.flatten(2).transpose(1, 2)
       x = x + self.pos_embed
 
       for blk in self.vit_blocks:
           x = blk(x)
 
       x = x.transpose(1, 2).reshape(B, C, H, W)
 
       # --- Later CNN Stages ---
       x = self.cnn_stage3(x)
       x = self.cnn_stage4(x)
 
       # --- Hierarchical Selection and Classification ---
       x = x.flatten(2).transpose(1, 2)
 
       current_tokens = x
       for attn_block, select_size in zip(self.hierarchical_blocks, self.selection_sizes):
           tokens_attn, importance = attn_block(current_tokens)
           current_tokens = self.select_patches(tokens_attn, importance, select_size)
 
       x = current_tokens.mean(dim=1)
       return self.classifier(x)
 

# ===========================
# Training, Evaluation, and Plotting (Unchanged)
# ===========================
def train_epoch(model, loader, criterion, optimizer):
   model.train()
   total_loss, all_preds, all_targets = 0.0, [], []
   for images, targets in tqdm(loader, desc="Training"):
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
   avg_loss = total_loss / len(loader)
   accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
   return avg_loss, accuracy
 

def evaluate(model, loader, criterion, desc="Evaluating"):
   model.eval()
   total_loss, all_preds, all_targets = 0.0, [], []
   with torch.no_grad():
       for images, targets in tqdm(loader, desc=desc):
           images, targets = images.to(DEVICE), targets.to(DEVICE)
           outputs = model(images)
           loss = criterion(outputs, targets)
           total_loss += loss.item()
           _, predicted = outputs.max(1)
           all_preds.extend(predicted.cpu().numpy())
           all_targets.extend(targets.cpu().numpy())
   avg_loss = total_loss / len(loader)
   accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
   return avg_loss, accuracy, all_targets, all_preds
 

def plot_curves(history):
   for metric in ['loss', 'acc']:
       plt.figure(figsize=(10, 6))
       plt.plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
       plt.plot(history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
       plt.plot(history[f'test_{metric}'], label=f'Test {metric.capitalize()}', linestyle='--')
       plt.title(f'Model {metric.capitalize()} Over Epochs')
       plt.xlabel('Epoch')
       plt.ylabel(metric.capitalize())
       plt.legend()
       plt.grid(True)
       plt.tight_layout()
       plt.savefig(f'{metric}_curves.png', dpi=300)
       plt.show()
 

# ===========================
# Main Training Logic
# ===========================
def main():
   print(f"Using device: {DEVICE}")
 
   # 1. Download Dataset
   print("🔄 Downloading dataset from Roboflow...")
   rf = Roboflow(api_key=API_KEY)
   project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
   dataset = project.version(1).download("folder")
   DATASET_DIR = dataset.location
 
   # 2. Setup DataLoaders
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
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
 
   train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, "train"), transform=train_transform)
   validation_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, "valid"), transform=val_test_transform)
   test_dataset = datasets.ImageFolder(root=os.path.join(DATASET_DIR, "test"), transform=val_test_transform)
 
   num_workers = 0 if os.name == 'nt' else 2
   train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
   validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
   test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
 
   class_names = train_dataset.classes
   num_classes = len(class_names)
   print(f"✅ Found {num_classes} classes: {class_names}")
 
   # 3. Class Weights
   counts = np.bincount(train_dataset.targets)
   class_weights = torch.tensor([len(train_dataset) / (c * num_classes + 1e-6) for c in counts], dtype=torch.float).to(
       DEVICE)
   print("Class Weights:", class_weights.cpu().numpy())
 
   # 4. Initialize Model, Loss, Optimizer
   model = CoAtGFT(num_classes=num_classes).to(DEVICE)
   criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
   optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
 
   try:
       print("\n--- Model Summary ---")
       summary(model, input_size=(BATCH_SIZE, 3, *TARGET_SIZE))
   except Exception as e:
       print(f"Could not show model summary due to: {e}")
 
   # 5. Main Training Loop
   history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_loss': [], 'test_acc': []}
   best_val_acc = 0.0
 
   for epoch in range(EPOCHS):
       print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
       train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
       val_loss, val_acc, _, _ = evaluate(model, validation_loader, criterion, desc="Validating")
       test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, desc="Testing")
       scheduler.step()
 
       history['train_loss'].append(train_loss);
       history['train_acc'].append(train_acc)
       history['val_loss'].append(val_loss);
       history['val_acc'].append(val_acc)
       history['test_loss'].append(test_loss);
       history['test_acc'].append(test_acc)
 
       print(f"📊 Epoch {epoch + 1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
       print(f"   Losses: Train: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}")
 
       if val_acc > best_val_acc:
           best_val_acc = val_acc
           torch.save(model.state_dict(), 'best_coat_gft_model.pth')
           print(f"🎉 New best model saved with Val Acc: {best_val_acc:.4f}")
 
   # 6. Final Evaluation
   print("\n--- Final Evaluation on Best Model ---")
   model.load_state_dict(torch.load('best_coat_gft_model.pth'))
   _, final_test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, desc="Final Test")
   print(f"✅ Final Test Accuracy: {final_test_acc:.4f}")
 
   # 7. Reports and Plots
   print("\n🧾 Classification Report:")
   print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
   cm = confusion_matrix(y_true, y_pred)
   plt.figure(figsize=(12, 10))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
   plt.xlabel('Predicted Label')
   plt.ylabel('True Label')
   plt.title('Confusion Matrix - CoAt-GFT Model')
   plt.tight_layout()
   plt.savefig('confusion_matrix_coat_gft.png', dpi=300)
   plt.show()
 
   plot_curves(history)
 

if __name__ == '__main__':
   main()
