import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix
from roboflow import Roboflow

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Configuration
API_KEY = "gXuxxWEMFJ8nK73o7pN7"
TARGET_SIZE = (224, 224)
BATCH_SIZE = 24
EPOCHS = 30
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Fair CNN Model Definition

class FairCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(FairCNN, self).__init__()
        # A lighter convolutional backbone
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # A modern, lightweight classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Reduces each feature map to 1x1
            nn.Flatten(),
            nn.Linear(128, num_classes)    # Connects to the 128 channels
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Training, Evaluation, and Plotting

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
        
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() if len(all_preds) > 0 else 0.0
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
            
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() if len(all_preds) > 0 else 0.0
    return avg_loss, accuracy, all_targets, all_preds


def plot_curves(history):
    metrics = ['loss', 'acc']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history[f'train_{metric}'], label=f'Train {metric.capitalize()}')
        plt.plot(history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        plt.plot(history[f'test_{metric}'], label=f'Test {metric.capitalize()}', linestyle='--')
        plt.title(f'Model {metric.capitalize()} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'fair_cnn_{metric}_curves.png', dpi=300)
        plt.show()


# Main Execution Logic

def main():
    print(f"Using device: {DEVICE}")

    # 1. Download Dataset
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("hi-l9ueo").project("ich-s-7lnsj")
    version = project.version(1)
    dataset = version.download("folder", overwrite=True)
    DATASET_DIR = dataset.location

    train_path = os.path.join(DATASET_DIR, "train")
    valid_path = os.path.join(DATASET_DIR, "valid")
    test_path = os.path.join(DATASET_DIR, "test")

    # 2. Setup Transforms and Loaders
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    validation_dataset = datasets.ImageFolder(valid_path, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=val_test_transform)

    num_workers = 0 if os.name == 'nt' else 2
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, generator=g)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # 3. Model Initialization
    model = FairCNN(num_classes=num_classes).to(DEVICE)
    
    # Class Weights
    counts = np.bincount(train_dataset.targets)
    class_weights = torch.tensor(
        [len(train_dataset) / (c * num_classes + 1e-6) for c in counts], 
        dtype=torch.float
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n--- Model Summary ---")
    try:
        summary(model, input_size=(BATCH_SIZE, 3, *TARGET_SIZE))
    except Exception as e:
        print(f"Could not show model summary: {e}")

    # 4. Training Loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_loss': [], 'test_acc': []
    }
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, validation_loader, criterion, "Validating")
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, "Testing")

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"Losses: Train: {train_loss:.4f}, Val: {val_loss:.4f}, Test: {test_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_fair_cnn_model.pth')
            print(f"New best model saved with Val Acc: {best_val_acc:.4f}")

    # 5. Final Evaluation
    print("\n--- Final Evaluation on Best Model ---")
    if os.path.exists('best_fair_cnn_model.pth'):
        model.load_state_dict(torch.load('best_fair_cnn_model.pth', weights_only=True))
        _, _, y_true_final, y_pred_final = evaluate(model, test_loader, criterion, "Re-Eval")
        np.save('cnn_y_true.npy', np.array(y_true_final))
        np.save('cnn_y_pred.npy', np.array(y_pred_final))
        _, final_test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, "Final Test")
        print(f"Final Test Accuracy: {final_test_acc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Fair CNN Model')
        plt.savefig('confusion_matrix_fair_cnn.png', dpi=300)
        plt.show()

        plot_curves(history)
    else:
        print("No best model was saved. Skipping final evaluation.")


if __name__ == '__main__':
    main()