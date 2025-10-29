import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Configuration
image_size = 98  # e.g., 98 (must be divisible by patch_size)
patch_size = 14  # e.g., 7
embed_dim = 4  # Transformer embedding dimension (must equal n_qubits_transformer for quantum attention)
num_heads = 2
num_blocks = 2
ffn_dim = 32
n_qubits_transformer = 4
n_qubits_ffn = 4
n_qlayers = 2
q_device = "lightning.gpu"  # Quantum device (e.g., default.qubit, braket.qubit, etc.)

dropout = 0.1
epochs = 10
batch_size = 8
learning_rate = 1e-5

df = pd.read_csv('rm_invalid.csv')

df = df.sample(frac=1).reset_index(drop=True)
df_mergedHard   = df[df['Label'] == 'mergedHard'].iloc[:1000]
df_notMerged    = df[df['Label'] == 'notMerged'].iloc[:1000]
df_notElectron  = df[df['Label'] == 'notElectron'].iloc[:1000]
df_limited = pd.concat([df_mergedHard, df_notMerged, df_notElectron], ignore_index=True)
df_limited = df_limited.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Prepare label mapping
labels = sorted(df_limited['Label'].unique().tolist())  # sorted unique labels
label_to_idx = {label: idx for idx, label in enumerate(labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(labels)
print("Classes:", labels)  # e.g., ['mergedHard', 'notElectron', 'notMerged']

df_limited['LabelIdx'] = df_limited['Label'].map(label_to_idx)

X = df_limited['ImagePath'].values
y = df_limited['LabelIdx'].values
# First split off 20% as test, then split the remaining 80% equally into train/val
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, 
    random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, 
    random_state=42, stratify=y_temp)
print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

# Data transformations and augmentation
train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    ## Random Zoom Out or In
    #transforms.RandomApply([
    #    transforms.RandomChoice([
    #        transforms.RandomAffine(degrees=0, scale=(0.6, 0.9), fill=0),  # zoom out
    #        transforms.RandomAffine(degrees=0, scale=(1.1, 1.4), fill=0)   # zoom in
    #    ])
    #], p=0.5),
    ## Random Rotation (±72° max)
    #transforms.RandomApply([transforms.RandomRotation(degrees=72)], p=0.5),
    ## Random Brightness
    #transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=0.5),
    ## Random Contrast
    #transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=0.5),
    ## Random Shear (~20°)
    #transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=20, fill=0)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transforms for validation/test (no augmentation)
test_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels  # numeric labels
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        # Open image (ensure 3 channels)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Create Dataset instances
train_dataset = ImageDataset(X_train, y_train, transform=train_transforms)
val_dataset   = ImageDataset(X_val,   y_val,   transform=test_transforms)
test_dataset  = ImageDataset(X_test,  y_test,  transform=test_transforms)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size*2, shuffle=False, num_workers=2)

import torch.nn as nn
from qvit import VisionTransformer

# Instantiate the Vision Transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(
    image_size=image_size,
    patch_size=patch_size,
    in_channels=3,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_blocks=num_blocks,
    num_classes=num_classes,
    ffn_dim=ffn_dim,
    n_qubits_transformer=n_qubits_transformer,
    n_qubits_ffn=n_qubits_ffn,
    n_qlayers=n_qlayers,
    dropout=dropout,
    q_device=q_device
)
model.to(device)

from sklearn.metrics import f1_score
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Define focal loss function
def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Compute the focal loss between `inputs` (logits) and `targets` (integer class indices).
    """
    # Standard cross-entropy (not averaged, per-sample)
    ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
    # p_t: probability of the true class for each sample
    p_t = torch.exp(-ce_loss)
    # Focal loss computation
    loss = alpha * ((1 - p_t) ** gamma) * ce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# Instantiate optimizer (AdamW)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Learning rate scheduler: ReduceLROnPlateau (monitor val F1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3)

# Early stopping parameters
patience = 100
best_val_f1 = -float('inf')
epochs_no_improve = 0

# Mixed precision setup
scaler = GradScaler()

import numpy as np

metrics_history = {
    "train_loss": [], "val_loss": [],
    "train_f1": [], "val_f1": []
}
best_model_path = f"/pscratch/sd/e/eoyun/4l/ckpts/pytorch/quantum_{pd.Timestamp.now():%Y%m%d_%H%M}/best_model.pth"
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
results_dir = f"/pscratch/sd/e/eoyun/4l//results/pytorch/quantum_{pd.Timestamp.now():%Y%m%d_%H%M}"
os.makedirs(results_dir, exist_ok=True)

for epoch in range(1, epochs+1):
    # Training
    model.train()
    train_losses = []
    all_train_preds = []
    all_train_labels = []
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Mixed precision forward and loss
        with autocast():
            logits = model(images)  # model outputs logits directly
            loss = focal_loss(logits, labels)
        train_losses.append(loss.item())
        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Collect predictions for F1
        preds = logits.argmax(dim=1)
        all_train_preds.append(preds.detach().cpu().numpy())
        all_train_labels.append(labels.detach().cpu().numpy())
    # Compute train loss and F1
    train_loss = np.mean(train_losses)
    all_train_preds = np.concatenate(all_train_preds)
    all_train_labels = np.concatenate(all_train_labels)
    train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
    
    # Validation
    model.eval()
    val_losses = []
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            with autocast():
                logits = model(images)
                loss = focal_loss(logits, labels)
            val_losses.append(loss.item())
            preds = logits.argmax(dim=1)
            all_val_preds.append(preds.cpu().numpy())
            all_val_labels.append(labels.cpu().numpy())
    val_loss = np.mean(val_losses)
    all_val_preds = np.concatenate(all_val_preds)
    all_val_labels = np.concatenate(all_val_labels)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
    
    # Record metrics
    metrics_history["train_loss"].append(train_loss)
    metrics_history["val_loss"].append(val_loss)
    metrics_history["train_f1"].append(train_f1)
    metrics_history["val_f1"].append(val_f1)
    
    # Print epoch results
    print(f"Epoch {epoch:03d}: Train Loss={train_loss:.4f}, Train F1={train_f1:.4f} | "
          f"Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")
    
    # LR scheduling step (monitor val F1)
    scheduler.step(val_f1)
    
    # Early stopping check
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_no_improve = 0
        # Save best model
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, best_model_path)
        best_epoch = epoch
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs. Early stopping at epoch {epoch}.")
            break

print(f"Best model was from epoch {best_epoch} with Val F1 = {best_val_f1:.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the best model for testing
ckpt = torch.load(best_model_path)
best_model = VisionTransformer(
    image_size=image_size,
    patch_size=patch_size,
    in_channels=3,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_blocks=num_blocks,
    num_classes=num_classes,
    ffn_dim=ffn_dim,
    n_qubits_transformer=n_qubits_transformer,
    n_qubits_ffn=n_qubits_ffn,
    n_qlayers=n_qlayers,
    dropout=dropout,
    q_device=q_device
)
best_model.load_state_dict(ckpt["model_state_dict"])
best_model.to(device)
best_model.eval()

# Get predictions on the test set
y_true = []
y_prob = []  # probabilities for each class
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = best_model(images)
        probs = torch.softmax(logits, dim=1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

y_true = np.array(y_true)
y_prob = np.array(y_prob)  # shape (n_samples, num_classes)

# Determine predicted class indices
y_pred_indices = np.argmax(y_prob, axis=1)
# Map indices to label names
y_pred_labels = [idx_to_label[idx] for idx in y_pred_indices]
y_true_labels = [idx_to_label[idx] for idx in y_true]

# Save predictions to CSV
df_submission = pd.DataFrame({"pred": y_pred_labels, "true": y_true_labels})
df_submission.to_csv(os.path.join(results_dir, "predictions.csv"), index=False)
print("Saved test predictions to CSV.")

# Plot training history (Loss and F1 over epochs)
epochs_range = range(1, len(metrics_history["train_loss"]) + 1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_range, metrics_history["train_loss"], label="Train Loss")
plt.plot(epochs_range, metrics_history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epochs"); plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, metrics_history["train_f1"], label="Train F1")
plt.plot(epochs_range, metrics_history["val_f1"], label="Val F1")
plt.xlabel("Epoch"); plt.ylabel("F1 Score"); plt.title("F1 Score vs Epochs"); plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "training_history.png"))
#plt.show()

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# 1) Check shapes
assert y_prob.ndim == 2, f"y_prob shape weird: {y_prob.shape}"
assert len(y_true) == y_prob.shape[0], f"len(y_true)={len(y_true)} vs y_prob={y_prob.shape}"

# 2) Class names
num_classes = y_prob.shape[1]
class_indices = list(range(num_classes))
try:
    class_names = [idx_to_label[i] for i in class_indices]
except Exception:
    class_names = [f"class_{i}" for i in class_indices]

# 3) Binarize
y_true_bin = label_binarize(y_true, classes=class_indices)  # shape: (N, num_classes)

# 4) Plot ROC (skip if one class)
plt.figure(figsize=(6, 6))
any_plotted = False
for i, name in enumerate(class_names):
    y_true_i = y_true_bin[:, i]
    # Skip if only one class in true labels
    if y_true_i.max() == 0 or y_true_i.min() == 1:
        print(f"[ROC] Skip '{name}': only one class present in y_true for this label.")
        continue
    fpr, tpr, _ = roc_curve(y_true_i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
    any_plotted = True

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve")
if any_plotted:
    plt.legend(loc="lower right")
else:
    plt.legend([], frameon=False)
plt.savefig(os.path.join(results_dir, "roc_curve.png"))
#plt.show()

