import os
import numpy as np
import matplotlib.pyplot as plt
#Per la confusion matrix
import seaborn as sns
 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
 
 

# Setup del modello

# Percorso immagini per il training

TRAIN_DIR = "seg_train/seg_train"
TEST_DIR  = "seg_test/seg_test"
 
BATCH  = 32 # Ho scelto 32 immagini alla volta per testare ma eventualmente si può aumentare
NUM_EPOCH  = 10
LR         = 1e-4        
VAL_SPLIT   = 0.2         # 20% del training set per la val
NUM_CLASS = 6             
DEVICE      = torch.device("cpu")
 
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
 

 
# Adatto le immaggini e le normalizzo per essere elaborate 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
 
train_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
 
 
# Carico il Dataset

full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
 
# Indici per train/val split stratificato
all_indices = list(range(len(full_train_dataset)))
all_labels  = [full_train_dataset.targets[i] for i in all_indices]
 
train_indices, val_indices = train_test_split(
    all_indices,
    test_size=VAL_SPLIT,
    stratify=all_labels,
)
 

val_dataset_base = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
 
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset   = Subset(val_dataset_base,   val_indices)
test_dataset  = datasets.ImageFolder(TEST_DIR, transform=train_transforms)
 
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
 
 
# Costruzione del modello
 
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model
 
model = build_model(NUM_CLASS).to(DEVICE)
 
# Gestore di loss
criterion = nn.CrossEntropyLoss()
 
#Gestore pesi della rete
optimizer = optim.Adam(model.parameters(), lr=LR)
#Gestore Learning Rate
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH)
 
# Funzione di Training
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
 
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
 
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
 
    return total_loss / total, correct / total
 
# Funzione di Valutation
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
 
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
 
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
 
    return total_loss / total, correct / total
 
#Gestore Predict
def get_predictions(model, loader, device):

    model.eval()
    all_preds, all_labels = [], []
 
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
 
    return np.array(all_labels), np.array(all_preds)
 
# Traing Loop 
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0
 

print("Training loop\n")

 

 
for epoch in range(1, NUM_EPOCH + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step()
 
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
 
    # Salva il miglior modello
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
 
    print(f"Epoch [{epoch:2d}/{NUM_EPOCH}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
          + (" ← best" if val_acc == best_val_acc else ""))
 
# Valutation utilizzando il miglior modello dopo il finetuning e stampa dei risultati
print("VALUTAZIONE SUL TEST SET (miglior modello)")

 
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
y_true, y_pred = get_predictions(model, test_loader, DEVICE)
 
acc       = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall    = recall_score(y_true, y_pred, average="macro")
f1        = f1_score(y_true, y_pred, average="macro")
 
print(f"\nAccuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}  (macro)")
print(f"Recall   : {recall:.4f}  (macro)")
print(f"F1-Score : {f1:.4f}  (macro)")
print(f"\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
 
# Creazione grafici 
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
epochs_range = range(1, NUM_EPOCH + 1)
 
# — Loss curve
axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss",   marker="o")
axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(True)
 
# — Accuracy curve
axes[1].plot(epochs_range, history["train_acc"], label="Train Acc", marker="o")
axes[1].plot(epochs_range, history["val_acc"],   label="Val Acc",   marker="o")
axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(True)
 

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    ax=axes[2]
)
axes[2].set_title("Confusion Matrix"); axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
axes[2].tick_params(axis="x", rotation=45)
 
plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGrafici salvati in results.png")
 