import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import shap
import warnings
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Configurazione e caricamento del modello (pesi di best model)
OUTPUT_DIR = "shap_risultati"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEST_DIR = "seg_test/seg_test"
DEVICE = torch.device("cpu")
NUM_CLASS = 6
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


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

print("Caricamento del modello")
model = build_model(NUM_CLASS).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()
print("Modello caricato con successo!\n")

# Caricamento del dataset
print("Caricamento del dataset di test...")
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
print(f" Dataset caricato: {len(test_dataset)} immagini\n")
print("Selezione immagini bilanciate da ogni classe per il test")
selected_indices = []
class_counts = {i: 0 for i in range(NUM_CLASS)}

for idx, (_, label) in enumerate(test_dataset):
    if class_counts[label] < 2:
        selected_indices.append(idx)
        class_counts[label] += 1
    if len(selected_indices) >= NUM_CLASS * 2:
        break

test_images = torch.stack([test_dataset[i][0] for i in selected_indices])
test_labels = torch.tensor([test_dataset[i][1] for i in selected_indices])

print(f"Selezionate {len(selected_indices)} immagini di test")
print(f"Shape immagini: {test_images.shape}")
print(f"Shape labels:   {test_labels.shape}\n")


print("Preparazione background SHAP")

BACKGROUND_SIZE = 100
bg_per_class    = BACKGROUND_SIZE // NUM_CLASS   # 16
bg_indices      = []
bg_class_counts = {i: 0 for i in range(NUM_CLASS)}

for idx, (_, label) in enumerate(test_dataset):
    # Salta le immagini già usate come test
    if idx in selected_indices:
        continue
    if bg_class_counts[label] < bg_per_class:
        bg_indices.append(idx)
        bg_class_counts[label] += 1
    if len(bg_indices) >= BACKGROUND_SIZE:
        break

background = torch.stack([test_dataset[i][0] for i in bg_indices])
print(f"Background preparato: {background.shape}")
print(f"Immagini per classe: { {CLASS_NAMES[k]: v for k,v in bg_class_counts.items()} }\n")

# Avvia di SHAP
print("Inizializzazione di SHAP")
explainer = shap.GradientExplainer(model, background)


print("Calcolo SHAP values")

shap_values_list = explainer.shap_values(test_images)
print(f"SHAP values calcolati!")
print(f"Numero di array (uno per classe): {len(shap_values_list)}")
print(f"Shape di ogni array: {shap_values_list[0].shape}\n")

# Iversione normalizzazione
def denormalize(tensor):
    
    t = tensor.clone()
    for i, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[i] = t[i] * s + m
    return t.clamp(0, 1)
#Per visualizzazione mappa dei pixel significativi 
def get_shap_2d(class_idx, img_idx):
    
    if isinstance(shap_values_list, list) and len(shap_values_list) == NUM_CLASS:
        # Struttura attesa: lista[class_idx] → (N, C, H, W)
        shap_chw = shap_values_list[class_idx][img_idx]   # (C, H, W)
    else:
        # Fallback: array unico, ignora class_idx
        arr = np.array(shap_values_list)
        if arr.ndim == 4:
            # (N, C, H, W)
            shap_chw = arr[img_idx]                        # (C, H, W)
        elif arr.ndim == 5:
            # (N, C, H, W, num_classes)
            shap_chw = arr[img_idx, :, :, :, class_idx]   # (C, H, W)
        else:
            raise ValueError(f"Struttura shap_values inattesa: {arr.shape}")

    return shap_chw.mean(axis=0)                           # (H, W)                       
#Normalizzazione mappa
def normalize_map(m):
    
    amax = np.abs(m).max()
    return m / amax if amax > 0 else m

#Analisi immagini
print("Generazione analisi dettagliata per ogni immagine")

for img_idx in range(len(test_images)):

    
    with torch.no_grad():
        output = model(test_images[img_idx:img_idx+1].to(DEVICE))
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()
    pred_class = int(np.argmax(probs))
    true_label = test_labels[img_idx].item()
    correct     = "✓" if pred_class == true_label else "✗"

    fig = plt.figure(figsize=(24, 14))
    gs  = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.35)

    fig.suptitle(
        f"Analisi SHAP — Immagine {img_idx}  {correct}\n"
        f"Vera: {CLASS_NAMES[true_label]}  |  "
        f"Predetta: {CLASS_NAMES[pred_class]} ({probs[pred_class]:.2%})",
        fontsize=16, fontweight='bold'
    )

    img_rgb = denormalize(test_images[img_idx]).permute(1, 2, 0).numpy()

    # [0,0] Immagine originale
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_rgb)
    ax.set_title('Immagine Originale', fontweight='bold', fontsize=12)
    ax.axis('off')

    # [0,1] Top-5 predizioni 
    ax = fig.add_subplot(gs[0, 1])
    top5 = np.argsort(probs)[-5:][::-1]
    txt  = "Top-5 Predizioni:\n" + "="*38 + "\n"
    for rank, idx in enumerate(top5, 1):
        bar = "█" * int(probs[idx]*34) + "░" * (34 - int(probs[idx]*34))
        txt += f"{rank}. {CLASS_NAMES[idx]:10s} {bar} {probs[idx]:6.2%}\n"
    ax.text(0.05, 0.95, txt, fontsize=9, family='monospace',
            va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.axis('off')
    ax.set_title('Top-5 Predizioni', fontweight='bold', fontsize=12)

    # [0,2] Heatmap SHAP classe predetta
    ax = fig.add_subplot(gs[0, 2])
    shap_pred_norm = normalize_map(get_shap_2d(pred_class, img_idx))
    im = ax.imshow(shap_pred_norm, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'SHAP: {CLASS_NAMES[pred_class]}\n(classe predetta)',
                 fontweight='bold', fontsize=12, color='green')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='SHAP')

    # [0,3] Overlay
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(img_rgb)
    shap_abs_norm = np.abs(shap_pred_norm)
    im = ax.imshow(shap_abs_norm, cmap='hot', alpha=0.55, vmin=0, vmax=1)
    ax.set_title('Overlay: Immagine + |SHAP|', fontweight='bold', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='|SHAP|')

    # Righe 1-3: SHAP per ogni classe
    for class_idx in range(NUM_CLASS):
        row = 1 + (class_idx // 4)
        col = class_idx % 4
        ax  = fig.add_subplot(gs[row, col])

        shap_norm = normalize_map(get_shap_2d(class_idx, img_idx))
        im = ax.imshow(shap_norm, cmap='RdBu_r', vmin=-1, vmax=1)

        title = f'{CLASS_NAMES[class_idx]}\n({probs[class_idx]:.2%})'
        color = 'green' if class_idx == pred_class else 'black'
        ax.set_title(title, fontweight='bold', fontsize=11, color=color)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='SHAP', fraction=0.046, pad=0.04)

    fname = os.path.join(OUTPUT_DIR, f'shap_analisi_immagine_{img_idx:02d}_{CLASS_NAMES[pred_class]}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Salvato: {fname}")

print()

# Media |SHAP| per classe
print("Generazione mappa media SHAP per classe...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    'Media |SHAP| per Classe\n'
    '(Quali zone spaziali influenzano generalmente la predizione)',
    fontsize=16, fontweight='bold'
)

for class_idx in range(NUM_CLASS):
    ax = axes[class_idx // 3, class_idx % 3]

    
    # shap_values_list[class_idx] → (N, C, H, W)
    mean_shap = np.abs(shap_values_list[class_idx]).mean(axis=(0, 1))  # (H, W)

    im = ax.imshow(mean_shap, cmap='hot')
    ax.set_title(f'{CLASS_NAMES[class_idx]}', fontweight='bold', fontsize=13)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Media |SHAP|')

plt.tight_layout()
plt.savefig('shap_media_per_classe.png', dpi=150, bbox_inches='tight')
plt.close()
print("Salvato: shap_media_per_classe.png\n")

# Analisi statistica
print("=" * 80)
print("STATISTICHE SHAP PER IMMAGINE")
print("=" * 80)

for img_idx in range(len(test_images)):
    true_label = test_labels[img_idx].item()
    with torch.no_grad():
        output = model(test_images[img_idx:img_idx+1].to(DEVICE))
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()
    pred_class = int(np.argmax(probs))
    correct = "✓" if pred_class == true_label else "✗"

    print(f"\n[Img {img_idx:02d}] {correct} {CLASS_NAMES[true_label]} → {CLASS_NAMES[pred_class]} "
          f"(conf: {probs[pred_class]:.2%})")

    shap_2d = np.abs(get_shap_2d(pred_class, img_idx))
    print(f"  |SHAP| max:  {shap_2d.max():.6f}")
    print(f"  |SHAP| mean: {shap_2d.mean():.6f}")
    print(f"  |SHAP| std:  {shap_2d.std():.6f}")

    flat = shap_2d.flatten()
    top3 = np.argsort(flat)[-3:][::-1]
    for rank, flat_idx in enumerate(top3, 1):
        r, c = np.unravel_index(flat_idx, shap_2d.shape)
        print(f"    Top-{rank} pixel: ({r:3d},{c:3d}) = {flat[flat_idx]:.6f}")

print("\n" + "=" * 80)
print("STATISTICHE SHAP PER CLASSE")
print("=" * 80)


#Log .txt
OUTPUT_DIR = "shap_risultati"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, "w", encoding="utf-8")
        self._write_header()

    def _write_header(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.write("=" * 80 + "\n")
        self.log.write(f"  ANALISI SHAP — LOG COMPLETO\n")
        self.log.write(f"  Data e ora: {now}\n")
        self.log.write("=" * 80 + "\n\n")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.write("\n" + "=" * 80 + "\n")
        self.log.write("  FINE LOG\n")
        self.log.write("=" * 80 + "\n")
        self.log.close()

log_path   = os.path.join(OUTPUT_DIR, "shap_log.txt")
logger     = Logger(log_path)
sys.stdout = logger

for class_idx in range(NUM_CLASS):
    arr = np.abs(shap_values_list[class_idx])   # (N, C, H, W)
    mean_map = arr.mean(axis=(0, 1))             # (H, W)
    print(f"\n{CLASS_NAMES[class_idx]}:")
    print(f"  Media |SHAP|: {mean_map.mean():.6f}")
    print(f"  Max   |SHAP|: {mean_map.max():.6f}")
    print(f"  Std   |SHAP|: {mean_map.std():.6f}")



logger.close()
sys.stdout = logger.terminal