import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import shap
import warnings
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Configurazione e caricamento del modello (pesi di best model)
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

def denormalize(tensor):
    t = tensor.clone()
    for i, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        t[i] = t[i] * s + m
    return t.clamp(0, 1)


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
bg_per_class    = BACKGROUND_SIZE // NUM_CLASS
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

# Avvio di SHAP
print("Inizializzazione SHAP")

def f(X):
    X_tensor = torch.from_numpy(X).permute(0, 3, 1, 2).float()
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    X_tensor  = torch.stack([normalize(x) for x in X_tensor])
    with torch.no_grad():
        out = model(X_tensor.to(DEVICE))
        return torch.softmax(out, dim=1).cpu().numpy()

sample_img = denormalize(test_images[0]).permute(1, 2, 0).numpy()
masker     = shap.maskers.Image("inpaint_telea", sample_img.shape)
explainer  = shap.Explainer(f, masker, output_names=CLASS_NAMES)

test_images_np = np.stack([
    denormalize(test_images[i]).permute(1, 2, 0).numpy()
    for i in range(len(test_images))
])

print("Calcolo SHAP value")
shap_values = explainer(
    test_images_np,
    max_evals=500,
    batch_size=50,
    outputs=shap.Explanation.argsort.flip[:NUM_CLASS]
)
print(f"SHAP values calcolati!\n")

#Per visualizzazione mappa dei pixel significativi
print("Generazione plot SHAP")
fname = os.path.join(OUTPUT_DIR, 'shap_image_plot.png')
shap.image_plot(shap_values, show=False)
plt.savefig(fname, dpi=150, bbox_inches='tight')
plt.close()
print(f"Salvato: {fname}\n")

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
    correct    = "✓" if pred_class == true_label else "✗"

    print(f"\n[Img {img_idx:02d}] {correct} {CLASS_NAMES[true_label]} → "
          f"{CLASS_NAMES[pred_class]} (conf: {probs[pred_class]:.2%})")

    # shap_values.values ha shape (N, H, W, C, num_classes_output)
    shap_img = shap_values.values[img_idx]        # (H, W, C, k)
    shap_2d  = np.abs(shap_img).mean(axis=(2, 3)) # (H, W)

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

for class_idx in range(NUM_CLASS):
    # shap_values.values: (N, H, W, C, k)
    arr      = np.abs(shap_values.values[:, :, :, :, class_idx])  # (N, H, W, C)
    mean_map = arr.mean(axis=(0, 3))                                # (H, W)
    print(f"\n{CLASS_NAMES[class_idx]}:")
    print(f"  Media |SHAP|: {mean_map.mean():.6f}")
    print(f"  Max   |SHAP|: {mean_map.max():.6f}")
    print(f"  Std   |SHAP|: {mean_map.std():.6f}")

#Log .txt
logger.close()
sys.stdout = logger.terminal
print(f"\nLog salvato in: {log_path}")