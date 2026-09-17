import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image
import warnings
import sys
from datetime import datetime
from lime import lime_image
from skimage.segmentation import mark_boundaries
warnings.filterwarnings('ignore')



# Configurazione Modello
OUTPUT_DIR = "lime_risultati"
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

# Trasformazione separata SENZA normalizzazione
raw_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),  
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
#Per il salvataggio dei risultati
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, "w", encoding="utf-8")
        self._write_header()

    def _write_header(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.write("=" * 80 + "\n")
        self.log.write(f"  ANALISI LIME — LOG COMPLETO\n")
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

log_path   = os.path.join(OUTPUT_DIR, "lime_log.txt")
logger     = Logger(log_path)
sys.stdout = logger

print("Caricamento del modello")
model = build_model(NUM_CLASS).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()
print("Modello caricato con successo!\n")

# Caricamento Dataset
print("Caricamento dataset di test")
test_dataset     = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
test_dataset_raw = datasets.ImageFolder(TEST_DIR, transform=raw_transforms)
print(f"Dataset caricato: {len(test_dataset)} immagini\n")


print("Selezione immagini bilanciate")
selected_indices = []
class_counts = {i: 0 for i in range(NUM_CLASS)}

for idx, (_, label) in enumerate(test_dataset):
    if class_counts[label] < 2:
        selected_indices.append(idx)
        class_counts[label] += 1
    if len(selected_indices) >= NUM_CLASS * 2:
        break

print(f"Selezionate {len(selected_indices)} immagini\n")

#Inizializzazione LIME

normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

def predict_fn(images_np):
    """
    Input:  numpy array (N, H, W, C) in [0, 1]  ← formato LIME
    Output: numpy array (N, num_classes)          ← probabilità
    """
    batch = []
    for img in images_np:
        # (H,W,C) float → (C,H,W) tensor → normalizza
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        t = normalize(t)
        batch.append(t)

    batch = torch.stack(batch).to(DEVICE)

    with torch.no_grad():
        output = model(batch)
        probs  = torch.softmax(output, dim=1).cpu().numpy()

    return probs
explainer = lime_image.LimeImageExplainer()


print("Calcolo LIME")
lime_results = {}

for idx_num, idx in enumerate(selected_indices):
    img_normalized, true_label = test_dataset[idx]
    img_raw, _                 = test_dataset_raw[idx]

   
    img_np = img_raw.permute(1, 2, 0).numpy()   

    # Predizione
    with torch.no_grad():
        output = model(img_normalized.unsqueeze(0).to(DEVICE))
        probs  = torch.softmax(output, dim=1)[0].cpu().numpy()
    pred_class = int(np.argmax(probs))

    
    explanation = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=NUM_CLASS,       # spiega tutte le classi
        hide_color=0,               # colore dei regioni nascoste (nero)
        num_samples=1000,           # perturbazioni generate
        num_features=20,            # regioni considerati
    )

    lime_results[idx] = {
        'img_np':      img_np,
        'true_label':  true_label,
        'pred_class':  pred_class,
        'probs':       probs,
        'explanation': explanation,
    }

    print(f"  [{idx_num+1:2d}/{len(selected_indices)}] "
          f"Vera: {CLASS_NAMES[true_label]:10s} | "
          f"Predetta: {CLASS_NAMES[pred_class]:10s} ({probs[pred_class]:.2%})")



#Wrapper LIME 
def get_lime_mask(explanation, class_idx, img_np,
                  positive_only=True, num_features=10, hide_rest=False):
  
    temp, mask = explanation.get_image_and_mask(
        class_idx,
        positive_only=positive_only,
        num_features=num_features,
        hide_rest=hide_rest,
    )
    return temp, mask

#
print("Generazione analisi per immagine")

for plot_idx, (dict_idx, data) in enumerate(lime_results.items()):
    img_np     = data['img_np']
    pred_class = data['pred_class']
    true_label = data['true_label']
    probs      = data['probs']
    expl       = data['explanation']
    correct    = "✓" if pred_class == true_label else "✗"

    fig = plt.figure(figsize=(24, 16))
    gs  = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

    fig.suptitle(
        f"Analisi LIME — Immagine {plot_idx}  {correct}\n"
        f"Vera: {CLASS_NAMES[true_label]}  |  "
        f"Predetta: {CLASS_NAMES[pred_class]} ({probs[pred_class]:.2%})",
        fontsize=16, fontweight='bold'
    )

    #  [0,0] Immagine originale 
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_np)
    ax.set_title('Immagine Originale', fontweight='bold', fontsize=12)
    ax.axis('off')

    #  [0,1] Top 5 predizioni 
    ax = fig.add_subplot(gs[0, 1])
    top5 = np.argsort(probs)[-5:][::-1]
    txt  = "Top-5 Predizioni:\n" + "="*38 + "\n"
    for rank, i in enumerate(top5, 1):
        bar = "█" * int(probs[i]*34) + "░" * (34 - int(probs[i]*34))
        txt += f"{rank}. {CLASS_NAMES[i]:10s} {bar} {probs[i]:6.2%}\n"
    ax.text(0.05, 0.95, txt, fontsize=9, family='monospace',
            va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.axis('off')
    ax.set_title('Top 5 Predizioni', fontweight='bold', fontsize=12)

    # [0,2] Regioni positive (classe predetta)
    ax = fig.add_subplot(gs[0, 2])
    temp, mask = get_lime_mask(expl, pred_class, img_np,
                               positive_only=True, num_features=10)
    ax.imshow(mark_boundaries(temp, mask))
    ax.set_title(f'Regioni POSITIVE\n({CLASS_NAMES[pred_class]})',
                 fontweight='bold', fontsize=12, color='green')
    ax.axis('off')

    # [0,3] Regioni negative (classe predetta) 
    ax = fig.add_subplot(gs[0, 3])
    temp, mask = get_lime_mask(expl, pred_class, img_np,
                               positive_only=False, num_features=10)
    ax.imshow(mark_boundaries(temp, mask))
    ax.set_title(f'Regioni Totali\n({CLASS_NAMES[pred_class]})',
                 fontweight='bold', fontsize=12, color='darkorange')
    ax.axis('off')

    # [1,0] Solo regioni importanti (hide_rest=True) 
    ax = fig.add_subplot(gs[1, 0])
    temp, mask = get_lime_mask(expl, pred_class, img_np,
                               positive_only=True, num_features=10,
                               hide_rest=True)
    ax.imshow(mark_boundaries(temp, mask))
    ax.set_title('Solo regioni importanti\n',
                 fontweight='bold', fontsize=12)
    ax.axis('off')

    # [1,1] Heatmap pesi 
    ax = fig.add_subplot(gs[1, 1])
    # Mappa continua dei pesi LIME su ogni pixel
    segments   = expl.segments                          # (H, W) int
    local_exp  = expl.local_exp[pred_class]             # [(seg_id, weight), ...]
    weight_map = np.zeros(segments.shape, dtype=float)
    for seg_id, weight in local_exp:
        weight_map[segments == seg_id] = weight

    amax = np.abs(weight_map).max()
    if amax > 0:
        weight_map_norm = weight_map / amax
    else:
        weight_map_norm = weight_map

    im = ax.imshow(weight_map_norm, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Heatmap pesi LIME\n',
                 fontweight='bold', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Peso normalizzato')

    # Overlay immagine + heatmap
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(img_np)
    im = ax.imshow(np.abs(weight_map_norm), cmap='hot', alpha=0.55, vmin=0, vmax=1)
    ax.set_title('Overlay: Immagine + |Pesi|',
                 fontweight='bold', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='|Peso|')

    # [1,3] Mappa segmenti 
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(mark_boundaries(img_np, segments))
    ax.set_title(f'Superpixel generati\n({segments.max()+1} regioni)',
                 fontweight='bold', fontsize=12)
    ax.axis('off')

    # LIME per tutte e 6 le classi 
    for class_idx in range(NUM_CLASS):
        row = 2
        col = class_idx % 4
        if class_idx == 4:
            row, col = 2, 4   
        if class_idx >= 4:
            continue           # mostra solo le prime 4 per spazio

        ax = fig.add_subplot(gs[2, class_idx])
        temp, mask = get_lime_mask(expl, class_idx, img_np,
                                   positive_only=True, num_features=8)
        color_title = 'green' if class_idx == pred_class else 'black'
        ax.imshow(mark_boundaries(temp, mask))
        ax.set_title(f'{CLASS_NAMES[class_idx]}\n({probs[class_idx]:.2%})',
                     fontweight='bold', fontsize=11, color=color_title)
        ax.axis('off')

    fname = os.path.join(OUTPUT_DIR, f'lime_analisi_{plot_idx:02d}_{CLASS_NAMES[pred_class]}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Salvato: {fname}")

print()

# Media pesi per classe 
print("Generazione mappa media pesi LIME per classe")

class_weight_maps = {i: [] for i in range(NUM_CLASS)}

for data in lime_results.values():
    expl     = data['explanation']
    segments = expl.segments

    for class_idx in range(NUM_CLASS):
        local_exp  = expl.local_exp.get(class_idx, [])
        weight_map = np.zeros(segments.shape, dtype=float)
        for seg_id, weight in local_exp:
            weight_map[segments == seg_id] = weight
        class_weight_maps[class_idx].append(np.abs(weight_map))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Media |Pesi LIME| per Classe\n'
             '(Quali zone spaziali influenzano la predizione)',
             fontsize=16, fontweight='bold')

for class_idx in range(NUM_CLASS):
    ax       = axes[class_idx // 3, class_idx % 3]
    mean_map = np.mean(class_weight_maps[class_idx], axis=0)
    im       = ax.imshow(mean_map, cmap='hot')
    ax.set_title(CLASS_NAMES[class_idx], fontweight='bold', fontsize=13)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Media |Peso LIME|')

plt.tight_layout()
plt.savefig('lime_media_per_classe.png', dpi=150, bbox_inches='tight')
plt.close()
print("Salvato: lime_media_per_classe.png\n")

# ANALISI STATISTICA
print("=" * 80)
print("STATISTICHE LIME PER IMMAGINE")
print("=" * 80)

for plot_idx, (dict_idx, data) in enumerate(lime_results.items()):
    expl       = data['explanation']
    pred_class = data['pred_class']
    true_label = data['true_label']
    probs      = data['probs']
    correct    = "✓" if pred_class == true_label else "✗"

    local_exp = expl.local_exp[pred_class]
    weights   = np.array([w for _, w in local_exp])

    print(f"\n[Img {plot_idx:02d}] {correct} "
          f"{CLASS_NAMES[true_label]} → {CLASS_NAMES[pred_class]} "
          f"(conf: {probs[pred_class]:.2%})")
    print(f"  N° superpixel:      {len(local_exp)}")
    print(f"  Peso max  (pos):    {weights.max():.6f}")
    print(f"  Peso min  (neg):    {weights.min():.6f}")
    print(f"  Peso medio |w|:     {np.abs(weights).mean():.6f}")

    top3 = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:3]
    print(f"  Top-3 superpixel:")
    for rank, (seg_id, w) in enumerate(top3, 1):
        direction = "favorisce" if w > 0 else "contrasta"
        print(f"    {rank}. Segmento {seg_id:3d}: peso={w:+.6f} ({direction})")

print("\n" + "=" * 80)
print("STATISTICHE LIME PER CLASSE")
print("=" * 80)

for class_idx in range(NUM_CLASS):
    maps     = class_weight_maps[class_idx]
    mean_map = np.mean(maps, axis=0)
    print(f"\n{CLASS_NAMES[class_idx]}:")
    print(f"  Media |peso|: {mean_map.mean():.6f}")
    print(f"  Max   |peso|: {mean_map.max():.6f}")
    print(f"  Std   |peso|: {mean_map.std():.6f}")

logger.close()
sys.stdout = logger.terminal
print(f"\nLog salvato in: {log_path}")