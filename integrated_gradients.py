import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


TEST_DIR = "seg_test/seg_test"
DEVICE = torch.device("cpu")
NUM_CLASS = 6
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_SAMPLES = 5  # Numero di immagini da analizzare per classe

# Trasformazioni per normalizzazione immagini
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Caricamento modello
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
print("✓ Modello caricato con successo!\n")

# Caricamento dataset
print("Caricamento del dataset di test...")
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
print(f"✓ Dataset caricato: {len(test_dataset)} immagini\n")

# Funzione Integrated Gradients

def integrated_gradients(model, input_img, target_class, steps=50):
  
    baseline = torch.zeros_like(input_img)
    
    
    accumulated_grads = torch.zeros_like(input_img)
    
    for step in range(steps):
        # Alpha: da 0 a 1
        alpha = step / steps
        
       
        interpolated = baseline + alpha * (input_img - baseline)
        interpolated = interpolated.clone().detach().requires_grad_(True)
        
        output = model(interpolated)
        
        # Estrai il logit della classe target
        target_logit = output[0, target_class]
        
        
        model.zero_grad()
        target_logit.backward()
        
        # Accumula i gradienti
        if interpolated.grad is not None:
            accumulated_grads += interpolated.grad.data
    
    # Media dei gradienti
    integrated_grads = (input_img - baseline) * (accumulated_grads / steps)
    
    return integrated_grads.detach()



# Seleziona alcune immagini di test 
selected_indices = []
class_counts = {i: 0 for i in range(NUM_CLASS)}

# Seleziona immagini bilanciate da ogni classe
for idx, (img, label) in enumerate(test_dataset):
    if class_counts[label] < 2:
        selected_indices.append(idx)
        class_counts[label] += 1
    if len(selected_indices) >= 12:
        break

print(f"Immagini selezionate per analisi: {len(selected_indices)}")


# Dictionary per memorizzare i risultati
gradients_dict = {}

for idx_num, idx in enumerate(selected_indices):
    test_img, true_label = test_dataset[idx]
    test_img_batch = test_img.unsqueeze(0).to(DEVICE)
    
    # Predizione
    with torch.no_grad():
        output = model(test_img_batch)
        probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
        pred_class = np.argmax(probs)
    
    # Calcola integrated gradients
    ig = integrated_gradients(model, test_img_batch, pred_class, steps=30)
    ig_cpu = ig.squeeze(0).cpu().detach()
    
    gradients_dict[idx] = {
        'image': test_img,
        'true_label': true_label,
        'pred_class': pred_class,
        'pred_probs': probs,
        'integrated_grads': ig_cpu
    }
    
    print(f"   [{idx_num+1}/{len(selected_indices)}] Immagine processata - "
          f"Vera: {CLASS_NAMES[true_label]}, Predetta: {CLASS_NAMES[pred_class]} "
          f"({probs[pred_class]:.2%})")

print("\n✓ Integrated Gradients calcolati per tutte le immagini\n")

# Generazione grafici
print("Generazione visualizzazioni\n")

# Denormalizza le immagini per la visualizzazione
def denormalize(tensor):
    
    tensor = tensor.clone()
    for i, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        tensor[i] = tensor[i] * std + mean
    return tensor.clamp(0, 1)



for plot_idx, (dict_idx, data) in enumerate(gradients_dict.items()):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    pred_conf = data['pred_probs'][data['pred_class']]
    fig.suptitle(
        f"Analisi Integrated Gradients - {data['image'].shape}\n"
        f"Vera: {CLASS_NAMES[data['true_label']]} | "
        f"Predetta: {CLASS_NAMES[data['pred_class']]} ({pred_conf:.2%})",
        fontsize=14, fontweight='bold'
    )
    
    # 1 Immagine originale
    img_denorm = denormalize(data['image'])
    axes[0, 0].imshow(img_denorm.permute(1, 2, 0).numpy())
    axes[0, 0].set_title('Immagine Originale', fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2 Media
    ig_mean = data['integrated_grads'].mean(dim=0).numpy()
    im1 = axes[0, 1].imshow(ig_mean, cmap='RdBu_r', 
                            vmin=-np.abs(ig_mean).max(), vmax=np.abs(ig_mean).max())
    axes[0, 1].set_title('Integrated Gradients (Media canali)\nRosso=Aumenta confidenza, Blu=Diminuisce', 
                         fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 3 Valore assoluto 
    ig_abs = np.abs(ig_mean)
    im2 = axes[0, 2].imshow(ig_abs, cmap='hot')
    axes[0, 2].set_title('Importanza Pixel\n(Valore Assoluto)', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # 4  Top 5 predizioni
    top5_idx = np.argsort(data['pred_probs'])[-5:][::-1]
    pred_text = "Top-5 Predizioni:\n" + "="*40 + "\n"
    for rank, idx in enumerate(top5_idx, 1):
        conf = data['pred_probs'][idx]
        bar_len = int(conf * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        pred_text += f"{rank}. {CLASS_NAMES[idx]:12s} {bar} {conf:6.2%}\n"
    
    axes[1, 0].text(0.05, 0.95, pred_text, fontsize=10, family='monospace',
                   verticalalignment='top', transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 0].axis('off')
    
    # 5 Immagine + Heatmap Integrated Gradients
    img_rgb = img_denorm.permute(1, 2, 0).numpy()
    axes[1, 1].imshow(img_rgb, alpha=0.6)
    im3 = axes[1, 1].imshow(ig_abs, cmap='hot', alpha=0.4)
    axes[1, 1].set_title('Overlay: Immagine + Importanza Pixel', fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    # 6. Visualizzazione canali separati
    
    channels_data = data['integrated_grads'].numpy()
    channel_names = ['Rosso', 'Verde', 'Blu']
    channel_max = max(np.abs(channels_data[i]).max() for i in range(3))
    
    ig_r = channels_data[0]
    im4 = axes[1, 2].imshow(ig_r, cmap='RdBu_r', vmin=-channel_max, vmax=channel_max)
    axes[1, 2].set_title(f'Gradiente Canale {channel_names[0]}', fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im4, ax=axes[1, 2])
    
    plt.tight_layout()
    filename = f'integrated_gradients_{plot_idx:02d}_{CLASS_NAMES[data["pred_class"]]}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Salvato: {filename}")

print("\n")

#  Mappa di importanza media per classe
print("Generazione mappe di importanza media per classe...")
class_importance_maps = {i: [] for i in range(NUM_CLASS)}

# Calcola le mappe di importanza per classe
for data in gradients_dict.values():
    pred_class = data['pred_class']
    ig_abs = torch.abs(data['integrated_grads']).mean(dim=0).numpy()
    class_importance_maps[pred_class].append(ig_abs)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Media Integrated Gradients per Classe (Quali pixel sono generalmente importanti)', 
             fontsize=16, fontweight='bold')

for class_idx in range(NUM_CLASS):
    ax = axes[class_idx // 3, class_idx % 3]
    
    if class_importance_maps[class_idx]:
        mean_importance = np.mean(class_importance_maps[class_idx], axis=0)
        im = ax.imshow(mean_importance, cmap='hot')
        ax.set_title(f'{CLASS_NAMES[class_idx]}\n(n={len(class_importance_maps[class_idx])})', 
                    fontweight='bold')
        plt.colorbar(im, ax=ax, label='|Gradiente|')
    else:
        ax.text(0.5, 0.5, f'Nessuna immagine\nper {CLASS_NAMES[class_idx]}',
               ha='center', va='center', fontsize=12)
    
    ax.axis('off')

plt.tight_layout()
plt.savefig('integrated_gradients_per_classe.png', dpi=150, bbox_inches='tight')
plt.close()
print("Salvato: integrated_gradients_per_classe.png\n")

# Analisi statistiche
print("Analisi Statistiche:")
print("=" * 80)

# Statistiche per immagine
print("\nStatistiche per ogni immagine analizzata:")
print("-" * 80)
for plot_idx, (dict_idx, data) in enumerate(gradients_dict.items()):
    ig_abs = torch.abs(data['integrated_grads']).mean(dim=0).numpy()
    
    print(f"\n[{plot_idx+1}] {CLASS_NAMES[data['true_label']]} → {CLASS_NAMES[data['pred_class']]}")
    print(f"    Confidenza predizione: {data['pred_probs'][data['pred_class']]:6.2%}")
    print(f"    Max importanza pixel: {ig_abs.max():.6f}")
    print(f"    Media importanza pixel: {ig_abs.mean():.6f}")
    print(f"    Min importanza pixel: {ig_abs.min():.6f}")
    
    # Top 3 pixel più importanti
    flat_idx = np.argsort(ig_abs.flatten())[-3:][::-1]
    print(f"    Top-3 pixel più importanti:")
    for rank, idx in enumerate(flat_idx, 1):
        row, col = np.unravel_index(idx, ig_abs.shape)
        print(f"      {rank}. Posizione ({row:3d}, {col:3d}): {ig_abs[row, col]:.6f}")

print("\n" + "=" * 80)
print("\nStatistiche per classe:")
print("-" * 80)
for class_idx in range(NUM_CLASS):
    if class_importance_maps[class_idx]:
        all_maps = np.array(class_importance_maps[class_idx])
        mean_map = all_maps.mean(axis=0)
        std_map = all_maps.std(axis=0)
        
        print(f"\n{CLASS_NAMES[class_idx]} ({len(all_maps)} immagini):")
        print(f"  Media importanza pixel: {mean_map.mean():.6f}")
        print(f"  Std dev importanza pixel: {std_map.mean():.6f}")
        print(f"  Max importanza pixel (media): {mean_map.max():.6f}")
        print(f"  Min importanza pixel (media): {mean_map.min():.6f}")
        
        # Top-5 regioni più importanti
        flat_idx = np.argsort(mean_map.flatten())[-5:][::-1]
        print(f"  Top-5 regioni importanti:")
        for rank, idx in enumerate(flat_idx, 1):
            row, col = np.unravel_index(idx, mean_map.shape)
            print(f"    {rank}. Posizione ({row:3d}, {col:3d}): {mean_map[row, col]:.6f}")

