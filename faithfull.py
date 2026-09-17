import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
import sys


#Configurazione
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH       = 32
LR          = 1e-4
VAL_SPLIT   = 0.2
NUM_CLASS   = 6
PATIENCE    = 10
MIN_DELTA   = 1e-4
MAX_EPOCHS  = 100 

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Dataset originali per calcolare la Baseline 0%
TRAIN_DIR = "seg_train/seg_train"
TEST_DIR  = "seg_test/seg_test"

# Metodi XAI e parametri ROAR
XAI_METHODS = [
    "shap_risultati",
    "lime_risultati",
    "Integrated_Gradients",
    "gradcam_results",
    "lrp_results"
]
ALPHAS = [10, 30, 50, 70, 90]
MASK_TYPES = ["xai", "random"]

RESULTS_FILE = "roar_benchmark_results.json"
PLOTS_DIR = "roar_plots"

data_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])



#FUNZIONI DI TRAINING 
def build_fresh_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model

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

def run_training_cycle(train_loader, val_loader, test_loader, model_name):
    model = build_fresh_model(NUM_CLASS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    
    best_val_acc = 0.0
    patience_counter = 0
    temp_model_path = f"temp_{model_name}.pth"
    
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc = val_acc
            torch.save(model.state_dict(), temp_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"      Early stop all'epoca {epoch}. Best Val: {best_val_acc:.4f}")
            break
            
    if os.path.exists(temp_model_path):
        model.load_state_dict(torch.load(temp_model_path, map_location=DEVICE))
        
    _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
        
    return float(test_acc)


# Grafici ROAR
def generate_plots(json_file):
    print("\n=== GENERAZIONE GRAFICI ROAR ===")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    METHOD_NAMES_MAP = {
        "shap_risultati": "SHAP",
        "lime_risultati": "LIME",
        "Integrated_Gradients": "Integrated Gradients",
        "gradcam_results": "GradCAM",
        "lrp_results": "LRP"
    }

    with open(json_file, "r") as f:
        data = json.load(f)

    baseline_acc = data.get("baseline_0", 0.0)
    X_AXIS = [0] + ALPHAS  # [0, 10, 30, 50, 70, 90]

    xai_curves = {}
    random_curves = []

    for raw_method, method_data in data.items():
        if raw_method == "baseline_0":
            continue
            
        pretty_name = METHOD_NAMES_MAP.get(raw_method, raw_method)
        curve_xai = [baseline_acc]
        curve_rnd = [baseline_acc]
        
        for a in ALPHAS:
            a_str = str(a)
            curve_xai.append(method_data["xai"].get(a_str, 0))
            curve_rnd.append(method_data["random"].get(a_str, 0))
            
        xai_curves[pretty_name] = curve_xai
        random_curves.append(curve_rnd)

    if not xai_curves:
        print("[!] Nessun dato trovato per tracciare i grafici.")
        return

    mean_random_curve = np.mean(random_curves, axis=0)

    # Calcolo AUC
    auc_scores = {}
    for method, curve in xai_curves.items():
        auc_scores[method] = np.trapezoid(curve, x=X_AXIS)
    auc_random = np.trapezoid(mean_random_curve, x=X_AXIS)

    
    plt.figure(figsize=(10, 7))
    plt.plot(X_AXIS, mean_random_curve, marker='s', linestyle='--', color='black', 
             linewidth=2, markersize=8, label=f'Random Baseline (AUC: {auc_random:.2f})')

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i, (method, curve) in enumerate(xai_curves.items()):
        plt.plot(X_AXIS, curve, marker='o', linewidth=2, color=colors[i % len(colors)], label=f'{method}')

    plt.title("ROAR: Degradazione dell'Accuratezza", fontsize=16, fontweight='bold')
    plt.xlabel("Percentuale di feature rimosse (%)", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.xticks(X_AXIS)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "1_roar_main_plot.png"), dpi=200)
    plt.close()

  
    sorted_auc = sorted(auc_scores.items(), key=lambda item: item[1])
    methods_sorted = [x[0] for x in sorted_auc]
    scores_sorted = [x[1] for x in sorted_auc]
    methods_sorted.append("Random Baseline")
    scores_sorted.append(auc_random)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods_sorted, scores_sorted, color=['tab:blue']*len(xai_curves) + ['black'])
    plt.title("Classifica Fedeltà (Metrica AUC)", fontsize=16, fontweight='bold')
    plt.ylabel("Area Under the Curve (Più bassa è meglio)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "2_roar_auc_ranking.png"), dpi=200)
    plt.close()

    
    num_methods = len(xai_curves)
    cols = 3
    rows = (num_methods + 1) // cols if (num_methods + 1) % cols != 0 else (num_methods + 1) // cols
    if num_methods == 5: rows = 2

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle("Analisi Individuale: Metodo XAI vs Random", fontsize=18, fontweight='bold')
    axes = axes.flatten()

    for i, (method, curve) in enumerate(xai_curves.items()):
        ax = axes[i]
        ax.plot(X_AXIS, curve, marker='o', linewidth=2.5, color=colors[i % len(colors)], label=method)
        specific_random = random_curves[i]
        ax.plot(X_AXIS, specific_random, marker='s', linestyle='--', color='gray', label='Random')
        
        ax.set_title(f"{method}\n(AUC: {auc_scores[method]:.1f} vs Rnd: {np.trapezoid(specific_random, X_AXIS):.1f})", fontweight='bold')
        ax.set_xlabel("% Rimosse")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(X_AXIS)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    fig.subplots_adjust(top=0.9) 
    plt.savefig(os.path.join(PLOTS_DIR, "3_roar_individual_subplots.png"), dpi=200)
    plt.close()
    print(f"I 3 grafici sono stati generati e salvati nella cartella '{PLOTS_DIR}'.")



def main():
    print(f"AVVIO RETRAINING E PLOTTING")
    print(f"Device utilizzato: {DEVICE}")
    print(f"==================================================\n")

    results_dict = {}

    # (0%)
    print(f">>Calcolo Originale (Alpha = 0%)")
    full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=data_transforms)
    test_dataset_orig  = datasets.ImageFolder(TEST_DIR, transform=data_transforms)
    
    all_indices = list(range(len(full_train_dataset)))
    all_labels  = [full_train_dataset.targets[i] for i in all_indices]
    train_indices, val_indices = train_test_split(all_indices, test_size=VAL_SPLIT, stratify=all_labels)
    
    train_loader_orig = DataLoader(Subset(full_train_dataset, train_indices), batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    val_loader_orig   = DataLoader(Subset(full_train_dataset, val_indices), batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    test_loader_orig  = DataLoader(test_dataset_orig, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    baseline_acc = run_training_cycle(train_loader_orig, val_loader_orig, test_loader_orig, "baseline_0")
    results_dict["baseline_0"] = baseline_acc
    print(f"   => Test Accuracy Originale (0%): {baseline_acc:.4f}\n")


    
    for method_dir in XAI_METHODS:
        roar_base = os.path.join(method_dir, "roar_datasets")
        
        if not os.path.exists(roar_base):
            print(f"[!] Directory {roar_base} non trovata. Salto...")
            continue
            
        print(f"\n>> Inizio analisi metodo: {method_dir.upper()}")
        results_dict[method_dir] = {}

        for mask_type in MASK_TYPES:
            results_dict[method_dir][mask_type] = {}
            
            for alpha in ALPHAS:
                train_dir = os.path.join(roar_base, mask_type, f"alpha_{alpha}", "train")
                test_dir  = os.path.join(roar_base, mask_type, f"alpha_{alpha}", "test")
                
                if not os.path.exists(train_dir) or not os.path.exists(test_dir):
                    print(f"   [!] Dati mancanti per {mask_type} alpha={alpha}%. Salto...")
                    continue

                print(f"   -- Addestramento {mask_type.upper()} | Alpha = {alpha}% --")
                
                full_train_mask = datasets.ImageFolder(train_dir, transform=data_transforms)
                test_mask       = datasets.ImageFolder(test_dir, transform=data_transforms)
                
                all_idx_mask = list(range(len(full_train_mask)))
                all_lbl_mask = [full_train_mask.targets[i] for i in all_idx_mask]
                
                try:
                    tr_idx_mask, val_idx_mask = train_test_split(all_idx_mask, test_size=VAL_SPLIT, stratify=all_lbl_mask)
                except ValueError:
                    print("     [Errore] Pochi dati, salto.")
                    continue
                
                tr_loader = DataLoader(Subset(full_train_mask, tr_idx_mask), batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
                vl_loader = DataLoader(Subset(full_train_mask, val_idx_mask), batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
                ts_loader = DataLoader(test_mask, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

                test_acc = run_training_cycle(tr_loader, vl_loader, ts_loader, f"{method_dir}_{mask_type}_{alpha}")
                
                print(f"      => Test Accuracy: {test_acc:.4f}\n")
                results_dict[method_dir][mask_type][alpha] = test_acc

    # Salvo il file JSON intermedio 
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"\nTraining completato! Dati salvati in {RESULTS_FILE}")

    
    generate_plots(RESULTS_FILE)

    
if __name__ == "__main__":
    generate_plots("roar_benchmark_results.json")