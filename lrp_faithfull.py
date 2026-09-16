import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image # Aggiunto per il salvataggio ROAR

#IMPORTO ZENNIT PER APPLICARE LRP, VISTO CHE é OTTIMIZZATO PER LA RESNET18
from zennit.composites import EpsilonPlusFlat
from zennit.attribution import Gradient
from captum.attr import visualization as viz
from zennit.torchvision import ResNetCanonizer

#SETUP MODELLO
DEVICE = torch.device("cpu") 
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASS = len(CLASS_NAMES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TEST_DIR = "seg_test/seg_test"
BATCH = 32
MAX_IMAGES_TO_PROCESS = 500 

OUTPUT_DIR = "lrp_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inizializzazione cartelle ROAR
ROAR_BASE_DIR = os.path.join(OUTPUT_DIR, "roar_datasets")
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
GRAY_VALUE = 0.5

for alpha in ALPHAS:
    alpha_pct = int(alpha * 100)
    for method in ["xai", "random"]:
        for c_name in CLASS_NAMES:
            os.makedirs(os.path.join(ROAR_BASE_DIR, method, f"alpha_{alpha_pct}", c_name), exist_ok=True)


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model

model = build_model(NUM_CLASS).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval() 

# TRASFORMAZIONE DATI
test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(DEVICE)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(DEVICE)
    return tensor * std + mean

#LOOP DI SERVIZIO
print(f"LRP con Zennit per {MAX_IMAGES_TO_PROCESS} immagini")
processed_count = 0

for images, labels in test_loader:
    if processed_count >= MAX_IMAGES_TO_PROCESS:
        break

    images = images.to(DEVICE)
    images.requires_grad = True
    labels = labels.to(DEVICE)

    #PREDIZIONE
    outputs = model(images)
    preds = outputs.argmax(dim=1)

    
    target_labels = F.one_hot(preds, num_classes=NUM_CLASS).float().to(DEVICE)

    canonizer = ResNetCanonizer()

    
    composite = EpsilonPlusFlat(canonizers=[canonizer])


    with Gradient(model, composite) as attributor:
        _, attributions = attributor(images, target_labels)
        
    

    #SALVATAGGIO RISULTATI
    for i in range(images.size(0)):
        if processed_count >= MAX_IMAGES_TO_PROCESS:
            break

        true_class = CLASS_NAMES[labels[i].item()]
        pred_class = CLASS_NAMES[preds[i].item()]

        orig_img = denormalize(images[i].detach()).cpu().numpy()
        orig_img = np.transpose(orig_img, (1, 2, 0))
        orig_img = np.clip(orig_img, 0, 1)

        attr = attributions[i].squeeze().cpu().detach().numpy()
        attr = np.transpose(attr, (1, 2, 0))

        fig, ax = viz.visualize_image_attr_multiple(
            attr,
            orig_img,
            methods=["original_image", "heat_map"],
            signs=["all", "positive"],
            titles=[f"Originale ({true_class})", f"LRP Zennit ({pred_class})"],
            show_colorbar=True,
            fig_size=(8, 4),
            use_pyplot=False 
        )

        status = "CORRETTA" if true_class == pred_class else "ERRATA"
        filename = f"img_{processed_count:03d}_{status}_T-{true_class}_P-{pred_class}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        fig.savefig(filepath, bbox_inches='tight')
        plt.close(fig) 

        
        # Faithfull Evalutation
        
        H, W, C = orig_img.shape
        num_pixels = H * W
        
        # L'attribuzione LRP è sui 3 canali, ne prendiamo la media assoluta
        lrp_2d = np.abs(attr).mean(axis=2)
        flat_lrp = lrp_2d.flatten()

        for alpha in ALPHAS:
            alpha_pct = int(alpha * 100)
            k = int(num_pixels * alpha)
            
            if k == 0:
                continue
                
            xai_path = os.path.join(ROAR_BASE_DIR, "xai", f"alpha_{alpha_pct}", true_class)
            rnd_path = os.path.join(ROAR_BASE_DIR, "random", f"alpha_{alpha_pct}", true_class)
            
            # Mascheramento XAI (LRP)
            top_k_indices_xai = np.argsort(flat_lrp)[-k:]
            mask_xai = np.zeros(num_pixels, dtype=bool)
            mask_xai[top_k_indices_xai] = True
            mask_xai_2d = mask_xai.reshape((H, W))
            
            img_xai = orig_img.copy()
            img_xai[mask_xai_2d] = GRAY_VALUE
            
            # Mascheramento RANDOM
            random_indices = np.random.choice(num_pixels, size=k, replace=False)
            mask_rnd = np.zeros(num_pixels, dtype=bool)
            mask_rnd[random_indices] = True
            mask_rnd_2d = mask_rnd.reshape((H, W))
            
            img_rnd = orig_img.copy()
            img_rnd[mask_rnd_2d] = GRAY_VALUE
            
            # Salvataggio
            img_xai_255 = (img_xai * 255).astype(np.uint8)
            img_rnd_255 = (img_rnd * 255).astype(np.uint8)
            
            Image.fromarray(img_xai_255).save(os.path.join(xai_path, f"img_{processed_count:04d}.png"))
            Image.fromarray(img_rnd_255).save(os.path.join(rnd_path, f"img_{processed_count:04d}.png"))

        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"Processate e salvate {processed_count}/{MAX_IMAGES_TO_PROCESS} immagini")