import os
import cv2
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
import warnings
import random 
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset 

warnings.filterwarnings('ignore')

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.metrics.road import ROADCombined

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
num_classes = 6

#Definisco le classi
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def category_name_to_index(category_name):
    """Restituisce l'indice della classe in base al nome."""
    return class_names.index(category_name)

def draw_text_with_shadow(img, text, position, font_scale=0.35, thickness=1):
    """Disegna un testo bianco con bordo nero per renderlo leggibile su qualsiasi sfondo."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(img, text, position, font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    
    cv2.putText(img, text, position, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

if __name__ == "__main__":

    # Trasformazioni 
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Trasformazione per l'immagine base (denormalizzata per il salvataggio ROAR)
    def denormalize(tensor):
        t = tensor.clone().detach().cpu()
        for i, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            t[i] = t[i] * s + m
        return t.clamp(0, 1)

    # Carico il Dataset completo ed estraggo 500 immaggini
    dataset_path = "seg_train/seg_train" 
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    

    num_samples = min(500, len(full_dataset))
    
    
    random_indices = random.sample(range(len(full_dataset)), num_samples)
    
    
    subset_dataset = Subset(full_dataset, random_indices)
    
    # Inizializzo il subset di 500 img
    dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

    # Carico il Modello
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(torch.load('best_model.pth')) 
    model.eval()

   
    target_layer = model.layer4[-1] 

   
    output_dir = "gradcam_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================
    # Inizializzazione cartelle ROAR
    # ==========================================
    ROAR_BASE_DIR = os.path.join(output_dir, "roar_datasets")
    ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
    GRAY_VALUE = 0.5
    
    for alpha in ALPHAS:
        alpha_pct = int(alpha * 100)
        for method in ["xai", "random"]:
            for c_name in class_names:
                os.makedirs(os.path.join(ROAR_BASE_DIR, method, f"alpha_{alpha_pct}", "train", c_name), exist_ok=True)


    # Inizializzo la metrica ROAD
    percentiles = [10, 50, 90]
    cam_metric = ROADCombined(percentiles=percentiles)

    print("Inizio elaborazione...")

    # Inizzializzo GradCAM (eventualmente da modificare per GradCAM++)
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        
        for idx, (img_tensor, label_idx) in enumerate(dataloader):
            

            original_idx = subset_dataset.indices[idx]
            img_path, _ = full_dataset.samples[original_idx]
            
            raw_image = Image.open(img_path).convert('RGB')
            
            # Estrapolo Altezza (H) e Larghezza (W) reali dal tensore (img_tensor ha forma [1, 3, H, W])
            H, W = img_tensor.shape[2], img_tensor.shape[3]
            
            # Ridimensiono l'immagine originale a W, H per lo sfondo
            rgb_img = cv2.resize(np.array(raw_image, dtype=np.float32) / 255.0, (W, H))
            
            true_class_name = class_names[label_idx.item()]
            
            
            targets = [ClassifierOutputTarget(label_idx.item())]
            
            # Generazione heatmap
            grayscale_cams = cam(input_tensor=img_tensor, targets=targets)
            grayscale_cam = grayscale_cams[0, :] 
            
            # Calcolo metrica
            scores = cam_metric(img_tensor, grayscale_cams, targets, model)
            score = scores[0] # Estraiamo il valore per la singola immagine
            
           
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Scrivo la metrica sull'immagine
            line1 = "GradCAM"
            line2 = f"Percentiles: {percentiles}"
            line3 = "Remove and Debias"
            line4 = f"{score:.5f}"
            
            draw_text_with_shadow(cam_image, line1, (5, 15))
            draw_text_with_shadow(cam_image, line2, (5, 30))
            draw_text_with_shadow(cam_image, line3, (5, 45))
            draw_text_with_shadow(cam_image, line4, (5, 60))
            
            # salvo tutto con img_numimg_predict
            save_path = os.path.join(output_dir, f"train_img_{original_idx}_{true_class_name}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
            
            
            # ==========================================
            # Faithfull Evalutation
            # ==========================================
            
            # L'immagine base pulita su cui operare (in [0, 1])
            img_base_tensor = denormalize(img_tensor.squeeze(0))
            img_base_np = img_base_tensor.permute(1, 2, 0).numpy().copy()
            
            num_pixels = H * W
            
            # grayscale_cam contiene già l'importanza in range [0, 1] scalata alla grandezza dell'immagine
            flat_cam = grayscale_cam.flatten()
            
            for alpha in ALPHAS:
                alpha_pct = int(alpha * 100)
                k = int(num_pixels * alpha)
                
                if k == 0:
                    continue
                    
                xai_path = os.path.join(ROAR_BASE_DIR, "xai", f"alpha_{alpha_pct}", "train", true_class_name)
                rnd_path = os.path.join(ROAR_BASE_DIR, "random", f"alpha_{alpha_pct}", "train", true_class_name)
                
                # Mascheramento XAI
                top_k_indices_xai = np.argsort(flat_cam)[-k:]
                mask_xai = np.zeros(num_pixels, dtype=bool)
                mask_xai[top_k_indices_xai] = True
                mask_xai_2d = mask_xai.reshape((H, W))
                
                img_xai = img_base_np.copy()
                img_xai[mask_xai_2d] = GRAY_VALUE
                
                # Mascheramento RANDOM
                random_indices = np.random.choice(num_pixels, size=k, replace=False)
                mask_rnd = np.zeros(num_pixels, dtype=bool)
                mask_rnd[random_indices] = True
                mask_rnd_2d = mask_rnd.reshape((H, W))
                
                img_rnd = img_base_np.copy()
                img_rnd[mask_rnd_2d] = GRAY_VALUE
                
                # Salvataggio
                img_xai_255 = (img_xai * 255).astype(np.uint8)
                img_rnd_255 = (img_rnd * 255).astype(np.uint8)
                
                Image.fromarray(img_xai_255).save(os.path.join(xai_path, f"img_{original_idx:04d}.png"))
                Image.fromarray(img_rnd_255).save(os.path.join(rnd_path, f"img_{original_idx:04d}.png"))
            
            print(f" Elaborata Immagine {idx+1}/{num_samples} (ID: {original_idx})")

    print("\nElaborazione e generazione ROAR datasets completata.")