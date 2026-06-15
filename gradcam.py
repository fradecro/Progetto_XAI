import os
import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms

from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
import torch.nn as nn
import random
import torch
from torchvision import models
from torchvision import datasets
from PIL import Image
import numpy as np
import cv2
import torch
from PIL import Image
from typing import List, Callable, Optional




 
num_classes = 6
method=GradCAM
DEVICE = torch.device("cpu")
NUM_CLASS = 6
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

model = models.resnet18(weights=None) 
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
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()




data_dir = "seg_test/seg_test"
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Seleziona 500 indici random
num_samples = 100
indices = random.sample(range(len(dataset)), num_samples)

img_tensors = []
images = []       # versioni PIL/numpy per la visualizzazione
labels = []

for idx in indices:
    img_tensor, label = dataset[idx]
    img_tensors.append(img_tensor)
    labels.append(label)
    # ricarica l'immagine originale (senza ToTensor) per la visualizzazione
    path, _ = dataset.samples[idx]
    images.append(Image.open(path).convert("RGB").resize((150, 150)))


img_tensors_batch = torch.stack(img_tensors)
class_names = dataset.classes  # lista nomi classi, es. ['buildings', 'forest', ...]











class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

target_layer = model.layer4[-1]

results = []





output_dir = "gradcam_results"
os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join(output_dir, "predictions.txt")

target_layer = model.layer4[-1]

with open(log_path, "w") as f:
    with GradCAM(model=model, target_layers=[target_layer]) as cam:

        for i, (img_tensor, label, img_pil) in enumerate(zip(img_tensors, labels, images)):

            target = [ClassifierOutputTarget(label)]

            # Predizione del modello
            with torch.no_grad():
                output = model(img_tensor.unsqueeze(0))
            pred_idx = output.argmax(dim=1).item()

            # Grad-CAM
            grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0), targets=target)
            grayscale_cam = grayscale_cam[0, :]

            rgb_img = np.array(img_pil) / 255.0
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Salva l'immagine con la heatmap (in BGR per cv2)
            out_path = os.path.join(output_dir, f"image_{i:03d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            # Scrivi il risultato nel file di testo
            true_class = class_names[label]
            pred_class = class_names[pred_idx]
            f.write(f"image_{i:03d}.png -> true: {true_class}, predicted: {pred_class}\n")

print(f"Fatto! Risultati salvati in '{output_dir}/'")


    

