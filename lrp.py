import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

#IMPORTO ZENNIT PER APPLICARE LRP, VISTO CHE é OTTIMIZZATO PER LA RESNET18
from zennit.composites import EpsilonPlusFlat
from zennit.attribution import Gradient
from captum.attr import visualization as viz

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

    
    composite = EpsilonPlusFlat()

    
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

        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"Processate e salvate {processed_count}/{MAX_IMAGES_TO_PROCESS} immagini")

