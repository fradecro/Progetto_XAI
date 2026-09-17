import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from zennit.composites import EpsilonPlusFlat
from zennit.attribution import Gradient
from zennit.torchvision import ResNetCanonizer
from captum.attr import visualization as viz


DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

NUM_CLASS = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TEST_DIR = "seg_test/seg_test"

BATCH = 32
IMAGES_PER_CLASS = 84

OUTPUT_DIR = "lrp_results"

ROAR_BASE_DIR = os.path.join(
    OUTPUT_DIR,
    "roar_datasets"
)

ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]

GRAY_VALUE = 0.5

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


def build_model(num_classes):

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    return model


def denormalize(tensor):

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=tensor.dtype,
        device=tensor.device
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        dtype=tensor.dtype,
        device=tensor.device
    ).view(3, 1, 1)

    return (
        tensor * std + mean
    ).clamp(0, 1)


model = build_model(
    NUM_CLASS
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=DEVICE
    )
)

model.eval()


test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transforms
)


selected_indices = []

class_counts = {
    i: 0
    for i in range(NUM_CLASS)
}


for idx, (_, label) in enumerate(
    test_dataset.samples
):

    if class_counts[label] < IMAGES_PER_CLASS:

        selected_indices.append(
            idx
        )

        class_counts[label] += 1

    if all(
        class_counts[i] >= IMAGES_PER_CLASS
        for i in range(NUM_CLASS)
    ):
        break


balanced_dataset = Subset(
    test_dataset,
    selected_indices
)


test_loader = DataLoader(
    balanced_dataset,
    batch_size=BATCH,
    shuffle=False
)


print("Immagini selezionate:")

for i in range(NUM_CLASS):

    print(
        f"{CLASS_NAMES[i]}: "
        f"{class_counts[i]}"
    )


print(
    f"Totale immagini: "
    f"{len(selected_indices)}"
)


for alpha in ALPHAS:

    alpha_pct = int(
        alpha * 100
    )

    for class_name in CLASS_NAMES:

        xai_dir = os.path.join(
            ROAR_BASE_DIR,
            "xai",
            f"alpha_{alpha_pct}",
            "test",
            class_name
        )

        random_dir = os.path.join(
            ROAR_BASE_DIR,
            "random",
            f"alpha_{alpha_pct}",
            "test",
            class_name
        )

        os.makedirs(
            xai_dir,
            exist_ok=True
        )

        os.makedirs(
            random_dir,
            exist_ok=True
        )


canonizer = ResNetCanonizer()

composite = EpsilonPlusFlat(
    canonizers=[canonizer]
)


processed_count = 0


for images, labels in test_loader:

    images = images.to(
        DEVICE
    )

    labels = labels.to(
        DEVICE
    )

    images = (
        images
        .detach()
        .requires_grad_(True)
    )


    with torch.no_grad():

        outputs = model(
            images
        )

        preds = outputs.argmax(
            dim=1
        )


    target_labels = F.one_hot(
        preds,
        num_classes=NUM_CLASS
    ).float().to(
        DEVICE
    )


    with Gradient(
        model,
        composite
    ) as attributor:

        _, attributions = attributor(
            images,
            target_labels
        )


    for i in range(
        images.size(0)
    ):

        true_idx = labels[i].item()

        pred_idx = preds[i].item()

        true_class = CLASS_NAMES[
            true_idx
        ]

        pred_class = CLASS_NAMES[
            pred_idx
        ]


        original_dataset_idx = selected_indices[
            processed_count
        ]


        source_path = test_dataset.samples[
            original_dataset_idx
        ][0]


        source_filename = os.path.basename(
            source_path
        )


        source_name = os.path.splitext(
            source_filename
        )[0]


        orig_img = (
            denormalize(
                images[i].detach()
            )
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )


        attr_chw = (
            attributions[i]
            .detach()
            .cpu()
            .numpy()
        )


        lrp_2d = np.abs(
            attr_chw
        ).mean(
            axis=0
        )


        H, W = lrp_2d.shape

        num_pixels = H * W

        flat_lrp = lrp_2d.flatten()


        for alpha in ALPHAS:

            alpha_pct = int(
                alpha * 100
            )

            k = int(
                num_pixels * alpha
            )


            top_k_indices = np.argpartition(
                flat_lrp,
                -k
            )[-k:]


            mask_xai = np.zeros(
                num_pixels,
                dtype=bool
            )


            mask_xai[
                top_k_indices
            ] = True


            mask_xai = mask_xai.reshape(
                H,
                W
            )


            img_xai = orig_img.copy()


            img_xai[
                mask_xai
            ] = GRAY_VALUE


            random_indices = np.random.choice(
                num_pixels,
                size=k,
                replace=False
            )


            mask_random = np.zeros(
                num_pixels,
                dtype=bool
            )


            mask_random[
                random_indices
            ] = True


            mask_random = mask_random.reshape(
                H,
                W
            )


            img_random = orig_img.copy()


            img_random[
                mask_random
            ] = GRAY_VALUE


            img_xai_255 = (
                np.clip(
                    img_xai,
                    0,
                    1
                ) * 255
            ).astype(
                np.uint8
            )


            img_random_255 = (
                np.clip(
                    img_random,
                    0,
                    1
                ) * 255
            ).astype(
                np.uint8
            )


            xai_dir = os.path.join(
                ROAR_BASE_DIR,
                "xai",
                f"alpha_{alpha_pct}",
                "test",
                true_class
            )


            random_dir = os.path.join(
                ROAR_BASE_DIR,
                "random",
                f"alpha_{alpha_pct}",
                "test",
                true_class
            )


            output_filename = (
                f"{source_name}.png"
            )


            xai_path = os.path.join(
                xai_dir,
                output_filename
            )


            random_path = os.path.join(
                random_dir,
                output_filename
            )


            Image.fromarray(
                img_xai_255
            ).save(
                xai_path
            )


            Image.fromarray(
                img_random_255
            ).save(
                random_path
            )


        attr_hwc = np.transpose(
            attr_chw,
            (1, 2, 0)
        )


        fig, _ = viz.visualize_image_attr_multiple(
            attr_hwc,
            orig_img,
            methods=[
                "original_image",
                "heat_map"
            ],
            signs=[
                "all",
                "positive"
            ],
            titles=[
                f"Originale ({true_class})",
                f"LRP Zennit ({pred_class})"
            ],
            show_colorbar=True,
            fig_size=(8, 4),
            use_pyplot=False
        )


        status = (
            "CORRETTA"
            if true_idx == pred_idx
            else "ERRATA"
        )


        plot_filename = (
            f"img_"
            f"{processed_count:04d}_"
            f"{status}_"
            f"T-{true_class}_"
            f"P-{pred_class}.png"
        )


        plot_path = os.path.join(
            OUTPUT_DIR,
            plot_filename
        )


        fig.savefig(
            plot_path,
            dpi=150,
            bbox_inches="tight"
        )


        plt.close(
            fig
        )


        processed_count += 1


        if processed_count % 50 == 0:

            print(
                f"Processate "
                f"{processed_count}/"
                f"{len(selected_indices)} "
                f"immagini"
            )


print()

print(
    f"Completato: "
    f"{processed_count} immagini"
)

print(
    f"Dataset ROAR salvato in: "
    f"{os.path.abspath(ROAR_BASE_DIR)}"
)