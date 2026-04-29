"""
Batch semantic segmentation for image datasets using a pretrained lightweight model.

This script runs a semantic segmentation model (LR-ASPP with MobileNetV3 backbone)
over a selected set of image subdirectories and writes per-pixel class predictions
as mask images.

Functionality:
- Recursively scans specified subdirectories for images
- Runs inference using a pretrained torchvision segmentation model
- Resizes predictions back to original image resolution
- Saves output masks as PNG files (one channel, class indices)

Command line usage:
    python segment.py

Configuration (inside script):
- Set `images_root` to dataset root
- Set `masks_root` to desired output directory
- Populate `selected` with subdirectories to process

Output:
- masks_root: directory mirroring input structure, containing segmentation masks

Directory structure example:
    images_root/
        vehicle_01/
            frame_0001.jpg
            ...
        vehicle_02/
            ...
    masks_root/
        vehicle_01/
            frame_0001.png
        vehicle_02/
            ...

Notes:
- Uses pretrained COCO weights (class indices follow COCO labeling)
- Output masks store raw class IDs (no color mapping)
- Runs on CPU by default; pass device="cuda" for GPU
"""

import os
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm

# ----------------------------
# Minimal ENet (pretrained loader)
# ----------------------------
# You can swap this with any ENet repo if you already have one
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

def load_model(device):
    model = lraspp_mobilenet_v3_large(weights="DEFAULT")
    model.eval()
    model.to(device)
    return model

# ----------------------------
# Transform
# ----------------------------
transform = T.Compose([
    # T.Resize(480),  # adjust if needed
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ----------------------------
# Main processing
# ----------------------------
def process_images(images_root, masks_root, selected=None, device="cpu"):
    model = load_model(device)

    images_root = Path(images_root)
    masks_root = Path(masks_root)

    os.makedirs(masks_root, exist_ok=True)

    image_paths = []

    # ----------------------------
    # Only process selected subdirs
    # ----------------------------
    for subdir in selected:
        sub_path = images_root / subdir
        if not sub_path.exists():
            print(f"[WARN] Skipping missing dir: {sub_path}")
            continue

        image_paths.extend(sub_path.rglob("*.[jp][pn]g"))

    print(f"Found {len(image_paths)} images")

    for img_path in tqdm(image_paths):
        rel_path = img_path.relative_to(images_root)
        out_path = (masks_root / rel_path).with_suffix(".png")

        os.makedirs(out_path.parent, exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        import torch.nn.functional as F

        # original image size
        orig_w, orig_h = img.size  # PIL gives (W, H)

        with torch.no_grad():
            output = model(input_tensor)["out"]  # (1, C, H, W)

            output = F.interpolate(
                output,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )

            pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        mask = pred.astype(np.uint8)
        Image.fromarray(mask).save(out_path)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    data = "../eval_data/"
    selected = []
    for i in range(1, 4):
        selected.append(os.path.join(data, "images", f"car{i}"))
    process_images(
        images_root="",
        masks_root="masks/",
        selected=selected,
    )