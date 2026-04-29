"""
Randomly sample, colorize, and visualize segmentation masks.
"""

import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from segmentation.label_maps import LABEL_TO_TYPE, TYPE_TO_COLOR


# ----------------------------
# Colorize mask using mappings
# ----------------------------
def colorize_mask(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    # map labels → types
    type_mask = np.zeros_like(mask)
    for label, t in LABEL_TO_TYPE.items():
        type_mask[mask == label] = t

    # map types → colors
    for t, rgb in TYPE_TO_COLOR.items():
        color[type_mask == t] = rgb

    return color


# ----------------------------
# Visualization
# ----------------------------
def visualize_random_masks(image_root, mask_root, num_samples=10):
    mask_root = Path(mask_root)
    image_root = Path(image_root)

    # collect both png and jpg masks
    mask_paths = list(mask_root.rglob("*.png")) + list(mask_root.rglob("*.jpg"))

    if len(mask_paths) == 0:
        print("No masks found")
        return

    samples = random.sample(mask_paths, min(num_samples, len(mask_paths)))

    for mask_path in samples:
        mask = np.array(Image.open(mask_path))

        # build corresponding image path using SAME filename + extension
        rel_path = mask_path.relative_to(mask_root)
        image_path = image_root / rel_path

        # if exact match doesn't exist, try fallback extensions
        if not image_path.exists():
            stem = image_path.stem
            parent = image_path.parent

            candidates = [
                parent / f"{stem}.jpg",
                parent / f"{stem}.png",
                parent / f"{stem}.jpeg"
            ]

            image_path = next((p for p in candidates if p.exists()), None)

        if image_path is None or not image_path.exists():
            print(f"\n{mask_path}")
            print("  WARNING: matching image not found, skipping")
            continue

        # debug info
        unique_labels = np.unique(mask)
        mapped_types = np.unique([LABEL_TO_TYPE.get(int(l), -1) for l in unique_labels])

        print(f"\n{mask_path}")
        print(f"  unique labels: {unique_labels}")
        print(f"  mapped types: {mapped_types}")

        color_mask = colorize_mask(mask)

        plt.figure(figsize=(10, 4))

        # original image
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(np.array(Image.open(image_path)))
        plt.axis("off")

        # raw mask
        plt.subplot(1, 3, 2)
        plt.title("Raw Mask (labels)")
        plt.imshow(mask)
        plt.axis("off")

        # colored mask
        plt.subplot(1, 3, 3)
        plt.title("Colorized (label → type → color)")
        plt.imshow(color_mask)
        plt.axis("off")

        plt.tight_layout()
        plt.show()


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    visualize_random_masks(
        image_root="../eval_data/images",
        mask_root="../eval_data/masks",
        num_samples=5
    )