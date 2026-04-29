"""
Visualization tool for COLMAP 3D point tracks with image or mask overlays to validate triangulated features
quality across views.

Displays random sampled 3D reconstructed points and their corresponding
multi-view observations. For each selected point, it shows the image (or
segmentation mask) views where the point appears and highlights the
associated 2D keypoint locations.

Options:
- show_mask=True: visualize segmentation masks instead of RGB images
- num_samples: number of 3D points to visualize
- masks_root: directory containing per-image segmentation masks

Inputs:
- data_path: dataset root containing images/
- db_path: COLMAP database (keypoints + image mapping)
- points3D_path: reconstructed 3D points file (COLMAP format)
- masks_root: directory of segmentation masks (optional)
"""

import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from src.utils.io_helpers import *
from src.ritcp.common.segmentation_constants import LABEL_TO_TYPE, TYPE_TO_COLOR
from src.ritcp.common.segmentation_constants import load_mask

from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
model = lraspp_mobilenet_v3_large(weights=weights)

# Get the class list (index 0 is background)
classes = weights.meta["categories"]

# Create a mapping dictionary if needed
int_to_class = {i: name for i, name in enumerate(classes)}

def colorize_mask(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for label, type_id in LABEL_TO_TYPE.items():
        rgb = TYPE_TO_COLOR.get(type_id, (0, 255, 0))  # fallback green if unknown label/error
        color[mask == label] = rgb

    return color

# ----------------------------
# Main visualization
# ----------------------------
def visualize_tracks(
    data_path,
    db_path,
    points3D_path,
    masks_root,
    show_mask,
    num_samples=10,
):
    data_path = Path(data_path)
    images_root = data_path / "images"
    masks_root = Path(masks_root) if masks_root else None

    image_map = load_image_id_to_name(db_path)
    points = load_points3D(points3D_path)

    # sample valid points (must have tracks)
    points = [p for p in points if len(p["track"]) > 0]
    samples = random.sample(points, min(num_samples, len(points)))

    for p in samples:

        fig, axes = plt.subplots(1, len(p["track"]), figsize=(5 * len(p["track"]), 5))
        if len(p["track"]) == 1:
            axes = [axes]

        fig.suptitle(f"3D point (error={p['error']:.3f})")

        for i, (image_id, kp_idx) in enumerate(p["track"]):

            if image_id not in image_map:
                continue

            image_name = image_map[image_id]

            if show_mask:
                # build mask path: masks_root / relative_image_path with .png
                rel_path = Path(image_name).with_suffix(".png")
                img_path = masks_root / rel_path

                if not img_path.exists():
                    continue

                mask = load_mask(masks_root, rel_path)
                img = colorize_mask(mask)
            else:
                img_path = images_root / image_name

                if not img_path.exists():
                    continue

                img = np.array(Image.open(img_path).convert("RGB"))

            # keypoints
            kpts = load_keypoints(db_path, image_id)
            if kpts is None or kp_idx >= len(kpts):
                continue

            x, y = kpts[kp_idx]

            axes[i].imshow(img)
            axes[i].scatter([x], [y], c="black" if show_mask else "lime", s=30)
            axes[i].set_title(image_name)
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_tracks(
        data_path="../data/eval_data",
        db_path="../data/eval_data/eval_sparse/database.db",
        points3D_path="../data/eval_data/eval_sparse/incremental/point_clouds/iteration2/points3D.txt",
        masks_root="../eval_data/masks/",
        show_mask=True,
        num_samples=5
    )