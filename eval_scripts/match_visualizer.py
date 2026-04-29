"""
Visualize feature matches between two images from a COLMAP reconstruction.

Displays two images side-by-side with matched keypoints connected by lines to inspection of match quality.

Inputs:
- db_path: COLMAP database containing keypoints and matches
- images_root: root directory of images
- img1_name, img2_name: relative image paths in dataset
"""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from incremental_sfm.colmap_helpers import *

def visualize_pair(db_path, images_root, img1_name, img2_name):
    images_root = Path(images_root)

    name_to_id = load_image_name_to_id(db_path)

    if img1_name not in name_to_id or img2_name not in name_to_id:
        print("Image names not found in database")
        return

    id1, id2 = name_to_id[img1_name], name_to_id[img2_name]

    # load images
    img1 = np.array(Image.open(images_root / img1_name).convert("RGB"))
    img2 = np.array(Image.open(images_root / img2_name).convert("RGB"))

    # keypoints
    kpts1 = load_keypoints(db_path, id1)
    kpts2 = load_keypoints(db_path, id2)

    matches = load_matches(db_path, id1, id2, use_two_view_table=False)

    if matches is None:
        print("No matches found")
        return

    # side-by-side canvas
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    plt.figure(figsize=(15, 8))
    plt.imshow(canvas)

    # draw matches
    for i1, i2 in matches:
        if i1 >= len(kpts1) or i2 >= len(kpts2):
            continue

        x1, y1 = kpts1[i1]
        x2, y2 = kpts2[i2]

        plt.plot([x1, x2 + w1], [y1, y2], linewidth=0.5)
        plt.scatter([x1, x2 + w1], [y1, y2], s=5)

    plt.axis("off")
    plt.title(f"{img1_name} ↔ {img2_name}")
    plt.show()

if __name__ == "__main__":
    visualize_pair(
        db_path="../data/eval_data/static_sparse/database.db",
        images_root="../eval_data/images",
        img1_name="car2/frame_0001.jpg",
        img2_name="car1/frame_0005.jpg",
    )