"""
Semantic labeling for COLMAP 3D points using 2D segmentation masks.

Projects each 3D point into its observed image keypoints, samples the
corresponding mask labels, assigns the majority semantic class, and writes:

Assumes mask filenames match COLMAP image names.
"""

from PIL import Image
from collections import Counter
from incremental_sfm.colmap_helpers import *
from segmentation.label_maps import TYPE_TO_COLOR, LABEL_TO_TYPE
import numpy as np
from pathlib import Path

def load_mask(mask_root, image_name):
    path = mask_root / image_name
    return np.array(Image.open(path))

def majority_label(labels):
    labels = [l for l in labels if l is not None]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]

def assign_semantic_colors(
    pc_dir,
    db_path,
    mask_dir,
):
    """
        Assign semantic labels to 3D points.

        Args:
            pc_dir: directory with COLMAP points file
            db_path: COLMAP database path
            mask_dir: segmentation masks directory

        Outputs:
            - points3D.txt with colored labels
            - points3D.npy with [x, y, z, label]
        """
    image_map = load_image_id_to_name(db_path)
    points3D_path = Path(pc_dir) / "points3D_colored.txt"
    points = load_points3D(points3D_path)

    mask_dir = Path(mask_dir)

    new_points = []

    for p in points:
        labels = []

        for image_id, kp_idx in p["track"]:
            if image_id not in image_map:
                continue

            # preserve relative path (e.g., car1/frame1.jpg → car1/frame1.png)
            rel_path = Path(image_map[image_id]).with_suffix(".png")
            mask_path = mask_dir / rel_path

            if not mask_path.exists():
                continue

            mask = np.array(Image.open(mask_path))

            # keypoint lookup
            kpts = load_keypoints(db_path, image_id)
            if kpts is None or kp_idx >= len(kpts):
                continue

            x, y = kpts[kp_idx]
            x = round(x)
            y = round(y)

            # mask size
            mask_h, mask_w = mask.shape[:2]

            # bounds check (after scaling)
            if (
                    x < 0 or y < 0 or
                    y >= mask_h or
                    x >= mask_w
            ):
                continue

            true_label = int(mask[y, x])
            labels.append(LABEL_TO_TYPE.get(true_label, 0))  # safer

        label = majority_label(labels)

        new_points.append({
            "xyz": p["xyz"],
            "label": label,
            "error": p["error"],
            "track": p["track"]
        })

    dir_name = os.path.dirname(points3D_path)
    # write back in COLMAP format
    with open(os.path.join(dir_name, "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK\n")

        for i, p in enumerate(new_points):
            xyz = p["xyz"]
            rgb = TYPE_TO_COLOR.get(p["label"], (0, 255, 0))

            track_flat = []
            for img_id, pt_id in p["track"]:
                track_flat.extend([img_id, pt_id])

            line = (
                f"{i} "
                f"{xyz[0]} {xyz[1]} {xyz[2]} "
                f"{rgb[0]} {rgb[1]} {rgb[2]} "
                f"{p['error']} "
                + " ".join(map(str, track_flat))
                + "\n"
            )

            f.write(line)

    xyz_label = []

    for p in new_points:
        xyz = p["xyz"]
        label = p["label"]

        xyz_label.append([
            xyz[0],
            xyz[1],
            xyz[2],
            label
        ])

    xyz_label = np.asarray(xyz_label, dtype=np.float32)

    np.save(os.path.join(pc_dir, "points3D.npy"), xyz_label)