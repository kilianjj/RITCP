"""
Detection pipeline for semantic 3D point clouds.

This script performs two main stages:

1) Semantic segmentation projection:
   Assigns a semantic label to each 3D point using per-image segmentation masks
   and COLMAP keypoint tracks.

2) Instance detection via clustering:
   Groups labeled 3D points into object instances using DBSCAN, optionally
   removes statistical outliers, and fits axis-aligned bounding boxes (AABB)
   to each detected instance.

Outputs:
- points3D.npy: Nx4 array of [x, y, z, label]
- detections.json: structured detection results with per-class instances
  and bounding boxes, along with segmentation and detection latency

Command line usage:
    python detection.py <pc_dir> <config.yaml> \
        --db_path <colmap.db> \
        --mask_dir <mask_directory>

Arguments:
- pc_dir: Directory containing COLMAP points3D.txt and where outputs are written
- config.yaml: Configuration file (merged with default detection config)
- db_path: Path to COLMAP database (for keypoints + image mapping)
- mask_dir: Directory containing segmentation masks aligned with images

Notes:
- Requires masks to match image filenames from COLMAP
- Uses DBSCAN for clustering; falls back to single cluster if sklearn missing
- Designed for offline batch processing, not real-time inference
"""

from __future__ import annotations
import argparse
import json
import os.path
import time
from pathlib import Path
from typing import Dict
from src.ritcp.common.segmentation_constants import TYPE_TO_DESCRIPTION
from segmentation_utils import assign_semantic_colors

from src.utils.config_utils import load_config
from src.ritcp.server.online.detection.config import DEFAULT_DETECTION_CONFIG
from src.utils.io_helpers import load_alignment_transform

import numpy as np

# todo: validate accurate values for the dbscan params per class?
CLASS_EPS = {
    "car": 30,
    "person": 0.5,
    "bike": 0.7
}

# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_points(xyz, eps, min_samples, verbose):
    """
    Cluster 3D points with DBSCAN.

    Args:
        xyz: Nx3 points
        eps: neighborhood radius
        min_samples: minimum cluster size
        verbose: print cluster info

    Returns:
        List of clusters (largest first), noise removed

    Falls back to a single cluster if sklearn is unavailable.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        print("WARNING: sklearn not installed → returning single cluster")
        return [xyz]

    if len(xyz) < min_samples:
        return [xyz] if len(xyz) > 0 else []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz)

    clusters = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        clusters.append(xyz[labels == lbl])

    clusters.sort(key=lambda c: len(c), reverse=True)

    if verbose:
        print(f"    clusters: {[len(c) for c in clusters]}")

    return clusters


def _remove_statistical_outliers(pts, nb_neighbors, std_ratio):
    """
    Filter outliers based on neighbor distance statistics.

    Args:
        pts: Nx3 points
        nb_neighbors: neighbors to consider
        std_ratio: threshold multiplier

    Returns:
        Filtered points
"""
    if len(pts) <= nb_neighbors:
        return pts

    sq_norms = np.sum(pts ** 2, axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2.0 * (pts @ pts.T)
    np.clip(sq_dists, 0.0, None, out=sq_dists)

    k = min(nb_neighbors, len(pts) - 1)
    partitioned = np.partition(sq_dists, k + 1, axis=1)[:, 1:k + 1]
    mean_dists = np.sqrt(partitioned).mean(axis=1)

    thresh = mean_dists.mean() + std_ratio * mean_dists.std()
    return pts[mean_dists <= thresh]


def _fit_aabb(pts):
    """
    Compute axis-aligned bounding box for points.

    Args:
        pts: Nx3 points

    Returns:
        Dict with min, max, center, extent, and point count
"""
    pt_min = pts.min(axis=0)
    pt_max = pts.max(axis=0)
    center = (pt_min + pt_max) / 2.0
    extent = pt_max - pt_min

    return {
        "min": pt_min.tolist(),
        "max": pt_max.tolist(),
        "center": center.tolist(),
        "extent": extent.tolist(),
        "num_points": len(pts),
    }


def _fit_bev_pca_obb(pts, R, t):
    """
    Compute an oriented 3D bounding box using PCA in BEV space.

    Technique:
        1. Transform points into BEV-aligned frame using R, t.
        2. Project to 2D (x, y).
        3. Run PCA (eigendecomposition of covariance) to get principal axes.
        4. Rotate points into PCA frame and compute axis-aligned bounds.
        5. Construct rectangle in PCA frame, rotate back to BEV frame.
        6. Use min/max z in BEV frame for height.
        7. Transform box corners back to original coordinates.

    Notes:
        - Fast (O(N)), but not guaranteed minimum-area rectangle.
        - Works well when point distribution is roughly elliptical.

    Args:
        pts: (N,3) array
        R: (3,3) rotation matrix (world -> BEV)
        t: (3,) or (3,1) translation vector (world -> BEV)

    Returns:
        Dict with min, max, center, extent, num_points
    """
    pts = np.asarray(pts)
    t = np.array([0, 0, -t])

    # Transform to BEV
    pts_bev = (R @ pts.T).T + t

    xy = pts_bev[:, :2]
    z = pts_bev[:, 2]

    # Center data
    mean_xy = xy.mean(axis=0)
    centered = xy - mean_xy

    # PCA via covariance eigendecomposition
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvectors by descending eigenvalue
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Rotate points into PCA frame
    xy_pca = centered @ eigvecs

    min_xy = xy_pca.min(axis=0)
    max_xy = xy_pca.max(axis=0)

    # Rectangle corners in PCA frame
    corners_pca = np.array([
        [min_xy[0], min_xy[1]],
        [max_xy[0], min_xy[1]],
        [max_xy[0], max_xy[1]],
        [min_xy[0], max_xy[1]],
    ])

    # Transform back to BEV frame
    corners_bev_xy = corners_pca @ eigvecs.T + mean_xy

    z_min, z_max = z.min(), z.max()

    # Create 8 corners (bottom + top)
    corners_bev = []
    for zz in [z_min, z_max]:
        for xy_corner in corners_bev_xy:
            corners_bev.append([xy_corner[0], xy_corner[1], zz])
    corners_bev = np.array(corners_bev)

    # Transform back to original frame
    R_inv = R.T
    corners_world = (R_inv @ (corners_bev - t).T).T

    pt_min = corners_world.min(axis=0)
    pt_max = corners_world.max(axis=0)
    center = (pt_min + pt_max) / 2.0
    extent = pt_max - pt_min

    return {
        "min": pt_min.tolist(),
        "max": pt_max.tolist(),
        "center": center.tolist(),
        "extent": extent.tolist(),
        "num_points": len(pts),
    }


def _fit_bev_min_area_obb(pts, R, t):
    """
    Compute an oriented 3D bounding box using convex hull + rotating calipers.

    Technique:
        1. Transform points into BEV-aligned frame using R, t.
        2. Project to 2D (x, y).
        3. Compute convex hull (reduces problem to boundary points).
        4. Apply rotating calipers:
            - For each hull edge, align it with x-axis
            - Compute bounding rectangle
            - Track minimum-area rectangle
        5. Lift rectangle using min/max z for height.
        6. Transform box corners back to original coordinates.

    Notes:
        - Computes true minimum-area bounding rectangle in 2D.
        - More accurate than PCA but more expensive:
            O(N log N) for hull + O(H) calipers.

    Args:
        pts: (N,3) array
        R: (3,3) rotation matrix (world -> BEV)
        t: (3,) or (3,1) translation vector (world -> BEV)

    Returns:
        Dict with min, max, center, extent, num_points
    """
    pts = np.asarray(pts)
    t = np.array([0, 0, -t])

    # Transform to BEV
    pts_bev = (R @ pts.T).T + t

    xy = pts_bev[:, :2]
    z = pts_bev[:, 2]

    # --- Convex Hull (Monotone Chain) ---
    def convex_hull(points):
        pts_sorted = points[np.lexsort((points[:,1], points[:,0]))]

        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

        lower = []
        for p in pts_sorted:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts_sorted):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return np.array(lower[:-1] + upper[:-1])

    hull = convex_hull(xy)

    if len(hull) < 3:
        # fallback to AABB-like behavior
        min_xy = xy.min(axis=0)
        max_xy = xy.max(axis=0)
        corners_bev_xy = np.array([
            [min_xy[0], min_xy[1]],
            [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]],
            [min_xy[0], max_xy[1]],
        ])
    else:
        best_area = np.inf
        best_rect = None

        for i in range(len(hull)):
            p0 = hull[i]
            p1 = hull[(i + 1) % len(hull)]

            edge = p1 - p0
            theta = -np.arctan2(edge[1], edge[0])

            c, s = np.cos(theta), np.sin(theta)
            R2 = np.array([[c, -s], [s, c]])

            rot = hull @ R2.T

            min_xy = rot.min(axis=0)
            max_xy = rot.max(axis=0)

            area = np.prod(max_xy - min_xy)

            if area < best_area:
                best_area = area
                best_rect = (min_xy, max_xy, R2)

        min_xy, max_xy, R2 = best_rect

        corners_rot = np.array([
            [min_xy[0], min_xy[1]],
            [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]],
            [min_xy[0], max_xy[1]],
        ])

        corners_bev_xy = corners_rot @ R2

    z_min, z_max = z.min(), z.max()

    corners_bev = []
    for zz in [z_min, z_max]:
        for xy_corner in corners_bev_xy:
            corners_bev.append([xy_corner[0], xy_corner[1], zz])
    corners_bev = np.array(corners_bev)

    # Back to world
    R_inv = R.T
    corners_world = (R_inv @ (corners_bev - t).T).T

    pt_min = corners_world.min(axis=0)
    pt_max = corners_world.max(axis=0)
    center = (pt_min + pt_max) / 2.0
    extent = pt_max - pt_min

    return {
        "min": pt_min.tolist(),
        "max": pt_max.tolist(),
        "center": center.tolist(),
        "extent": extent.tolist(),
        "num_points": len(pts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main logic (CONFIG-DRIVEN)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bboxes_from_npy(npy_path: str, config: Dict,
                            rot_mat: np.ndarray, t_vec: np.ndarray) -> Dict | None:
    """
    Detect object instances from labeled point cloud.

    Args:
        npy_path: path to Nx4 [x,y,z,label] array
        config: detection settings

    Returns:
        Dict of per-class instances or error message

    Per class: cluster → filter → AABB.
    """
    db_cfg = config["db_scan"]
    outlier_cfg = config["outlier_filtering"]
    class_cfg = config["classes"]
    runtime_cfg = config["runtime"]
    bbox_cfg = config["bbox"]

    eps_default = db_cfg["epsilon"]
    min_samples = db_cfg["min_samples"]
    min_points = db_cfg["min_points"]

    use_outlier = outlier_cfg["enabled"]
    nb_neighbors = outlier_cfg["num_neighbors"]
    std_ratio = outlier_cfg["std_ratio"]

    bbox_type = bbox_cfg["type"]

    verbose = runtime_cfg["verbose"]

    include = set(class_cfg["include"])
    exclude = set(class_cfg["exclude"])

    try:
        data = np.load(npy_path)
    except Exception as e:
        return {
            "msg": f"detection failed to load npy: {str(e)}"
        }
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 4:
        return {
            "msg": "detection skipped due to lack of valid points"
        }

    xyz = data[:, :3]
    labels = data[:, 3].astype(int)

    result = {}

    if verbose:
        print(f"\nLoaded {len(xyz)} points")

    for label_id, class_name in TYPE_TO_DESCRIPTION.items():
        if label_id == 0:
            continue

        if class_name not in include or class_name in exclude:
            continue

        mask = labels == label_id
        class_points = xyz[mask]

        if len(class_points) == 0:
            continue

        if verbose:
            print(f"\n[{class_name}] {len(class_points)} pts")

        # Per-class epsilon override
        eps = CLASS_EPS.get(class_name, eps_default)

        clusters = _cluster_points(class_points, eps, min_samples, verbose)

        instances = {}
        instance_idx = 0

        for cluster in clusters:
            if len(cluster) < min_points:
                continue

            if use_outlier:
                cluster = _remove_statistical_outliers(
                    cluster,
                    nb_neighbors=nb_neighbors,
                    std_ratio=std_ratio,
                )

            if len(cluster) < min_points:
                continue

            name = f"{class_name}_{instance_idx}"

            if bbox_type == "aa":
                instances[name] = _fit_aabb(cluster)
            elif bbox_type == "pca":
                instances[name] = _fit_bev_pca_obb(cluster, rot_mat, t_vec)
            elif bbox_type == "min":
                instances[name] = _fit_bev_min_area_obb(cluster, rot_mat, t_vec)
            else:
                raise NotImplementedError
            instance_idx += 1

        if instances:
            result[class_name] = {
                "num_instances": len(instances),
                "instances": instances,
            }

    output = {
        "num_classes_detected": len(result),
        **result,
    }

    return output


# ─────────────────────────────────────────────────────────────────────────────
# CLI: accept new point cloud txt on cmd line, convert to npy and segment, run detections for each class
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Run segmentation + detection pipeline from CLI.

    Loads config, assigns semantic labels, clusters points,
    records latency, and saves detections.json.

    Usage:
        python detection.py <pc_dir> <config.yaml> \
            --db_path <db> --mask_dir <masks>
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("pc_dir")
    parser.add_argument("config_path")
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--mask_dir", required=True)

    args = parser.parse_args()

    config = load_config(args.config_path, DEFAULT_DETECTION_CONFIG)

    seg_start = time.time()
    assign_semantic_colors(
        pc_dir=args.pc_dir,
        db_path=args.db_path,
        mask_dir=args.mask_dir,
    )
    seg_latency = time.time() - seg_start

    npy_path = os.path.join(args.pc_dir, "points3D.npy")

    # try to load ground alignment transformation
    R, t = np.eye(3), np.zeros(3)
    if config["bbox"]["type"] != "aa":
        try:
            base_path = Path(args.pc_dir).parent.parent.parent / "base" / "0"
            R, t = load_alignment_transform(base_path)
        except Exception as e:
            # if failure to load alignment, fallback to aa bboxes
            print(e)
            print("Falling back to axis aligned bboxes")
            config["bbox"]["type"] = "aa"

    det_start = time.time()
    result = compute_bboxes_from_npy(npy_path=npy_path, config=config, rot_mat=R, t_vec=t)
    det_latency = time.time() - det_start

    if result is None:
        result = {}

    result["segmentation_latency_sec"] = seg_latency
    result["detection_latency_sec"] = det_latency

    out_path = os.path.join(args.pc_dir, "detections.json")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved detections → {out_path}")


if __name__ == "__main__":
    main()