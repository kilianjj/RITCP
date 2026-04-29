"""
This script aggregates and "plays back" the full reconstruction pipeline,
including static COLMAP models, incremental point clouds, detections,
and feature-level diagnostics (matches and tracks).

Inputs:
- project_dir: SfM project root containing:
    base/0/                 -> static COLMAP model
    incremental/point_clouds/
        iteration*/         -> incremental outputs (points, detections)
    incremental/logs/       -> SfM runtime logs
    database.db             -> COLMAP database
    images/                 -> input images
    masks/                  -> segmentation masks (optional)

Main behavior:
1) Optionally merges all point clouds into a unified COLMAP model
2) Loads and displays static reconstruction
3) Iterates through incremental reconstruction batches:
    - loads point clouds
    - loads detection results
    - prints SfM + detection summaries
    - renders 3D scene
    - optionally shows matches and tracks
4) Prints final SfM summary

Options:
- toggle display_with_static: include static model in visualization
- toggle display_matches: show example pairwise feature matches
- toggle display_tracks: show example 3D point track projections
- toggle merge_pcs: merge incremental outputs into one model
"""

import json
import shutil
from pathlib import Path
from numpy_pc_render import render_point_clouds, load_detection_jsons
from src.utils.io_helpers import read_colmap_points3D, copy_model_with_new_points
import os
import numpy as np
from src.ritcp.server.online.ritcp_engine import print_batch_summary, print_final_summary
from match_visualizer import visualize_pair
from track_visualizer import visualize_tracks

def print_detection_summary(detection_log: dict):
    num_classes = detection_log.get("num_classes_detected", 0)
    seg_time = detection_log.get("segmentation_latency_sec", 0.0)
    det_time = detection_log.get("detection_latency_sec", 0.0)

    total_instances = 0
    class_lines = []

    for class_name, class_data in detection_log.items():
        if class_name in {
            "num_classes_detected",
            "segmentation_latency_sec",
            "detection_latency_sec",
            "latency_sec",
            "msg"
        }:
            continue

        num_instances = class_data.get("num_instances", 0)
        total_instances += num_instances

        instances = class_data.get("instances", {})
        instance_details = []

        for inst_name, inst in instances.items():
            center = inst["center"]
            extent = inst["extent"]
            pts = inst["num_points"]

            instance_details.append(
                f"      - {inst_name}: pts={pts:>4} | "
                f"center=({center[0]:>6.2f},{center[1]:>6.2f},{center[2]:>6.2f}) | "
                f"extent=({extent[0]:>6.2f},{extent[1]:>6.2f},{extent[2]:>6.2f})"
            )

        class_block = f"""
    [{class_name}] instances: {num_instances}
{chr(10).join(instance_details)}
"""
        class_lines.append(class_block)

    s = f"""
==================== Detection Summary ====================
Classes detected: {num_classes}
Total instances: {total_instances}

Latency:
 • Segmentation: {seg_time:>7.3f}s
 • Detection:    {det_time:>7.3f}s
 • Total:        {seg_time + det_time:>7.3f}s

Per-class details:
{''.join(class_lines)}
===========================================================
"""
    print(s)

def merge_all_point_clouds(proj_root):
    base_dir = Path(proj_root) / "base" / "0"
    new_model = Path(proj_root) / "incremental" / "result"
    if os.path.exists(new_model):
        shutil.rmtree(new_model)

    new_points = []
    pcs_dir = Path(proj_root) / "incremental" / "point_clouds"
    if os.path.exists(pcs_dir):
        for d in sorted(os.listdir(pcs_dir)):
            iter_dir = os.path.join(pcs_dir, d)

            if not os.path.isdir(iter_dir):
                continue
            if not d.startswith("iteration"):
                continue

            pts_path = os.path.join(iter_dir, "points3D.txt")
            if os.path.exists(pts_path):
                new_points.append(pts_path)

    base_points = os.path.join(base_dir, "points3D.txt")
    if os.path.exists(base_points):
        new_points.append(base_points)

    copy_model_with_new_points(base_dir, new_model, new_points,
                               set_all_grey=[base_points])


if __name__ == "__main__":
    data_path = Path("../data/eval_data/")
    proj = "eval3"
    project_dir = data_path / (proj + "_sparse")
    incr_root = project_dir / "incremental"
    pc_dir = incr_root / "point_clouds"
    log_dir = project_dir / "incremental" / "logs"
    base_pc_path = project_dir / "base" / "0" / "points3D.txt"
    display_with_static = True
    grey_static = False
    display_matches = False
    display_tracks = False
    merge_pcs = True

    # merge into one colmap style model for visualization if desired
    if merge_pcs:
        merge_all_point_clouds(proj_root=project_dir)

    # get stats
    log = log_dir / "log.json"
    with open(str(log), "r") as f:
        run_log = json.load(f)

    # print static map stats and render it
    base_pc = read_colmap_points3D(base_pc_path)
    render_point_clouds(base_pc)

    # grey out static points
    if grey_static:
        base_pc[:, 3:] = [192, 192, 192]

    # iterate through and render each iteration
    points_cloud_dir = project_dir / "incremental" / "point_clouds"
    batch = 0
    for iteration in sorted(os.listdir(points_cloud_dir)):
        pc_path = Path.joinpath(points_cloud_dir, iteration, "points3D.npy")
        new_pc = np.load(pc_path)
        detection_path = Path.joinpath(points_cloud_dir, iteration, "detections.json")
        detections = []
        if os.path.exists(detection_path):
            detections = load_detection_jsons(detection_path)

        print_batch_summary(run_log["batches"][batch])
        if detections[0].get("msg", None) != "detection skipped due to lack of valid points":
            print_detection_summary(detections[0])
        batch += 1

        if display_with_static:
            render_point_clouds(new_pc, base_pc, bbox_jsons=detections)
        else:
            render_point_clouds(new_pc, bbox_jsons=detections)

        # hardcoded examples
        if display_matches:
            visualize_pair(
                db_path=str(project_dir / "database.db"),
                images_root=str(data_path / "images"),
                img1_name="stream1/frame_0001.png",
                img2_name="stream2/frame_0011.png",
            )
        if display_tracks:
            visualize_tracks(
                data_path=data_path,
                db_path=str(project_dir / "database.db"),
                points3D_path=Path.joinpath(points_cloud_dir, iteration, "points3D.txt"),
                masks_root=str(data_path / "masks"),
                show_mask=True,
                num_samples=3,
            )

    print_final_summary(run_log)