"""
3D visualization utilities for SfM point clouds, detections, and bounding boxes.

- Render multiple point clouds simultaneously
- Color points by semantic label or RGB values
- Overlay axis-aligned bounding boxes from detection outputs
- Load and visualize COLMAP or NumPy-based point clouds
- Inspect incremental SfM reconstruction stages

Possible inputs:
- COLMAP points3D.txt (via helper loader)
- .npy point clouds (Nx3, Nx4, or Nx6 formats)
- Detection JSON outputs with bounding boxes
"""

import numpy as np
import open3d as o3d
from src.ritcp.common.segmentation_constants import TYPE_TO_COLOR, CLASS_TO_COLOR
from src.utils.io_helpers import read_colmap_points3D
import json

def load_detection_jsons(*paths):
    jsons = []
    for p in paths:
        if p is None:
            continue
        with open(p) as f:
            jsons.append(json.load(f))
    return jsons

def create_bbox_geometries(detection_jsons):
    geometries = []

    for data in detection_jsons:
        for class_name, class_data in data.items():
            if class_name in ["segmentation_latency_sec", "detection_latency_sec", "num_classes_detected", "msg"]:
                continue

            color = CLASS_TO_COLOR.get(class_name, [1, 1, 1])

            instances = class_data.get("instances", {})
            for _, box in instances.items():
                geom = bbox_to_lineset(box, color=color)
                geometries.append(geom)

    return geometries

def bbox_to_lineset(box, color=[1, 0, 0]):
    """
    Convert bbox dict → Open3D LineSet
    """
    min_pt = np.array(box["min"])
    max_pt = np.array(box["max"])

    # 8 corners of AABB
    x0, y0, z0 = min_pt
    x1, y1, z1 = max_pt

    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])

    # edges
    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]

    colors = [color for _ in lines]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls

def render_point_clouds(*clouds, bbox_jsons=None):
    geometries = []

    # ─── Point clouds ─────────────────────────────────────
    for cloud in clouds:
        if cloud is None or len(cloud) == 0:
            continue

        cloud = np.asarray(cloud)
        dim = cloud.shape[1]

        points = cloud[:, :3]
        colors = np.zeros_like(points)

        if dim == 3:
            pass

        elif dim == 4:
            labels = cloud[:, 3].astype(int)
            colors_rgb = np.array([
                TYPE_TO_COLOR.get(c, [128, 128, 128]) for c in labels
            ])
            colors = colors_rgb / 255.0

        elif dim >= 6:
            colors = cloud[:, 3:6]
            if colors.max() > 1.0:
                colors = colors / 255.0

        else:
            raise ValueError(f"Unsupported point format {dim}")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        geometries.append(pcd)

    # ─── Bounding boxes ───────────────────────────────────
    if bbox_jsons is not None:
        bbox_geoms = create_bbox_geometries(bbox_jsons)
        geometries.extend(bbox_geoms)

    # ─── Optional grid ────────────────────────────────────
    # geometries.append(create_grid())

    o3d.visualization.draw_geometries(geometries)


def render_point_cloud_file(npy_path):
    data = np.load(npy_path)
    render_point_clouds(data)


def render_point_cloud(points):
    render_point_clouds(points)

def create_grid(size=500, n=100, color=[0.5, 0.5, 0.5]):
    """
    Currently unused (disabled in renderer).
    Keeping for future debugging/reference.
    """
    mesh_box = o3d.geometry.TriangleMesh.create_box(
        width=1000, height=1000, depth=0.01
    )
    mesh_box.translate([-500, -500, 0])
    mesh_box.paint_uniform_color(color)
    return mesh_box

if __name__ == "__main__":
    # pc_new1 = np.load("../car_data/test10_sparse/incremental/point_clouds/iteration1/points3D.npy")
    # pc_new2 = np.load("../car_data/test10_sparse/incremental/point_clouds/iteration2/points3D.npy")
    # pc_static = read_colmap_points3D("../car_data/test9_sparse/base/0/points3D.txt")
    # detections = load_detection_jsons(
    #     "../car_data/test10_sparse/incremental/point_clouds/iteration1/points3D_detections.json",
    #     "../car_data/test10_sparse/incremental/point_clouds/iteration2/points3D_detections.json"
    # )
    #
    # render_point_clouds(
    #     pc_new1,
    #     pc_new2,
    #     pc_static,
    #     bbox_jsons=detections
    # )

    pc1 = np.load("../data/car_data/test3_sparse/incremental/point_clouds/iteration1/points3D.npy")
    pc_static = read_colmap_points3D("../data/car_data/test3_sparse/base/0/points3D.txt")
    # pc_static[:, 3:] = [192, 192, 192]
    # pc1 *= 10
    # pc_static *= 10
    pc_static[:, 3:] = [192, 192, 192]

    json = load_detection_jsons("../car_data/test3_sparse/incremental/point_clouds/iteration1/points3D_detections.json")
    render_point_clouds(pc1, pc_static, bbox_jsons=json)