"""
Various COLMAP I/O utilities used throughout the project.
"""

import os
import re
import shutil
import sqlite3
from utils.util import run_command
import numpy as np

def load_cams_imgs(cameras_path, images_path):
    """Load both cameras and images from COLMAP format into dictionaries."""
    return read_cameras_txt(cameras_path), read_images_txt(images_path)

def load_keypoints(db_path, image_id):
    """
    Loads the keypoints (features) from COMAP database for a specified image.
    :return: the Nx2 array of keypoints as x,y locations in the images.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT cols, data FROM keypoints WHERE image_id=?", (image_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    num_cols = row[0]
    kpts = np.frombuffer(row[1], dtype=np.float32).reshape(-1, num_cols)
    return kpts[:, :2]

def decode_matches(blob):
    """Read a COLMAP match blob and return it as Nx2 array of [id1, id2]."""
    matches = np.frombuffer(blob, dtype=np.uint32)
    return matches.reshape(-1, 2)

def load_image_name_to_id(db_path):
    """Get the dictionary image_name -> image_id for all images in COMAP database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images")
    data = {name: image_id for image_id, name in cur.fetchall()}
    conn.close()
    return data

def image_ids_to_pair_id(image_id1, image_id2):
    """Get the pair id for two images in COMAP database."""
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * 2147483647 + image_id2

def load_matches(db_path, id1, id2, use_two_view_table=True):
    """
    Get the keypoint ids for matches between two images in COMAP database.
    :param use_two_view_table: use two_view_geometries (geo verified matches) table if true otherwise matches table.
    :return: Nx2 array of keypoint ids like [img1_kpt_id, img2_kpt_id]
    """
    table = "two_view_geometries" if use_two_view_table else "matches"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    pair_id = image_ids_to_pair_id(id1, id2)

    # cur.execute("""
    #     SELECT data FROM two_view_geometries
    #     WHERE pair_id=?
    # """, (pair_id,))
    cur.execute(f"""
                SELECT data
                FROM {table}
                WHERE pair_id = ?
                """, (pair_id,))

    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    data = np.frombuffer(row[0], dtype=np.uint32).reshape(-1, 2)
    return data

def pair_id_to_image_ids(pair_id):
    """Get the image ids from a COLMAP pair id."""
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2

def read_cameras_txt(path):
    """
    Read in a COLMAP cameras.txt file.
    :return: dictionary camera_id -> camera_fields (model, width, height, params).
    """
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = map(int, parts[2:4])
            params = np.array(list(map(float, parts[4:])))
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras

def read_images_txt(path):
    """
    Read in a COLMAP images.txt file. .
    :return: Dictionary image_id -> image (qvec, tvec, camera_id, name).
    """
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            i += 1
            continue

        parts = line.split()
        img_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])))
        tvec = np.array(list(map(float, parts[5:8])))
        cam_id = int(parts[8])
        name = parts[9]

        images[img_id] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": cam_id,
            "name": name
        }

        i += 2  # skip keypoints line
    return images

def load_image_id_to_name(db_path):
    """Get the map img_id -> img_name from COLMAP db."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT image_id, name FROM images")
    rows = cur.fetchall()

    conn.close()
    return {img_id: name for img_id, name in rows}

def load_points3D(path):
    """Load points from COLMAP points3D file as a list of dictionaries for each point. Includes all
    fields of the points."""
    points = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or len(line.strip()) == 0:
                continue

            parts = line.split()

            point_id = int(parts[0])
            xyz = list(map(float, parts[1:4]))
            rgb = list(map(int, parts[4:7]))
            error = float(parts[7])

            # track starts after index 8
            track = list(map(int, parts[8:]))

            # pair list: [(image_id, point2D_idx), ...]
            track_pairs = [
                (track[i], track[i + 1])
                for i in range(0, len(track), 2)
            ]

            points.append({
                "id": point_id,
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track_pairs
            })

    return points

def copy_model_with_new_points(base_model, new_model, new_points_paths, set_all_grey=None):
    """
    Copies the base model files (cameras, images, etc.) into the new model directory. Merges base model points3D.txt
    with all other points3D files.
    :param new_points_paths: Path to each new points3D file.
    :param set_all_grey: specify List[str] of points3D paths to set grey for all point colors.
    """
    shutil.copytree(base_model, new_model)
    merge_colmap_points(
        input_paths=new_points_paths,
        output_path=os.path.join(new_model, "points3D.txt"),
        grey_paths=set_all_grey
    )

def read_image_poses(model_dir):
    """
    Reads COLMAP images.txt and returns:
    {image_name: (qvec, tvec)}
    """
    poses = {}

    path = os.path.join(model_dir, "images.txt")

    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            line = line.strip()

            # skip comments / empty
            if not line or line.startswith("#"):
                continue

            vals = line.split()

            # must be header line
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            if len(vals) < 10:
                continue

            try:
                image_name = vals[-1]
                qvec = np.array(vals[1:5], dtype=np.float32)
                tvec = np.array(vals[5:8], dtype=np.float32)

                poses[image_name] = (qvec, tvec)
            except:
                # skip malformed lines
                pass

            # 🔑 IMPORTANT: skip the next line (POINTS2D)
            f.readline()

    return poses

def get_counts_from_header(model_path):
    """
    Reads the first few lines of COLMAP text files to extract 'Number of points' and 'Number of images'.
    """
    num_pts = 0
    num_imgs = 0

    pts_file = os.path.join(model_path, "points3D.txt")
    img_file = os.path.join(model_path, "images.txt")

    # Extract Point Count
    if os.path.exists(pts_file):
        with open(pts_file, "r") as f:
            for _ in range(5):  # Usually on line 3 or 4
                l = f.readline()
                if "Number of points:" in l:
                    # Regex to find the integer following the label
                    match = re.search(r"Number of points:\s+(\d+)", l)
                    if match:
                        num_pts = int(match.group(1))
                    break

    # Extract Image Count
    if os.path.exists(img_file):
        with open(img_file, "r") as f:
            for _ in range(5):
                l = f.readline()
                if "Number of images:" in l:
                    match = re.search(r"Number of images:\s+(\d+)", l)
                    if match:
                        num_imgs = int(match.group(1))
                    break

    return num_pts, num_imgs

def conversion_bin_to_txt(colmap, in_model, out_model):
    """Convert binary COLMAP in_model to txt format in out_model directory.
    Cleans and recreates out_model dir if it already exists."""
    if os.path.exists(out_model):
        shutil.rmtree(out_model)
    os.makedirs(out_model)
    run_command(
        f"{colmap} model_converter "
        f"--input_path {in_model} "
        f"--output_path {out_model} "
        f"--output_type TXT",
        "Model conversion failed"
    )

def read_colmap_points3D(points3d_path):
    """
    Reads COLMAP points3D.txt

    Returns:
        (N, 6) array → [x, y, z, r, g, b]
    """
    points = []

    with open(points3d_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()

            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])

            points.append([x, y, z, r, g, b])

    return np.array(points)

def read_points3D_slim(points3d_path):
    """
    Reads a COLMAP points3D.txt file and extracts:
      - point_ids: (N,) array of point IDs
      - xyz: (N, 3) array of 3D coordinates

    Returns:
        (point_ids, xyz)
    """

    point_ids = []
    xyz_list = []

    with open(points3d_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments / empty lines
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Expected format:
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            if len(parts) < 4:
                continue  # skip malformed lines

            try:
                pid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError:
                continue  # skip malformed entries

            point_ids.append(pid)
            xyz_list.append([x, y, z])

    point_ids = np.array(point_ids, dtype=np.int64)
    xyz = np.array(xyz_list, dtype=np.float64)

    return point_ids, xyz

def merge_colmap_points(
    input_paths,
    output_path,
    grey_paths=None  # list of paths to grey out
):
    """
    Merge all the points3D files specified in input_paths into a single points3D.txt file that saves
    to output_path.
    """
    merged = []

    grey_paths = set(grey_paths or [])

    for path in input_paths:
        pts = read_colmap_points3D_full(path)

        if not pts:
            continue

        for p in pts:
            if path in grey_paths:
                p["rgb"] = [192, 192, 192]

            merged.append(p)

    write_colmap_points3D_structured(merged, output_path)

    print(f"Merged {len(input_paths)} models")
    print(f"Total points: {len(merged)}")
    print(f"Saved to: {output_path}")

def read_colmap_points3D_full(points3d_path):
    """
    Read a points3D.txt file and return as a list of dictionaries.
    Dictionary contains:
        xyz => [x, y, z]
        rgb => [r, g, b]
        error => mean reprojection error
        track => Nx2 [img_id, kpt_id]
    """
    points = []

    with open(points3d_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()

            xyz = list(map(float, parts[1:4]))
            rgb = list(map(int, parts[4:7]))
            error = float(parts[7])

            track = list(map(int, parts[8:]))  # keep raw

            points.append({
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track
            })

    return points

def write_colmap_points3D_structured(points, output_path):
    """
    Write a points3Ds dictionary to COLMAP points3D.txt format at output_path.
    """
    with open(output_path, "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK\n")

        for i, p in enumerate(points):
            xyz = p["xyz"]
            rgb = p["rgb"]
            error = p["error"]
            track = p["track"]

            line = (
                f"{i} "
                f"{xyz[0]} {xyz[1]} {xyz[2]} "
                f"{rgb[0]} {rgb[1]} {rgb[2]} "
                f"{error} "
                + " ".join(map(str, track))
                + "\n"
            )

            f.write(line)

def write_colmap_points3D(points_xyzrgb, output_path):
    """Writes an N,6 array of points to COLMAP format ignoring error and tracks."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points_xyzrgb)}, mean track length: 0.0\n")

        for i, p in enumerate(points_xyzrgb, start=1):
            x, y, z, r, g, b = p
            f.write(f"{i} {x} {y} {z} {int(r)} {int(g)} {int(b)} 0.0\n")

# def apply_ground_transform_to_images(images_path, R, z_offset):
#     t_world = np.array([0.0, 0.0, -z_offset])
#
#     new_lines = []
#
#     with open(images_path, "r") as f:
#         for line in f:
#             line = line.strip()
#
#             if not line or line.startswith("#"):
#                 new_lines.append(line + "\n")
#                 continue
#
#             vals = line.split()
#
#             if len(vals) < 10:
#                 new_lines.append(line + "\n")
#                 continue
#
#             q = np.array(list(map(float, vals[1:5])))
#             t = np.array(list(map(float, vals[5:8])))
#
#             R_cw = qvec_to_rotmat(q)
#
#             # --- camera center ---
#             C = -R_cw.T @ t
#
#             # --- apply SAME transform as points ---
#             C_new = C @ R.T + t_world
#
#             # --- rotate orientation (THIS IS THE FIX) ---
#             R_cw_new = R_cw @ R
#
#             # --- recompute translation ---
#             t_new = -R_cw_new @ C_new
#
#             q_new = rotmat_to_qvec(R_cw_new)
#
#             vals[1:5] = list(map(str, q_new))
#             vals[5:8] = list(map(str, t_new))
#
#             new_lines.append(" ".join(vals) + "\n")
#
#     os.remove(images_path)
#     print("deleted images.txt")
#
#     with open(images_path, "w") as f:
#         f.writelines(new_lines)