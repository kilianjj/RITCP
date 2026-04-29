"""
Build a COLMAP base (static) reconstruction from a set of static images.

This script initializes a COLMAP database and reconstructs a sparse 3D model
using only images in the `images/static/` directory. It is intended to create
a fixed reference model of the scene before incremental or dynamic processing.

Command line usage:
    python build_static_model.py \
        --name <project_name> \
        --data_path <path_to_data_dir> \
        [--colmap_executable <path_to_colmap>]

Arguments:
    - name: Name used to create output directory ([name]_sparse)
    - data_path: Path to dataset root (contains images/)
    - colmap_executable: Optional path to COLMAP binary (default: "colmap")

Expected input structure:
    [data_path]/
        images/
            new/
                ...
            static/
                image_*.jpg / .png / .jpeg
        build_static_model.py

Processing steps:
1) Generate an image list restricted to static images
2) Run COLMAP feature extraction (single shared camera intrinsics)
3) Run sequential feature matching
4) Run COLMAP mapper to build sparse model
5) Convert model from BIN → TXT format
6) Remove intermediate .bin files

Output structure:
    [data_path]/
        images/
            new/
            static/
        [name]_sparse/
            database.db
            base/
                0/
                    cameras.txt
                    images.txt
                    points3D.txt
                    rigs.txt
                    frames.txt

Notes:
- Assumes all static images share the same camera intrinsics
- Fails if output model directory already exists
- Only processes images under images/static/
- Designed as a one-time initialization step before incremental SfM
"""

import os
from argparse import ArgumentParser
from utils.util import run_command, remove_bin_files
from pathlib import Path
import time
import shutil

# ============================
# Argument parsing
# ============================

parser = ArgumentParser("COLMAP Base Model Builder")
parser.add_argument("--name", required=True, help="Project name")
parser.add_argument("--data_path", required=True, help="Path to data directory")
parser.add_argument("--colmap_executable", default="colmap", help="Path to COLMAP executable")
args = parser.parse_args()

colmap = f'"{args.colmap_executable}"'

# ============================
# Directory structure
# ============================
data_dir = os.path.dirname(args.data_path)

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory {data_dir} does not exist")

images_dir = os.path.join(data_dir, "images")
static_dir = os.path.join(images_dir, "static")

if not os.path.exists(static_dir):
    raise FileNotFoundError(f"No static images to build model found")

project_sparse_dir = os.path.join(data_dir, f"{args.name}_sparse")
database_path = os.path.join(project_sparse_dir, "database.db")
base_model_dir = os.path.join(project_sparse_dir, "base")
base_model_out_dir = os.path.join(base_model_dir, "0")

if os.path.exists(base_model_out_dir):
    raise Exception(f"Base model output directory {base_model_out_dir} already exists."
                    f"Delete first if want to recreate it.")

os.makedirs(project_sparse_dir, exist_ok=True)
os.makedirs(base_model_dir, exist_ok=True)

# iterate over the frames in the image dir (png or jpg and put static/[img_name] on each line of static/static_images.txt
# this to ensure the image names in database use static/ prefix
output_file = Path(static_dir, 'static_images.txt')
extensions = {'.png', '.jpg', '.jpeg'}
with output_file.open('w') as f:
    for file in Path(static_dir).iterdir():
        if file.suffix.lower() in extensions:
            f.write(f"static/{file.name}\n")

# parse the actual images we need so the root can be reused
image_list_path = os.path.join(static_dir, "static_images.txt")

with open(image_list_path, "w") as f:
    for img in os.listdir(static_dir):
        f.write(f"static/{img}\n")

pipeline_start = time.time()

def get_largest_model_dir(base):
    subdir = sorted(os.listdir(base))[-1]
    return os.path.join(base, subdir)

# ============================
# Step 1: Feature Extraction (STATIC ONLY)
# ============================

run_command(
    f"{colmap} feature_extractor "
    f"--database_path {database_path} "
    f"--image_path {images_dir} "
    f"--SiftExtraction.max_num_features 4000 "
    f"--ImageReader.single_camera 1 " # assumes all images come from camera with same intrinsics
    f"--image_list_path {image_list_path} ",
    "Feature extraction failed"
)

feature_extractor_duration = time.time() - pipeline_start

# ============================
# Step 2: Feature Matching
# ============================

run_command(
    f"{colmap} sequential_matcher "
    f"--database_path {database_path}",
    "Feature matching failed"
)
feature_match_duration = time.time() - feature_extractor_duration

# ============================
# Step 3: Mapping (Build Base Model)
# ============================

run_command(
    f"{colmap} mapper "
    f"--database_path {database_path} "
    f"--image_path {images_dir} "
    f"--output_path {base_model_dir}",
    "Mapping failed"
)
mapper_duration = time.time() - feature_match_duration

base_model_out_dir = get_largest_model_dir(base_model_dir)

# convert to txt format
run_command(
    f"{colmap} model_converter "
    f"--input_path {base_model_out_dir} "
    f"--output_path {base_model_out_dir} "
    f"--output_type TXT",
    "Model conversion to TXT failed"
)

# cleanup .bin
remove_bin_files(base_model_out_dir)

if os.path.basename(base_model_out_dir) != "0":
    zero_dir = os.path.join(base_model_dir, "0")
    tmp_dir = os.path.join(base_model_dir, "_tmp_0")
    shutil.copytree(base_model_out_dir, tmp_dir)

    if os.path.exists(zero_dir):
        shutil.rmtree(zero_dir)

    os.rename(tmp_dir, zero_dir)

    for subdir in os.listdir(base_model_dir):
        if subdir.split("/")[-1] != "0":
            shutil.rmtree(os.path.join(base_model_dir, subdir))

print(f"\n✅ Base model created successfully.")
print(f"Database: {database_path}")
print(f"Model: {base_model_out_dir}")
print(f"Latency Breakdown:")
print(f"Feature extraction time: {feature_extractor_duration}")
print(f"Feature matching time: {feature_match_duration }")
print(f"Mapper time: {mapper_duration }")
print(f"Total elapsed time: {time.time() - pipeline_start}")