"""
Build a COLMAP base reconstruction from a set of images.
Uses exhaustive matching as our "ground truth" for reconstruction accuracy.
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
data_dir = args.data_path

images_dir = os.path.join(data_dir, "images")
static_dir = os.path.join(images_dir, "static")

if not os.path.exists(static_dir):
    raise FileNotFoundError(f"No static images to build model found")

project_sparse_dir = os.path.join(data_dir, f"{args.name}_sparse")
database_path = os.path.join(project_sparse_dir, "database.db")
base_model_dir = os.path.join(project_sparse_dir, "base")
base_model_out_dir = os.path.join(base_model_dir, "0")

os.makedirs(project_sparse_dir, exist_ok=True)
os.makedirs(base_model_dir, exist_ok=True)

pipeline_start = time.time()

def get_largest_model_dir(base):
    subdir = sorted(os.listdir(base))[-1]
    return os.path.join(base, subdir)


car1 = [
    "frame_0005.jpg",
    "frame_0006.jpg",
    "frame_0007.jpg",
    "frame_0008.jpg",
    "frame_0009.jpg",
    "frame_0010.jpg",
    "frame_0011.jpg",
    "frame_0012.jpg",
    "frame_0013.jpg",
    "frame_0014.jpg"
]
prefixed_car1 = [f"car1/{frame}" for frame in car1]

car2 = [
    "frame_0001.jpg",
    "frame_0002.jpg",
    "frame_0003.jpg",
    "frame_0004.jpg",
    "frame_0005.jpg",
    "frame_0006.jpg",
    "frame_0007.jpg",
    "frame_0008.jpg",
    "frame_0009.jpg",
    "frame_0010.jpg"
]
prefixed_car2 = [f"car2/{frame}" for frame in car2]

car3 = [
    "frame_0003.jpg",
    "frame_0004.jpg",
    "frame_0005.jpg",
    "frame_0006.jpg",
    "frame_0007.jpg",
    "frame_0008.jpg",
    "frame_0009.jpg",
    "frame_0010.jpg",
    "frame_0011.jpg",
    "frame_0012.jpg"
]

prefixed_car3 = [f"car3/{frame}" for frame in car3]

imgs_list = os.path.join(static_dir, "image_list.txt")
with open(imgs_list, "w") as f:
    f.write("\n".join(prefixed_car1 + prefixed_car2 + prefixed_car3))

# ============================
# Step 1: Feature Extraction
# ============================

start = time.time()
run_command(
    f"{colmap} feature_extractor "
    f"--database_path {database_path} "
    f"--image_path {images_dir} "
    f"--SiftExtraction.max_num_features 4000 "
    f"--ImageReader.single_camera 1 ", # assumes all images come from camera with same intrinsics
    "Feature extraction failed"
)

feature_extractor_duration = time.time() - start

# ============================
# Step 2: Feature Matching
# ============================

start = time.time()
run_command(
    f"{colmap} exhaustive_matcher "
    f"--database_path {database_path}"
    f"--image_list_path {imgs_list}",
    "Feature matching failed"
)
feature_match_duration = time.time() - start

# ============================
# Step 3: Mapping (Build Base Model)
# ============================

start = time.time()
run_command(
    f"{colmap} mapper "
    f"--database_path {database_path} "
    f"--image_path {images_dir} "
    f"--output_path {base_model_dir}",
    "Mapping failed"
)
mapper_duration = time.time() - start

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
# print(f"Feature extraction time: {feature_extractor_duration}")
print(f"Feature matching time: {feature_match_duration }")
print(f"Mapper time: {mapper_duration }")
print(f"Total elapsed time: {time.time() - pipeline_start}")