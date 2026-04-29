"""
Scale a COLMAP reconstruction to real-world units.

This script applies a global scale to an SfM model by aligning it to
scaled camera positions derived from a known real-world distance.

Workflow:
1) Compute scale factor = (real distance) / (model distance)
2) Multiply camera translation vectors (from images.txt) by this factor
3) Save scaled camera positions as image_world_positions.txt
4) Run COLMAP model_aligner to transform the full model
5) Output a scaled version of the model (TXT format)

Behavior:
- Original model is moved to: <parent>/unscaled_model
- Scaled model is written back to the original model_path
- Skips recomputing image_world_positions.txt if it already exists

Command line usage:
    python scale_model.py \
        --model_path <path_to_model_dir> \
        --scale_factor <float> \
        [--gps 0|1] \
        [--max_error <float>] \
        [--colmap_executable <path>]

Arguments:
- model_path: Path to COLMAP model directory (contains images.txt, etc.)
- scale_factor: Scaling factor (world / model distance)
- gps: Whether reference positions are GPS (0 = no, 1 = yes)
- max_error: RANSAC alignment threshold
- colmap_executable: Path to COLMAP binary (default: "colmap")

Notes:
- Requires accurate reference distance for correct scaling
- Uses COLMAP’s model_aligner for robust transformation
"""

import os
import time
import numpy as np
from utils.util import run_command, remove_bin_files
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser("COLMAP Model Aligner")
parser.add_argument("--model_path", required=True, help="Path to model directory")
parser.add_argument("--scale_factor", required=True, help="Model->World scaling factor")
parser.add_argument("--gps", default=0, help='Use GPS location (0) or cartesian (1)')
parser.add_argument("--max_error", default=0.1, help='Max error for alignment')
parser.add_argument("--colmap_executable", default="colmap", help="Path to COLMAP executable")
args = parser.parse_args()

colmap = f'"{args.colmap_executable}"'
scaling_factor = float(args.scale_factor)
use_gps = int(args.gps)
max_error = float(args.max_error)

model_dir = Path(args.model_path)
img_data = model_dir / "images.txt"
img_world_positions = model_dir / "image_world_positions.txt"
unscaled_path = model_dir.parent / 'unscaled_model'

print('Scaling points to world units')
print(f'Scaling factor: {scaling_factor}')

start = time.time()

# Using the scaling factor, determine the image locations in the world based on map_coord * scaling_factor
# Save for later
with open(img_world_positions, 'w') as out:
    with open(img_data, 'r') as f:
        for line in f:
            vals = line.split()
            if vals[-1].endswith('.png') or vals[-1].endswith('.jpg'):
                out_vals = [vals[-1]]
                x = float(vals[5])
                y = float(vals[6])
                z = float(vals[7])
                point = np.array([x, y, z])
                point = point * scaling_factor
                out_vals.extend(point)
                out.write(' '.join(map(str, out_vals)))
                out.write('\n')
os.rename(model_dir, unscaled_path)
print('Generated image world positions')

os.makedirs(model_dir, exist_ok=True)
img_world_positions = unscaled_path / "image_world_positions.txt"

# use COLMAP's model_aligner to do the transformation (applies to all aspects of the model)
run_command(f'{colmap} model_aligner \
    --input_path {unscaled_path} \
    --output_path {model_dir} \
    --ref_images_path {img_world_positions} \
    --ref_is_gps {use_gps} \
    --alignment_max_error {max_error}',
            'aligning failed')

# convert to txt format and cleanup
run_command(
    f"{colmap} model_converter "
    f"--input_path {model_dir} "
    f"--output_path {model_dir} "
    f"--output_type TXT",
    "Model conversion failed"
)

remove_bin_files(str(model_dir))

print(f'World scaling complete in {time.time() - start} seconds')
