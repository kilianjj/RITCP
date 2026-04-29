"""
COLMAP triangulation module
"""

import os
import shutil
import time
from triangulation.triangulation_config import TriangulationConfig
from triangulation.triangulator import Triangulator
from pathlib import Path
from incremental_sfm.colmap_helpers import read_colmap_points3D, conversion_bin_to_txt
from typing import Tuple
from utils.util import logged_runner
from incremental_sfm.colmap_helpers import load_image_id_to_name

class ColmapTriangulator(Triangulator):
    @staticmethod
    def triangulate(config: TriangulationConfig) -> Tuple[int, float]:
        """
        Triangulate points with COLMAP's points_triangulator command
        """
        if os.path.exists(config.out_model_path):
            shutil.rmtree(config.out_model_path)
        os.makedirs(config.out_model_path, exist_ok=True)

        run_logged = logged_runner(log_dir=config.log_path, log_file="triangulation_log.txt")
        start = time.time()
        run_logged(
            description="COLMAP Triangulation",
            command=f"colmap point_triangulator "
                f"--database_path {config.db_path} "
                f"--image_path {config.image_path} "
                f"--input_path {config.in_model_path} "
                f"--output_path {config.out_model_path} "
        )
        duration = time.time() - start

        conversion_bin_to_txt(
            "colmap",
            config.out_model_path,
            config.final_out_dir,
        )

        new_points_path = Path(config.final_out_dir).joinpath("points3D.txt")
        points3D = read_colmap_points3D(str(new_points_path))

        return len(points3D), duration

    @staticmethod
    def filter_new_points3D(
            db_path: str,
            new_image_names: list[str],
            input_points3D_path: str,
            output_points3D_path: str,
    ) -> int:
        """
        Filters a COLMAP points3D.txt file to only keep points whose tracks are composed
        entirely of features from the provided new images. Needed because the segmentation segmentation
        in the detection module will take a long time to go through all points from a full reconstruction.

        :param conn: sqlite3 connection to COLMAP database
        :param new_image_names: list of image names (e.g., "vehicle_01/frame_0001.jpg")
        :param input_points3D_path: path to original points3D.txt
        :param output_points3D_path: path to write filtered points3D.txt
        :param use_verification: whether to use two_view_geometries or matches
        """

        # ----------------------------------------
        # Build mapping: image_id -> image_name
        # ----------------------------------------
        id_to_name = load_image_id_to_name(db_path)

        # Convert new image names to ids
        new_ids = {
            img_id for img_id, name in id_to_name.items()
            if name in new_image_names
        }

        # ----------------------------------------
        # Build tracks using your existing logic
        # ----------------------------------------
        tracks = Triangulator.build_tracks(db_path, use_verification=True, new_ids=new_ids)

        # Map (image_id, point2D_idx) -> track_id
        obs_to_track = {}
        for track_id, track in enumerate(tracks):
            for obs in track:
                obs_to_track[obs] = track_id

        # ----------------------------------------
        # Read and filter points3D.txt
        # ----------------------------------------
        with open(input_points3D_path, "r") as f:
            lines = f.readlines()

        output_lines = []
        new_points = 0

        for line in lines:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()

            # COLMAP format:
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] (IMAGE_ID, POINT2D_IDX pairs)
            point_id = int(parts[0])

            # track starts at index 8
            track_data = list(map(int, parts[8:]))

            obs = [(track_data[i], track_data[i + 1]) for i in range(0, len(track_data), 2)]

            # check if all observations belong to tracks formed only from new images
            keep = True
            for (img_id, p2d_idx) in obs:
                if img_id not in new_ids:
                    keep = False
                    break

            if keep:
                output_lines.append(line)
                new_points += 1

        # ----------------------------------------
        # Write filtered output
        # ----------------------------------------
        with open(output_points3D_path, "w") as f:
            f.write("# 3D point list with one line of data per point:")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX")
            f.write(f"# Number of points: {new_points}, mean track length: 11.802110977080821)")
            f.writelines(output_lines)

        return new_points