"""
Custom DLT-based triangulation module for incremental SfM pipelines.

This implementation builds 3D points from multi-view feature tracks using
direct linear triangulation (DLT), with additional validation steps.

- Filters tracks to get points only from the current batch of images
- Applies chirality check to ensure points are in front of cameras
- Rejects points with high reprojection error
- Outputs both .npy and COLMAP-style .txt formats
"""

import time
from typing import Tuple
from src.ritcp.server.online.triangulation.triangulator import Triangulator
from src.ritcp.server.online.triangulation.triangulation_config import TriangulationConfig
from src.utils.io_helpers import *
from src.utils.math_util import reprojection_error, triangulate_dlt, get_intrinsics, get_projection_matrix, qvec_to_rotmat


class CustomTriangulator(Triangulator):
    @staticmethod
    def triangulate(config: TriangulationConfig) -> Tuple[int, float]:
        """
        Triangulate 3D points from multi-view feature tracks.

        Args:
            config (TriangulationConfig): Configuration containing:
                - camera and image paths
                - database path
                - new image names
                - reprojection threshold
                - output paths

        Returns:
            Tuple[int, float]:
                - Number of reconstructed points
                - Execution time (seconds)

        Process:
            1. Load COLMAP cameras and images
            2. Build feature tracks (optionally filtered by verification)
            3. Collect keypoints per image
            4. Triangulate valid tracks using DLT
            5. Filter by:
                - minimum number of new images per track
                - reprojection error threshold
                - chirality constraint
            6. Save results to .npy and .txt formats
        """
        start = time.time()

        cameras, images = load_cams_imgs(
            config.cameras_file,
            config.images_file
        )

        name_to_id = {v["name"]: k for k, v in images.items()}
        new_ids = {name_to_id[n] for n in config.new_image_names if n in name_to_id}

        tracks = Triangulator.build_tracks(config.db_path, config.use_verification, new_ids)

        keypoints_cache = {
            img_id: load_keypoints(config.db_path, img_id)
            for img_id in images
        }

        results = []

        for track in tracks:
            img_ids = [img_id for img_id, _ in track]
            num_new = sum(1 for i in img_ids if i in new_ids)

            # same filtering logic as original function
            if num_new < 2:
                continue
            if num_new < len(track) / 2:
                continue

            Ps, pts = [], []
            valid_track = []

            for img_id, kp_idx in track:
                if img_id not in keypoints_cache:
                    continue

                kpts = keypoints_cache[img_id]
                if kpts is None or kp_idx >= len(kpts):
                    continue

                pt = kpts[kp_idx]
                K, _ = get_intrinsics(cameras[images[img_id]["camera_id"]])
                P = get_projection_matrix(images[img_id], K)

                Ps.append(P)
                pts.append(pt)
                valid_track.append((img_id, kp_idx))

            if len(Ps) < 2:
                continue

            X = triangulate_dlt(Ps, pts)

            if not CustomTriangulator.passes_chirality(X, images, valid_track):
                continue

            errors = [
                reprojection_error(P, X, pt)
                for P, pt in zip(Ps, pts)
            ]

            if (
                    config.reprojection_error_thresh is not None
                    and np.mean(errors) > config.reprojection_error_thresh
            ):
                continue

            results.append({
                "xyz": X,
                "rgb": (0, 255, 0),
                "error": float(np.mean(errors)),
                "track": valid_track
            })

        duration = time.time() - start

        results = np.array(results)

        # ---- Save outputs (unchanged logic) ----
        points = []
        for r in results:
            pt = np.concatenate([r["xyz"], r["rgb"]])
            points.append(pt)

        points = np.asarray(points, dtype=np.float32)
        np.save(config.output_npy, points)

        with open(config.output_txt, "w") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

            mean_track_len = np.mean([len(r["track"]) for r in results]) if len(results) > 0 else 0
            f.write(f"# Number of points: {len(results)}, mean track length: {mean_track_len}\n")

            for i, r in enumerate(results, start=1):
                x, y, z = r["xyz"]
                r_c, g_c, b_c = r["rgb"]
                err = r["error"]

                track_str = " ".join(
                    f"{img_id} {kp_idx}" for img_id, kp_idx in r["track"]
                )

                f.write(
                    f"{i} {x} {y} {z} {r_c} {g_c} {b_c} {err} {track_str}\n"
                )

        return len(results), duration

    @staticmethod
    def passes_chirality(X, images, track, min_positive_ratio=0.75):
        """
        Check if a 3D point is geometrically valid via chirality constraint.

        Args:
            X: 3D point in world coordinates
            images: COLMAP image dictionary (poses included)
            track: list of (image_id, keypoint_idx)
            min_positive_ratio: minimum fraction of cameras where point is in front

        Returns:
            bool: True if point passes chirality test

        Notes:
            A point is valid only if it lies in front of most observing cameras
            after projection into each camera frame.
        """
        positive = 0
        for img_id, _ in track:
            image = images[img_id]
            R = qvec_to_rotmat(image["qvec"])
            t = image["tvec"].reshape(3, 1)

            if (R @ X.reshape(3, 1) + t)[2, 0] > 0:
                positive += 1

        return positive / len(track) >= min_positive_ratio
