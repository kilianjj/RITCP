import numpy as np
from typing import Dict, List, Tuple
from matching.proximity_matcher import ProximityMatcher
from utils.math_util import camera_center

class DictMatcher(ProximityMatcher):
    """
    Simple neighbor matcher for COLMAP images based on pose proximity.
    Stores images in a Map: image -> pose (tvec, qvec)
    Iterates through all pairs to find closest K matches (3d distance). O(nlogn) basic linear scan and sort strategy.
    """

    def __init__(self, image_path: str):
        super().__init__(image_path)
        self.image_poses: Dict[str, Dict[str, np.ndarray]] = {}
        self._load_images()

    def _load_images(self):
        """
        Parses COLMAP images.txt file and populates self.image_poses.
        """
        with open(self.images_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # skip comments / empty
            if len(line) == 0 or line.startswith("#"):
                continue

            tokens = line.split()
            if not tokens[-1].endswith(".png"):
                continue

            # COLMAP format:
            # IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, IMAGE_NAME
            if len(tokens) < 10:
                continue

            image_id = int(tokens[0])
            qvec = np.array(list(map(float, tokens[1:5])))
            tvec = np.array(list(map(float, tokens[5:8])))
            image_name = tokens[9]

            self.add_image(image_name, tvec, qvec)

        print(f"[NeighborMatcher] Loaded {len(self.image_poses)} images")

    def add_image(self, image_name: str, tvec: np.ndarray, qvec: np.ndarray):
        assert tvec.shape == (3,)
        assert qvec.shape == (4,)
        self.image_poses[image_name] = {"tvec": tvec, "qvec": qvec}

    def get_matches(
        self,
        query_tvec: np.ndarray,
        query_qvec: np.ndarray,
        max_neighbors: int = 50,
        max_distance: float = None,
        max_angle_deg: float = None,
    ) -> List[Tuple[str, float]]:
        """
        Finds nearest images based on translation + rotation similarity.

        Args:
            query_tvec: (3,) camera position
            query_qvec: (4,) quaternion
            max_neighbors: max number of neighbors to return
            max_distance: optional distance threshold
            max_angle_deg: optional rotation threshold (degrees)

        Returns:
            List of (image_name, score) sorted by score (lower is better)
        """
        # extract true 3d position in map coordinates
        query_xyz = camera_center(query_qvec, query_tvec)

        results = []
        for image_name, pose in self.image_poses.items():
            tvec = pose["tvec"]
            qvec = pose["qvec"]
            cand_xyz = camera_center(qvec, tvec)
            # Euclidean distance
            dist = np.linalg.norm(cand_xyz - query_xyz)
            if max_distance is not None and dist > max_distance:
                continue
            results.append((image_name, dist))
        # sort by distance, return only closest k neighbors
        results.sort(key=lambda x: x[1])
        return [(name, score) for name, score in results[:max_neighbors]]
