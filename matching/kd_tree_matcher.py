import time
from matching.kd_tree_store import KDStore, KDType
import os
import numpy as np
from typing import List, Tuple
from matching.proximity_matcher import ProximityMatcher

class KdTreeMatcher(ProximityMatcher):
    """
    Matcher that uses Kd tree for O(log(n)) pose queries to find nearest static image matches.
    Supports the higher-level get_matches API needed by ProximityMatcher.
    The Kd representation maintained by KDStore.
    """

    def __init__(self, image_path: str, kd_type: KDType) -> None:
        super().__init__(image_path)
        self.rep = KDStore(kd_type, rebuild_threshold=50)
        self._load_images()

    def _load_images(self):
        with open(self.images_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            if len(line) == 0 or line.startswith("#"):
                continue

            tokens = line.split()

            if not (
                tokens[-1].endswith(".png")
                or tokens[-1].endswith(".jpg")
                or tokens[-1].endswith(".jpeg")
            ):
                continue

            if len(tokens) < 10:
                continue

            image_id = int(tokens[0])
            qvec = np.array(list(map(float, tokens[1:5])))
            tvec = np.array(list(map(float, tokens[5:8])))
            image_name = tokens[9]

            self.add_image(image_name, tvec, qvec)

        print(f"[KdTreeMatcher] Loaded {len(self.rep)} images")

    def add_image(self, image_name: str, tvec: np.ndarray, qvec: np.ndarray):
        assert tvec.shape == (3,)
        assert qvec.shape == (4,)
        self.rep.add_image(image_name, tvec, qvec)

    def get_matches(
        self,
        query_tvec: np.ndarray,
        query_qvec: np.ndarray,
        max_neighbors: int = 10,
        max_distance: float = None,
        max_angle_deg: float = None,
    ) -> List[Tuple[str, float]]:
        # get matches from the hybrid kd tree container
        start = time.time()
        results = self.rep.query(query_tvec, query_qvec, k=max_neighbors * 2)
        matches = []
        # check each neighbor
        for idx, dist in results:
            if max_distance is not None and dist > max_distance:
                continue

            # todo: could check orientation info or other criteria here
            # if max_angle_deg is not None:
            #     q1 = query_qvec
            #     q2 = self.rep.qvecs[idx]
            #     dot = np.abs(np.dot(q1, q2))
            #     dot = np.clip(dot, -1.0, 1.0)
            #     angle = 2 * np.arccos(dot) * (180.0 / np.pi)
            #     if angle > max_angle_deg:
            #         continue

            matches.append((self.rep.image_names[idx], dist))
            if len(matches) >= max_neighbors:
                break
        print(f"{len(results)} kd-tree image matching pairs found in {time.time() - start} seconds")
        return matches