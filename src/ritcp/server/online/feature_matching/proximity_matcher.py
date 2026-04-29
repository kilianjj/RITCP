import time
from abc import ABC, abstractmethod
from io import TextIOWrapper
from typing import List, Tuple
import numpy as np

class ProximityMatcher(ABC):
    """
    A base class that represents any general proximity matching method. Enforces that any subclass can:
    - load in the images to a data structure along with their locations in the world
    - get the images that match a given spatial query
    - write the set of matches to file format to be consumed by colmap matches_importer
    """

    def __init__(self, images_txt_path: str):
        super().__init__()
        self.images_path = images_txt_path
        self.image_poses = None  # subclasses use their own structures as needed

    @abstractmethod
    def _load_images(self):
        pass

    @abstractmethod
    def add_image(self, image_name: str, tvec: np.ndarray, qvec: np.ndarray):
        pass

    def get_match_names(
            self,
            query_tvec: np.ndarray,
            query_qvec: np.ndarray,
            max_neighbors: int = 10,
            max_distance: float = None,
    ) -> List[str]:
        matches = self.get_matches(query_tvec, query_qvec, max_neighbors, max_distance)
        return [t[0] for t in matches]

    @abstractmethod
    def get_matches(
        self,
        query_tvec: np.ndarray,
        query_qvec: np.ndarray,
        max_neighbors: int = 10,
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
        pass

    def write_matches(self, image_name: str, tvec: np.ndarray, qvec: np.ndarray, matches_path: str | TextIOWrapper, clear_existing, max_neighbors: int = 10,
        max_distance: float = None,
        max_angle_deg: float = None,
        ):
        """
        Query the geometrically closest images to a given tvec and qvec.
        Write the image pairs to the matches.txt file at matches_path location.
        Set max number of neighbors to be written and max distance threshold to consider.
        :param image_name: the file name of the current image
        :param tvec: tvec estimate for current image
        :param qvec: qvec estimate for current image
        :param matches_path: the output path to write results to
        :param clear_existing: should it erase existing contents if any?
        """
        start = time.time()
        matches = self.get_matches(tvec, qvec, max_neighbors, max_distance, max_angle_deg)
        print("=" * 60)
        print(f"{len(matches)} kd-tree image matching pairs found in {time.time() - start} seconds")
        print("=" * 60)
        if type(matches_path) == str:
            with open(matches_path, "w") as f:
                for match in matches:
                    f.write(f"{image_name} {match[0]} \n")
        else:
            if clear_existing:
                matches_path.truncate(0)
                matches_path.seek(0)
            matches_path.writelines(f"{image_name} {match[0]} \n" for match in matches)

    def write_matches_list(self, matches: dict[str, List[Tuple[str, float]]], path: str):
        """
        Write existing list of matches to file (come from dictionary of image_name -> matches)
        """
        with open(path, "w") as f:
            for img in matches:
                for pair in matches[img]:
                    f.write(f"{img} {pair[0]} \n")

    @staticmethod
    def _quat_angle_diff(q1: np.ndarray, q2: np.ndarray) -> float:
        # Normalize
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        # Quaternion dot product
        dot = np.abs(np.dot(q1, q2))
        dot = np.clip(dot, -1.0, 1.0)

        # angle = 2 * arccos(dot)
        angle_rad = 2 * np.arccos(dot)
        angle_deg = np.degrees(angle_rad)

        return angle_deg