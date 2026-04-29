"""
Base class for any triangulation approach (colmap or custom).
"""

from abc import ABC, abstractmethod
from typing import Tuple
from triangulation.triangulation_config import TriangulationConfig
import numpy as np
from incremental_sfm.colmap_helpers import pair_id_to_image_ids, decode_matches
from utils.union_find import UnionFind
import sqlite3

class Triangulator(ABC):

    @staticmethod
    @abstractmethod
    def triangulate(config: TriangulationConfig) -> Tuple[np.ndarray, float]:
        """
        Returns number of triangulated points and the duration for the triangulation.
        :param config: TriangulationConfig used
        """
        pass

    @staticmethod
    def build_tracks(db_path, use_verification, new_ids=None):
        uf = UnionFind()

        table = "two_view_geometries" if use_verification else "matches"
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            f"SELECT pair_id, rows, cols, data FROM {table}"
        )

        for pair_id, rows, cols, blob in cursor:
            if blob is None or rows == 0:
                continue

            img1, img2 = pair_id_to_image_ids(pair_id)

            if new_ids is not None:
                # ensure the match comes from two new images
                if img1 not in new_ids and img2 not in new_ids:
                    continue

            matches = decode_matches(blob)

            for i1, i2 in matches:
                uf.union((img1, i1), (img2, i2))

        tracks = {}
        for key in uf.parent:
            root = uf.find(key)
            tracks.setdefault(root, []).append(key)

        return list(tracks.values())