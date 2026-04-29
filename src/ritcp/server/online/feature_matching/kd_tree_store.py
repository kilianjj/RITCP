import numpy as np
from sklearn.neighbors import KDTree
from enum import Enum
from src.utils.math_util import camera_center, qvec_to_rotmat

class KDType(Enum):
    NN2D = 1 # 2d matching (x, y)
    NN3D = 2 # 3d matching (x, y, z)
    NN3D_ORIENTATION = 3 # adds orientation measures from R but weighted down (yaw, roll really all needed...)

class KDStore:
    """
    Stores actual KD tree structure, how to query, how to add images, rebuild, etc.
    """

    ORIENTATION_WEIGHT = 0.1

    def __init__(self, kd_type: KDType, rebuild_threshold: int = 100):
        self.kd_type = kd_type
        self.rebuild_threshold = rebuild_threshold
        self.points3d = []
        self.qvecs = []
        self.image_names = []
        self.tree: KDTree = None
        self.new_indices = []

    def _feature_vector(self, idx: int):
        center = self.points3d[idx]
        if self.kd_type == KDType.NN2D:
            return center[:2]
        elif self.kd_type == KDType.NN3D:
            return center
        elif self.kd_type == KDType.NN3D_ORIENTATION:
            R = qvec_to_rotmat(self.qvecs[idx])
            forward = R.T @ np.array([0, 0, 1])  # camera forward direction
            return np.concatenate([center, forward])
        else:
            raise ValueError("Unsupported KDType")

    def _query_feature_vector(self, query_xyz: np.ndarray, query_qvec: np.ndarray):
        if self.kd_type == KDType.NN2D:
            return query_xyz[:2]
        elif self.kd_type == KDType.NN3D:
            return query_xyz
        elif self.kd_type == KDType.NN3D_ORIENTATION:
            R = qvec_to_rotmat(query_qvec)
            forward = R.T @ np.array([0, 0, 1])
            return np.concatenate([query_xyz, forward])
        else:
            raise ValueError("Unsupported KDType")

    def add_image(self, image_name: str, tvec: np.ndarray, qvec: np.ndarray):
        idx = len(self.image_names)
        point = camera_center(qvec, tvec)
        self.points3d.append(point)
        self.qvecs.append(qvec)
        self.image_names.append(image_name)
        self.new_indices.append(idx)
        if len(self.new_indices) >= self.rebuild_threshold:
            self._rebuild_tree()

    def _rebuild_tree(self):
        if len(self.points3d) == 0:
            return
        features = np.vstack([self._feature_vector(i) for i in range(len(self.points3d))])
        self.tree = KDTree(features)
        self.new_indices = []

    def query(self, query_xyz: np.ndarray, query_qvec: np.ndarray, k: int):
        results = []
        query_vec = self._query_feature_vector(query_xyz, query_qvec)
        if self.tree is not None:
            dist, ind = self.tree.query(query_vec.reshape(1, -1),
                        k=min(k, len(self.image_names) - len(self.new_indices))) # cap k to max be # of images total
            for d, i in zip(dist[0], ind[0]):
                results.append((i, d))
        for i in self.new_indices:
            feat = self._feature_vector(i)
            d = np.linalg.norm(query_vec - feat)
            results.append((i, d))
        results = sorted(results, key=lambda x: x[1])[:k]
        return results

    def __len__(self):
        return len(self.image_names)