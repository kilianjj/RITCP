"""
Computes and saves the 3d transformation needed to align the ground plane of the static map with coordinate axis plane.

todo: could apply along with scaling at the same time
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor
from incremental_sfm.colmap_helpers import read_points3D_slim

def align_to_ground(points, thresh):
    """
    points: (n, 3) numpy array
    """
    # --- Step 1: RANSAC to find the plane ---
    # We treat Z as a function of X and Y: z = ax + by + d
    X_f = points[:, :2]
    z_f = points[:, 2]

    ransac = RANSACRegressor(residual_threshold=thresh)  # Adjust threshold based on noise
    ransac.fit(X_f, z_f)

    # Plane equation: z = a*x + b*y + d  =>  ax + by - z + d = 0
    a, b = ransac.estimator_.coef_
    d = ransac.estimator_.intercept_

    # Normal vector of the fitted plane
    normal = np.array([a, b, -1])
    normal /= np.linalg.norm(normal)

    # Ensure the normal points "up" (positive Z)
    if normal[2] < 0:
        normal = -normal

    # --- Step 2: Calculate Rotation ---
    # Target vector is the world Z-axis
    target_z = np.array([0, 0, 1])

    # Rodrigues' rotation formula components
    v = np.cross(normal, target_z)
    s = np.linalg.norm(v)
    c = np.dot(normal, target_z)

    if s == 0:
        # Already aligned
        rotation_matrix = np.eye(3)
    else:
        # Skew-symmetric cross-product matrix
        v_skew = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1 - c) / (s ** 2))

    # --- Step 3: Apply Transformation ---
    # Rotate all points
    rotated_points = points @ rotation_matrix.T

    # Vertical Translation: Find the new median Z of inliers to ground it
    # This is more robust than using 'd' directly after rotation
    inlier_mask = ransac.inlier_mask_
    z_offset = np.median(rotated_points[inlier_mask, 2])

    return rotation_matrix, z_offset

if __name__ == "__main__":
    model_path = "../data/car_data/static_sparse/base/0/"
    pids, xyzs = read_points3D_slim(model_path + "points3D.txt")
    R, z_offset = align_to_ground(xyzs, thresh=1)

    f_name = f"{model_path}alignment_transform.npz"
    np.savez(f_name, rotation=R, offset=z_offset)

    print(f"""
    Calculated ground alignment transformation for static model. Saved to {f_name}.\n
    R={R}\n
    z_offset={z_offset}\n
    """)
