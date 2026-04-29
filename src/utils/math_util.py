"""
Various math utils used throughout project
"""
import numpy as np

def triangulate_dlt(Ps, pts):
    A = []
    for P, (x, y) in zip(Ps, pts):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]

def get_intrinsics(camera):
    """
    Input: colmap camera model
    Return camera intrinsics matrix and distortion coefficients
    """
    model = camera["model"]
    p = camera["params"]

    if model in ["SIMPLE_PINHOLE"]:
        f, cx, cy = p
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]])
        dist = None
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        dist = np.array([k])
    elif model in ["PINHOLE"]:
        fx, fy, cx, cy = p
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        dist = None
    else:
        # fallback: assume pinhole first 4 params
        fx, fy, cx, cy = p[:4]
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        dist = None

    return K, dist

def get_projection_matrix(image, K):
    R = qvec_to_rotmat(image["qvec"])
    t = image["tvec"].reshape(3, 1)
    return K @ np.hstack((R, t))

def reprojection_error(P, X, pt):
    X_h = np.append(X, 1.0)
    proj = P @ X_h
    proj = proj[:2] / proj[2]
    return np.linalg.norm(proj - pt)

def is_in_front_of_camera(X, cam_from_world):
    # Transform point into camera coordinates
    X_cam = cam_from_world * X
    return X_cam[2] > 0  # positive Z = in front

def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [
            1 - 2 * qy ** 2 - 2 * qz ** 2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw
        ],
        [
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx ** 2 - 2 * qz ** 2,
            2 * qy * qz - 2 * qx * qw
        ],
        [
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx ** 2 - 2 * qy ** 2
        ]
    ])

def projection_matrix(K, R, t):
    Rt = np.hstack((R, t.reshape(3, 1)))
    return K @ Rt

def triangulate_point(P1, P2, pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2

    A = np.array([
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
        x2 * P2[2] - P2[0],
        y2 * P2[2] - P2[1],
    ])

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]

def camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R = qvec_to_rotmat(qvec)
    return -R.T @ tvec

def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)

    trace = np.trace(R)

    if trace > 0:
        t = np.sqrt(1.0 + trace)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (R[2, 1] - R[1, 2]) * t
        q[2] = (R[0, 2] - R[2, 0]) * t
        q[3] = (R[1, 0] - R[0, 1]) * t
    else:
        # Find largest diagonal element
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])

        if i == 0:
            t = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[1] = 0.5 * t
            t = 0.5 / t
            q[0] = (R[2, 1] - R[1, 2]) * t
            q[2] = (R[0, 1] + R[1, 0]) * t
            q[3] = (R[0, 2] + R[2, 0]) * t

        elif i == 1:
            t = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[2] = 0.5 * t
            t = 0.5 / t
            q[0] = (R[0, 2] - R[2, 0]) * t
            q[1] = (R[0, 1] + R[1, 0]) * t
            q[3] = (R[1, 2] + R[2, 1]) * t

        else:
            t = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = 0.5 * t
            t = 0.5 / t
            q[0] = (R[1, 0] - R[0, 1]) * t
            q[1] = (R[0, 2] + R[2, 0]) * t
            q[2] = (R[1, 2] + R[2, 1]) * t

    # Normalize (important for numerical stability)
    return q / np.linalg.norm(q)