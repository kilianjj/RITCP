DEFAULT_SFM_CONFIG = {
    "sfm": {
        "colmap": "colmap",
        "batch_size": 1,
        "use_gpu": False,
        "gpu_idx": 0,
        "matcher": "kd-3d",
        "triangulator": "custom",
    },
    "feature_extraction": {
        "sift_max_features": 4000,
        "single_camera": True,
    },
    "matching": {
        "max_neighbors": 10,
        "max_distance": None,
        "colmap_method": "vocab"
    },
    "triangulation": {
        "reprojection_error_thresh": None,
    },
    "verification": {
        "enabled": True,
        "max_num_trials": 10000,
        "min_inlier_ratio": 0.1,
    },
    "detection": {
        "detection_module": "detection.semantic_detection",
        "detection_config": "detection_config.yaml"
    },
    "logging": True,
    "debug": True
}