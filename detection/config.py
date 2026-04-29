DEFAULT_DETECTION_CONFIG = {
    "db_scan": {
        "epsilon": 1.0,
        "min_samples": 20,
        "min_points": 3,
    },
    "outlier_filtering": {
        "enabled": True,
        "num_neighbors": 20,
        "std_ratio": 1.5,
    },
    "bbox": {
        "type": "aa",
    },
    "classes": {
        "include": ["car", "bus", "person", "bike", "animal"],
        "exclude": ["background"],
    },
    "runtime": {
        "verbose": True,
        "measure_latency": True,
    },
}