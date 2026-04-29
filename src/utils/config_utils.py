import yaml
import os
from copy import deepcopy

def load_config(config_path: str, DEFAULT_CONFIG: dict) -> dict:
    if not os.path.exists(config_path):
        print(f"[WARN] Config not found at {config_path}, using defaults.")
        return deepcopy(DEFAULT_CONFIG)

    with open(config_path, "r") as f:
        user_cfg = yaml.safe_load(f)

    return merge_dicts(DEFAULT_CONFIG, user_cfg)

def load_raw_config(config_path: str, ) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[WARN] Config not found at {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def merge_dicts(base, override):
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result:
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def save_config_copy(config, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "config_used.yaml")

    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"[INFO] Saved config to {path}")