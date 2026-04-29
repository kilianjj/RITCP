"""
Config class that contains all info needed by either COLMAP or custom triangulator.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class TriangulationConfig:
    # shared field
    db_path: str
    # colmap fields
    image_path: str
    in_model_path: str
    out_model_path: str
    log_path: str
    final_out_dir: str
    # custom fields
    cameras_file: str
    images_file: str
    reprojection_error_thresh: float | None = None
    output_txt: str = "points3D.txt"
    output_npy: str = "points.npy"
    new_image_names: List[str] | None=None
    use_verification: bool = True

    @staticmethod
    def get_default_config_dict():
        return {
            "db_path": "",
            "image_path": "",
            "in_model_path": "",
            "out_model_path": "",
            "log_path": "",
            "final_out_dir": "",
            "cameras_file": "",
            "images_file": "",
            "reprojection_error_thresh": None,
            "output_txt": "points3D.txt",
            "output_npy": "points.npy",
            "new_image_names": None,
            "use_verification": True,
        }

    @staticmethod
    def colmap_config(db_path, image_path, in_model_path, out_model_path, final_out, log_path) -> TriangulationConfig:
        dict_config = TriangulationConfig.get_default_config_dict()
        dict_config["db_path"] = db_path
        dict_config["image_path"] = image_path
        dict_config["in_model_path"] = in_model_path
        dict_config["out_model_path"] = out_model_path
        dict_config["final_out_dir"] = final_out
        dict_config["log_path"] = log_path
        return TriangulationConfig(**dict_config)

    @staticmethod
    def custom_config(db_path: str, cam_file: str, img_file: str,
                      new_image_names: List[str] | None, reproj_err: None | float=None,
                      out_txt: str | None=None, out_npy: str | None=None,
                      use_verification: bool=True) -> TriangulationConfig:
        dict_config = TriangulationConfig.get_default_config_dict()
        dict_config["db_path"] = db_path
        dict_config["cameras_file"] = cam_file
        dict_config["images_file"] = img_file
        dict_config["reprojection_error_thresh"] = reproj_err
        dict_config["new_image_names"] = new_image_names
        dict_config["output_txt"] = out_txt
        dict_config["output_npy"] = out_npy
        dict_config["use_verification"] = use_verification
        return TriangulationConfig(**dict_config)