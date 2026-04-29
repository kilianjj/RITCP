from src.ritcp.server.online.ritcp_engine import RapidTriangulationEngine
import numpy as np

from src.ritcp.server.online.triangulation.custom_triangulator import CustomTriangulator
from src.ritcp.server.online.triangulation.triangulator import Triangulator
from src.ritcp.server.online.triangulation.triangulation_config import TriangulationConfig
import shutil
import os

"""
Run custom triangulation on an existing model to see how it compares.
Don't forget to change COLMAP's rendering settings when visualizing to see points - 
change min_track_number = 0 since don't write tracks to points3D.txt.

"""

if __name__ == "__main__":

    if os.path.exists("demo_model/"):
        shutil.rmtree("demo_model/")
    shutil.copytree("../data/car_data/static_sparse/base/0/", "demo_model/")

    triangulator = CustomTriangulator()
    cfg = TriangulationConfig()
    new3d = triangulator.triangulate(
        cfg
    )
    np.save("demo_model/points.npy", new3d)

    Triangulator.write_points3D_txt(new3d, "demo_model/points3D.txt")