from incremental_sfm.incremental_sfm_engine import IncrementalSFMEngine
import numpy as np
from triangulation.triangulator import Triangulator
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

    new3d = triangulator.triangulate(
        pose_dict=IncrementalSFMEngine.read_latest_image_poses("../car_data/static_sparse/base/0"),
        image_names=None
    )
    np.save("demo_model/points.npy", new3d)

    Triangulator.write_points3D_txt(new3d, "demo_model/points3D.txt")