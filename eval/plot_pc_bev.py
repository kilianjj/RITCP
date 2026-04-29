from src.utils.io_helpers import load_alignment_transform
import numpy as np
from eval.numpy_pc_render import render_point_cloud

# can use the saved alignment transform to display things against the "ground"
# maybe this could be used for 2d db scan or other detection strategies
# hieght could be inferred with the min/max z value within the cluster for instance

if __name__ == '__main__':
    alignment_path = "../data/car_data/static_sparse/base/0/"
    pc_path = "../data/car_data/test5_sparse/incremental/point_clouds/iteration1/points3D.npy"
    rot, offset = load_alignment_transform(alignment_path)

    points = np.load(pc_path)
    points[:, :3] = points[:, :3] @ rot.T
    points[:, 2] -= offset

    # colors = np.array([TYPE_TO_COLOR[x] for x in points[:, 3]]) / 255.0
    # plt.scatter(points[:, 0], points[:, 1], c=colors)
    # plt.xlim([-10, 0])
    # plt.ylim([0, 50])
    # plt.show()

    render_point_cloud(points)
