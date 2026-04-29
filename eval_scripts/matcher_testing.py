# create a small test images.txt file and test that the matches actually return the correct results
import numpy as np
from matching.matching_utils import *

# manually kind of testing things out here...

def test_linear():
    matcher = get_matcher(MatcherType.LINEAR_SCAN, "/image_positions.txt")

    neighbors = matcher.get_matches(
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 0]),
        max_neighbors=50,
        max_distance=10,
    )
    print(neighbors)
    print(len(neighbors))

    neighbors = matcher.get_matches(
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 0]),
        max_neighbors=50,
        max_distance=2,
    )
    print(neighbors)
    print(len(neighbors))

    neighbors = matcher.get_matches(
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 0]),
        max_neighbors=6,
        max_distance=10,
    )
    print(neighbors)
    print(len(neighbors))

    neighbors = matcher.get_matches(
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 0]),
        max_neighbors=8,
        max_distance=5,
    )
    print(neighbors)
    print(len(neighbors))

    # new images can be added
    # for i in range(1, 51):
    #     matcher.add_image(f"img_00{50 + i}.png",
    #                       np.array([0, 0, 0]),
    #                       np.array([0, 0, 0, 0]),
    #                       )
    #     matches = matcher.get_matches(
    #         np.array([0, 0, 0]),
    #         np.array([0, 0, 0, 0]),
    #         max_neighbors=i+1,
    #         max_distance=1,
    #     )
    #     print(len(matches), matches[-1][1])


def test_3dkd():
    matcher = get_matcher(MatcherType.KD_TREE_3D, "image_positions.txt")
    neighbors = matcher.get_matches(
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 0]),
        max_neighbors=50,
        max_distance=10,
    )
    print(neighbors)
    print(len(neighbors))

    neighbors = matcher.get_matches(
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 0]),
        max_neighbors=50,
        max_distance=2,
    )
    print(neighbors)
    print(len(neighbors))

    neighbors = matcher.get_matches(
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 0]),
        max_neighbors=6,
        max_distance=10,
    )
    print(neighbors)
    print(len(neighbors))

    neighbors = matcher.get_matches(
        np.array([0, 0, 0]),
        np.array([0, 0, 0, 0]),
        max_neighbors=8,
        max_distance=5,
    )
    print(neighbors)
    print(len(neighbors))

    # test tree rebuilds and queries still work
    # for i in range(1, 51):
    #     matcher.add_image(f"img_00{50 + i}.png",
    #                       np.array([0, 0, 0]),
    #                       np.array([0, 0, 0, 0]),
    #                       )
    #     matches = matcher.get_matches(
    #         np.array([0, 0, 0]),
    #         np.array([0, 0, 0, 0]),
    #         max_neighbors=i+1,
    #         max_distance=1,
    #     )
    #     print(len(matches), matches[-1][1])

test_linear()
test_3dkd()