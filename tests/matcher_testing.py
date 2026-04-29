# create a small test images.txt file and test that the matches actually return the correct results
import numpy as np
from src.ritcp.server.online.feature_matching.matching_utils import get_matcher, MatcherType

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

test_3dkd()