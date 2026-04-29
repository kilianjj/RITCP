from src.ritcp.server.online.feature_matching.kd_tree_matcher import KdTreeMatcher, KDType
from enum import Enum
from src.ritcp.server.online.feature_matching.proximity_matcher import ProximityMatcher

class MatcherType(Enum):
    KD_TREE_2D = 1
    KD_TREE_3D = 2
    KD_TREE_3D_ORIENTATION = 3

def get_matcher(matcher_type: MatcherType, static_img_path) -> ProximityMatcher:
    if matcher_type == MatcherType.KD_TREE_2D:
        return KdTreeMatcher(static_img_path, KDType.NN2D)
    elif matcher_type == MatcherType.KD_TREE_3D:
        return KdTreeMatcher(static_img_path, KDType.NN3D)
    elif matcher_type == MatcherType.KD_TREE_3D_ORIENTATION:
        return KdTreeMatcher(static_img_path, KDType.NN3D_ORIENTATION)
    raise ValueError("not a valid matcher type")