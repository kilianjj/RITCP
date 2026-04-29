from enum import Enum
from src.ritcp.server.online.triangulation.custom_triangulator import CustomTriangulator
from src.ritcp.server.online.triangulation.colmap_triangulation import ColmapTriangulator

class TriangulatorType(Enum):
    COLMAP = 0
    CUSTOM = 1

    @staticmethod
    def get_triangulator(triangulator_type):
        if triangulator_type == TriangulatorType.COLMAP:
            return ColmapTriangulator()
        elif triangulator_type == TriangulatorType.CUSTOM:
            return CustomTriangulator()
        else:
            raise NotImplementedError