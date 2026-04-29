"""
Main program for creating and running the RIT pipeline.
"""

from src.ritcp.server.online.ritcp_engine import RapidTriangulationEngine
from src.ritcp.common.client import Client
import os

if __name__ == "__main__":
    data = "eval_data"
    proj = "eval5"

    sfm = RapidTriangulationEngine(
        data_path=data,
        project_name=proj,
        config_path="../../../../sfm_config.yaml",
    )

    # register the directories/streams to process
    for i in range(1, 4):
        sfm.register_stream(
            Client(
                os.path.join(data, "images", f"car{i}")
            )
        )

    sfm.incremental_sfm()