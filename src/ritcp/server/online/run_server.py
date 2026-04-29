"""
Main program for creating and running the incremental SfM pipeline.
Create an IncrementalSFMEngine instance and register desired image streams.
"""

from src.ritcp.server.online.server import RapidTriangulationEngine, Server

if __name__ == "__main__":
    data = "eval_data"
    proj = "eval5"

    server = Server(
        data_path=data,
        proj_name=proj,
        config_path="../../../../sfm_config.yaml",
    )

    while False:
        engine.run()
       # add clients