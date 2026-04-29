from src.ritcp.server.online.ritcp_engine import RapidTriangulationEngine

class Server:
    def __init__(self, data_path, proj_name, config_path):
        self.engine = RapidTriangulationEngine(data_path, proj_name, config_path)