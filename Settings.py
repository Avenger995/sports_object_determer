import numpy as np
import ast


class Settings:
    file_path: str
    model_path: str
    source_points: []
    map_points: []
    boundaries: []

    def __init__(self, file_path, model_path, source_points, map_points, boundaries):
        self.file_path = file_path
        self.model_path = model_path
        self.source_points = np.float32(ast.literal_eval(source_points))
        self.map_points = np.float32(ast.literal_eval(map_points))
        self.boundaries = ast.literal_eval(boundaries)
