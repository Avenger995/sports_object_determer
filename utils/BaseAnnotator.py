import math
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from utils import DrawUtil
from utils.Detection import Detection
from utils.DrawUtil import Color
from utils.Consts import TEAM2, TEAM1
from utils.PixelMapper import PixelMapper


@dataclass
class BaseAnnotator:
    colors: List[DrawUtil.Color]
    thickness: int
    pm: PixelMapper

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = DrawUtil.draw_ellipse(
                image=image,
                rect=detection.rect,
                color=self.__get_color_by_team(detection.team, detection.class_id),
                thickness=self.thickness
            )
        return annotated_image

    def annotate_map(self,
                     image: np.ndarray,
                     detections: List[Detection],
                     radius: int = 5,
                     fill: int = cv2.FILLED) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            lonlat = tuple(self.pm.pixel_to_lonlat((int(detection.rect.x), int(detection.rect.y)))[0])
            lonlat = (math.trunc(lonlat[0]), math.trunc(lonlat[1]))
            annotated_image = DrawUtil.draw_circle(
                image=image,
                lonlat=lonlat,
                color=self.__get_color_by_team(detection.team, detection.class_id),
                radius=radius,
                fill=fill
            )
        return annotated_image

    def __get_color_by_team(self, team, class_id) -> Color:
        if team == TEAM1:
            return self.colors[class_id]

        if team == TEAM2:
            return self.colors[class_id + 1]

        return self.colors[0]
