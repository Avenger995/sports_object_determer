import math
from dataclasses import dataclass
from typing import List, Optional
from utils import PointRect as pr, Consts
from utils.Detection import Detection
import numpy as np
from utils.DrawUtil import Color, draw_polygon, draw_filled_polygon, draw_circle
from utils.PixelMapper import PixelMapper


# calculates coordinates of possession marker
def calculate_marker(anchor: pr.Point) -> np.ndarray:
    x, y = anchor.int_xy_tuple
    return (np.array([
        [x - Consts.MARKER_WIDTH // 2, y - Consts.MARKER_HEIGHT - Consts.MARKER_MARGIN],
        [x, y - Consts.MARKER_MARGIN],
        [x + Consts.MARKER_WIDTH // 2, y - Consts.MARKER_HEIGHT - Consts.MARKER_MARGIN]
    ]))


# draw single possession marker
def draw_marker(image: np.ndarray, anchor: pr.Point, color: Color) -> np.ndarray:
    possession_marker_countour = calculate_marker(anchor=anchor)
    image = draw_filled_polygon(
        image=image,
        countour=possession_marker_countour,
        color=color)
    image = draw_polygon(
        image=image,
        countour=possession_marker_countour,
        color=Consts.MARKER_CONTOUR_COLOR,
        thickness=Consts.MARKER_CONTOUR_THICKNESS)
    return image


# dedicated annotator to draw possession markers on video frames
@dataclass
class MarkerAnnotator:
    color: Color
    pm: PixelMapper
    is_ball: bool = False

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = draw_marker(
                image=image,
                anchor=detection.rect.top_center,
                color=self.color)
        return annotated_image

    def annotate_map(self, image: np.ndarray, detections: List[Detection], radius: int = 6, fill: int = 2) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            if self.is_ball:
                lonlat = tuple(self.pm.pixel_to_lonlat((int(detection.rect.x), int(detection.rect.y-2.5*detection.rect.height)))[0])
            else:
                lonlat = tuple(self.pm.pixel_to_lonlat((int(detection.rect.x), int(detection.rect.y)))[0])
            lonlat = (math.trunc(lonlat[0]), math.trunc(lonlat[1]))
            annotated_image = draw_circle(
                image=image,
                lonlat=lonlat,
                color=self.color,
                radius=radius,
                fill=fill
            )
        return annotated_image


def get_player_in_possession(
        player_detections: List[Detection],
        ball_detections: List[Detection],
        proximity: int) -> Optional[Detection]:
    if len(ball_detections) != 1:
        return None
    ball_detection = ball_detections[0]
    for player_detection in player_detections:
        if player_detection.rect.pad(proximity).contains_point(point=ball_detection.rect.center):
            return player_detection
