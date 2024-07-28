from dataclasses import dataclass
from typing import List
from utils import PointRect as pr, Consts
from utils.DrawUtil import Color, draw_filled_rect, draw_text
from utils.Detection import Detection
import cv2
import numpy as np
from typing import Optional

from utils.PixelMapper import PixelMapper


# text annotator to display tracker_id
@dataclass
class TextAnnotator:
    background_color: Color
    text_color: Color
    text_thickness: int
    pm: PixelMapper
    reserve_background_color: Optional[Color] = None

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            background_color = self.background_color
            if detection.team == Consts.TEAM2:
                background_color = self.reserve_background_color

            # if tracker_id is not assigned skip annotation
            if detection.tracker_id is None:
                continue

            # calculate text dimensions
            size, _ = cv2.getTextSize(
                str(detection.tracker_id),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                thickness=self.text_thickness)
            width, height = size

            # calculate text background position
            center_x, center_y = detection.rect.bottom_center.int_xy_tuple
            x = center_x - width // 2
            y = center_y - height // 2 + 10

            # draw background
            annotated_image = draw_filled_rect(
                image=annotated_image,
                rect=pr.Rect(x=x, y=y, width=width, height=height).pad(padding=1),
                color=background_color)

            # draw text
            annotated_image = draw_text(
                image=annotated_image,
                anchor=pr.Point(x=x, y=y + height),
                text=str(detection.tracker_id),
                color=self.text_color,
                thickness=self.text_thickness)
        return annotated_image

    def annotate_map(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            lonlat = tuple(self.pm.pixel_to_lonlat((int(detection.rect.x), int(detection.rect.y)))[0])
            annotated_image = draw_text(
                image=annotated_image,
                anchor=pr.Point(x=lonlat[0], y=lonlat[1]),
                text=str(detection.tracker_id),
                color=self.text_color,
                thickness=self.text_thickness)
        return annotated_image
