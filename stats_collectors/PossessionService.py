from dataclasses import dataclass
from utils.DrawUtil import Color
import numpy as np
from typing import List, Optional
from utils.Detection import Detection
from utils.Consts import TEAM2, TEAM1, POSSESSION_POINT_MAIN, POSSESSION_POINT_RESERVE
from utils.DrawUtil import draw_text
import utils.PointRect as pr


@dataclass
class PossessionService:
    color_main: Color
    color_reserved: Color
    frames_total: Optional[int] = 0
    frames_main: Optional[int] = 0
    frames_reserve: Optional[int] = 0
    possession_main: Optional[float] = 0
    possession_reserve: Optional[float] = 0

    def __calculate_frames(self, detections: List[Detection]):
        for detection in detections:
            if detection.team == TEAM1:
                self.frames_main += 1
                self.frames_total += 1
            if detection.team == TEAM2:
                self.frames_reserve += 1
                self.frames_total += 1
        return

    def __calculate_possession(self, detections: List[Detection]):
        if len(detections) == 0:
            return
        self.__calculate_frames(detections)
        if self.frames_total == 0:
            return
        self.possession_main = round(self.frames_main / self.frames_total * 100, 1)
        self.possession_reserve = round(self.frames_reserve / self.frames_total * 100, 1)

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        self.__calculate_possession(detections)
        annotated_image = image.copy()
        annotated_image = draw_text(image=image,
                                    text=f'Team #1: {self.possession_main} %',
                                    anchor=pr.Point(x=POSSESSION_POINT_MAIN[0], y=POSSESSION_POINT_MAIN[1]),
                                    color=self.color_main,
                                    font_scale=0.7)
        annotated_image = draw_text(image=annotated_image,
                                    text=f'Team #2: {self.possession_reserve} %',
                                    anchor=pr.Point(x=POSSESSION_POINT_RESERVE[0], y=POSSESSION_POINT_RESERVE[1]),
                                    color=self.color_reserved,
                                    font_scale=0.7)
        return annotated_image
