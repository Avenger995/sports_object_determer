import copy
import math
from typing import List
from dataclasses import dataclass, field
from utils import Consts, DrawUtil
from utils.Detection import Detection
from utils.DrawUtil import Color
from utils.PixelMapper import PixelMapper
from utils import PointRect as pr


@dataclass
class Touch:
    x: int
    y: int
    id: int
    color: Color


@dataclass
class TouchCollector:
    pm: PixelMapper
    colors: List[Color]
    touches: List[Touch] = field(default_factory=list)

    def __get_color_by_team(self, team) -> Color:
        if team == Consts.TEAM1:
            return self.colors[0]
        if team == Consts.TEAM2:
            return self.colors[1]
        return self.colors[0]

    def __get_touch(self, detections: List[Detection]) -> Touch:
        for detection in detections:
            lonlat = tuple(self.pm.pixel_to_lonlat((int(detection.rect.x), int(detection.rect.y)))[0])
            id = detection.tracker_id
            color = self.__get_color_by_team(detection.team)
        return Touch(x=lonlat[0], y=lonlat[1], id=id, color=color)

    def append(self, player_in_possession_detection: List[Detection]):
        if not player_in_possession_detection:
            return
        self.touches.append(self.__get_touch(player_in_possession_detection))

    def __draw_adapter(self, touches_pitch, touch: Touch):
        touches_pitch = DrawUtil.draw_circle(
            image=touches_pitch,
            lonlat=(math.trunc(touch.x), math.trunc(touch.y)),
            color=touch.color)
        touches_pitch = DrawUtil.draw_text(
            image=touches_pitch,
            anchor=pr.Point(x=touch.x, y=touch.y),
            text=str(touch.id),
            color=Color(255, 255, 255),
            thickness=1)
        return touches_pitch

    def get_image(self, pitch):
        touches_pitch = copy.deepcopy(pitch)
        for touch in self.touches:
            touches_pitch = self.__draw_adapter(touches_pitch, touch)
        return touches_pitch
