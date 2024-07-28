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
class PreviousPlayerData:
    id: int
    team: int
    x: int
    y: int


@dataclass
class Pass:
    src_x: int
    src_y: int
    dest_x: int
    dest_y: int
    src_id: int
    dest_id: int
    color: Color


@dataclass
class PassCollector:
    pm: PixelMapper
    colors: List[Color]
    passes: List[Pass] = field(default_factory=list)
    previous_player: PreviousPlayerData = None

    def __get_color_by_team(self, team) -> Color:
        if team == Consts.TEAM1:
            return self.colors[0]
        if team == Consts.TEAM2:
            return self.colors[1]
        return self.colors[0]

    def __new_previous_player_data(self, detection: Detection) -> PreviousPlayerData:
        lonlat = tuple(self.pm.pixel_to_lonlat((int(detection.rect.x), int(detection.rect.y)))[0])
        return PreviousPlayerData(
            id=detection.tracker_id,
            team=detection.team,
            x=int(lonlat[0]),
            y=int(lonlat[1]))

    def append(self, player_in_possession_detection: List[Detection]):
        if not player_in_possession_detection:
            return
        if self.previous_player is None:
            self.previous_player = self.__new_previous_player_data(player_in_possession_detection[-1])
            return
        if self.previous_player.id == player_in_possession_detection[-1].tracker_id:
            self.previous_player = self.__new_previous_player_data(player_in_possession_detection[-1])
            return
        if self.previous_player.team != player_in_possession_detection[-1].team:
            self.previous_player = self.__new_previous_player_data(player_in_possession_detection[-1])
            return
        lonlat = tuple(self.pm.pixel_to_lonlat((int(player_in_possession_detection[-1].rect.x),
                                                int(player_in_possession_detection[-1].rect.y)))[0])
        self.passes.append(Pass(
            src_x=self.previous_player.x,
            src_y=self.previous_player.y,
            dest_x=int(lonlat[0]),
            dest_y=int(lonlat[1]),
            src_id=self.previous_player.id,
            dest_id=player_in_possession_detection[-1].tracker_id,
            color=self.__get_color_by_team(self.previous_player.team)
        ))
        self.previous_player = self.__new_previous_player_data(player_in_possession_detection[-1])
        return

    def __draw_adapter(self, passes_pitch, pass_item: Pass):
        passes_pitch = DrawUtil.draw_circle(
            image=passes_pitch,
            lonlat=(math.trunc(pass_item.src_x), math.trunc(pass_item.src_y)),
            color=pass_item.color,
            radius=1)
        passes_pitch = DrawUtil.draw_text(
            image=passes_pitch,
            anchor=pr.Point(x=pass_item.src_x, y=pass_item.src_y),
            text=str(pass_item.src_id),
            color=Color(255, 255, 255),
            thickness=1)
        passes_pitch = DrawUtil.draw_circle(
            image=passes_pitch,
            lonlat=(math.trunc(pass_item.dest_x), math.trunc(pass_item.dest_y)),
            color=pass_item.color,
            radius=3)
        passes_pitch = DrawUtil.draw_text(
            image=passes_pitch,
            anchor=pr.Point(x=pass_item.dest_x, y=pass_item.dest_y),
            text=str(pass_item.dest_id),
            color=Color(255, 255, 255),
            thickness=1)
        passes_pitch = DrawUtil.draw_line(
            image=passes_pitch,
            src_x=pass_item.src_x,
            src_y=pass_item.src_y,
            dest_x=pass_item.dest_x,
            dest_y=pass_item.dest_y,
            color=pass_item.color
        )
        return passes_pitch

    def get_image(self, pitch):
        passes_pitch = copy.deepcopy(pitch)
        for pass_item in self.passes:
            passes_pitch = self.__draw_adapter(passes_pitch, pass_item)
        return passes_pitch
