from dataclasses import dataclass
from typing import Tuple
import numpy as np
from utils import PointRect as pr
import cv2


@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int

    @property
    def bgr_tuple(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r

    @classmethod
    def from_hex_string(cls, hex_string: str):
        r, g, b = tuple(int(hex_string[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        return Color(r=r, g=g, b=b)


def draw_rect(image: np.ndarray, rect: pr.Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, thickness)
    return image


def draw_filled_rect(image: np.ndarray, rect: pr.Rect, color: Color) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, -1)
    return image


def draw_polygon(image: np.ndarray, countour: np.ndarray, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, thickness)
    return image


def draw_filled_polygon(image: np.ndarray, countour: np.ndarray, color: Color) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, -1)
    return image


def draw_text(image: np.ndarray,
              anchor: pr.Point,
              text: str,
              color: Color,
              font_scale: float = 0.5,
              thickness: int = 2) -> np.ndarray:
    cv2.putText(image,
                text, anchor.int_xy_tuple,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color.bgr_tuple,
                thickness,
                2,
                False)
    return image


def draw_ellipse(image: np.ndarray, rect: pr.Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.ellipse(
        image,
        center=rect.bottom_center.int_xy_tuple,
        axes=(int(rect.width), int(0.35 * rect.width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color.bgr_tuple,
        thickness=thickness,
        lineType=cv2.LINE_4
    )
    return image


def draw_circle(image: np.ndarray, lonlat: tuple, color: Color, radius: int = 5, fill: int = cv2.FILLED):
    cv2.circle(image, lonlat, radius, color.bgr_tuple, fill)
    return image


def draw_line(image: np.ndarray, src_x: int, src_y: int, dest_x: int, dest_y: int, color: Color) -> np.ndarray:
    cv2.line(image, (src_x, src_y), (dest_x, dest_y), color.bgr_tuple)
    return image
