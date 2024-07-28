from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Dict
from utils import PointRect as pr
from onemetric.cv.utils.iou import box_iou_batch
from ByteTrack.yolox.tracker.byte_tracker import STrack
import cv2
from utils.Consts import TEAM1, TEAM2


@dataclass
class Detection:
    rect: pr.Rect
    class_id: int
    class_name: str
    confidence: float
    tracker_id: Optional[int] = None
    team: Optional[int] = None

    @classmethod
    def __get_team_id(cls, boundaries, roi) -> Optional[int]:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower = boundaries[0][0]
        upper = boundaries[0][1]
        mask = cv2.inRange(hsv, lower, upper)
        num_pixels1 = np.count_nonzero(mask)

        lower = boundaries[1][0]
        upper = boundaries[1][1]
        mask = cv2.inRange(hsv, lower, upper)
        num_pixels2 = np.count_nonzero(mask)

        if num_pixels1 > num_pixels2:
            return TEAM1
        elif num_pixels1 < num_pixels2:
            return TEAM2
        return None

    @classmethod
    def from_results(cls, pred: np.ndarray, names: Dict[int, str], boundaries, frame: np.ndarray):
        result = []
        for x_min, y_min, x_max, y_max, confidence, class_id in pred:
            team = None
            class_id = int(class_id)
            if class_id == 2:
                roi = frame[abs(int(y_min)):abs(int(y_max)), abs(int(x_min)):abs(int(x_max))]
                team = cls.__get_team_id(boundaries, roi)
            result.append(Detection(
                rect=pr.Rect(
                    x=float(x_min),
                    y=float(y_min),
                    width=float(x_max - x_min),
                    height=float(y_max - y_min)
                ),
                class_id=class_id,
                class_name=names[class_id],
                confidence=float(confidence),
                team=team
            ))
        return result


def filter_detections_by_class(detections: List[Detection], class_name: str) -> List[Detection]:
    return [
        detection
        for detection
        in detections
        if detection.class_name == class_name
    ]


# converts List[Detection] into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: List[Detection], with_confidence: bool = True) -> np.ndarray:
    return np.array([
        [
            detection.rect.top_left.x,
            detection.rect.top_left.y,
            detection.rect.bottom_right.x,
            detection.rect.bottom_right.y,
            detection.confidence
        ] if with_confidence else [
            detection.rect.top_left.x,
            detection.rect.top_left.y,
            detection.rect.bottom_right.x,
            detection.rect.bottom_right.y
        ]
        for detection
        in detections
    ], dtype=float)


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
        detections: List[Detection],
        tracks: List[STrack]
) -> List[Detection]:
    detection_boxes = detections2boxes(detections=detections, with_confidence=False)
    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detection_boxes)
    track2detection = np.argmax(iou, axis=1)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            detections[detection_index].tracker_id = tracks[tracker_index].track_id
    return detections
